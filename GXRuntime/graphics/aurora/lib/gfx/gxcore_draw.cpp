#include "gxcore_draw.hpp"

#include <aurora/aurora.h> // aurora_set_forced_anisotropy

#include "../webgpu/gpu.hpp"
#include "../gx/gx.hpp" // UseReversedZ + set_logical_viewport (substrate glue)
#include "frame_interp.hpp"
#include "pipeline_cache.hpp"
#include "thread_cpu.hpp"
#include "texture.hpp"
#include "tex_copy_conv.hpp" // EFB-copy format conversion (63/S16)
#include "texture_replacement.hpp" // Dolphin-format HD texture packs

#include <gxruntime/gxcore/gxcore.hpp> // EfbCopyCommand
#include <gxruntime/gxcore/texture_decode.hpp>

#include <absl/container/flat_hash_map.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace aurora::gfx::gxcore {

namespace gxc = gxruntime::gxcore;

using webgpu::g_device;
using webgpu::g_graphicsConfig;

static Module Log("aurora::gfx::gxcore");

namespace {

wgpu::CompareFunction to_compare(gxc::CompareMode func) {
  // GC compare flipped for the substrate's reversed-Z (gx/gx.cpp:526 shape).
  switch (func) {
  case gxc::CompareMode::Never:
    return wgpu::CompareFunction::Never;
  case gxc::CompareMode::Less:
    return gx::UseReversedZ ? wgpu::CompareFunction::Greater
                            : wgpu::CompareFunction::Less;
  case gxc::CompareMode::Equal:
    return wgpu::CompareFunction::Equal;
  case gxc::CompareMode::LEqual:
    return gx::UseReversedZ ? wgpu::CompareFunction::GreaterEqual
                            : wgpu::CompareFunction::LessEqual;
  case gxc::CompareMode::Greater:
    return gx::UseReversedZ ? wgpu::CompareFunction::Less
                            : wgpu::CompareFunction::Greater;
  case gxc::CompareMode::NEqual:
    return wgpu::CompareFunction::NotEqual;
  case gxc::CompareMode::GEqual:
    return gx::UseReversedZ ? wgpu::CompareFunction::LessEqual
                            : wgpu::CompareFunction::GreaterEqual;
  case gxc::CompareMode::Always:
  default:
    return wgpu::CompareFunction::Always;
  }
}

wgpu::BlendFactor to_blend_factor_src(gxc::SrcBlendFactor factor,
                                      bool dual_source) {
  switch (factor) {
  case gxc::SrcBlendFactor::Zero:
    return wgpu::BlendFactor::Zero;
  case gxc::SrcBlendFactor::One:
    return wgpu::BlendFactor::One;
  case gxc::SrcBlendFactor::DstClr:
    return wgpu::BlendFactor::Dst;
  case gxc::SrcBlendFactor::InvDstClr:
    return wgpu::BlendFactor::OneMinusDst;
  case gxc::SrcBlendFactor::SrcAlpha:
    return dual_source ? wgpu::BlendFactor::Src1Alpha
                       : wgpu::BlendFactor::SrcAlpha;
  case gxc::SrcBlendFactor::InvSrcAlpha:
    return dual_source ? wgpu::BlendFactor::OneMinusSrc1Alpha
                       : wgpu::BlendFactor::OneMinusSrcAlpha;
  case gxc::SrcBlendFactor::DstAlpha:
    return wgpu::BlendFactor::DstAlpha;
  case gxc::SrcBlendFactor::InvDstAlpha:
  default:
    return wgpu::BlendFactor::OneMinusDstAlpha;
  }
}

wgpu::BlendFactor to_blend_factor_dst(gxc::DstBlendFactor factor,
                                      bool dual_source) {
  switch (factor) {
  case gxc::DstBlendFactor::Zero:
    return wgpu::BlendFactor::Zero;
  case gxc::DstBlendFactor::One:
    return wgpu::BlendFactor::One;
  case gxc::DstBlendFactor::SrcClr:
    return wgpu::BlendFactor::Src;
  case gxc::DstBlendFactor::InvSrcClr:
    return wgpu::BlendFactor::OneMinusSrc;
  case gxc::DstBlendFactor::SrcAlpha:
    return dual_source ? wgpu::BlendFactor::Src1Alpha
                       : wgpu::BlendFactor::SrcAlpha;
  case gxc::DstBlendFactor::InvSrcAlpha:
    return dual_source ? wgpu::BlendFactor::OneMinusSrc1Alpha
                       : wgpu::BlendFactor::OneMinusSrcAlpha;
  case gxc::DstBlendFactor::DstAlpha:
    return wgpu::BlendFactor::DstAlpha;
  case gxc::DstBlendFactor::InvDstAlpha:
  default:
    return wgpu::BlendFactor::OneMinusDstAlpha;
  }
}

// Forced anisotropy (aurora_set_forced_anisotropy): 1 leaves the game's samplers.
std::uint16_t initial_forced_anisotropy() {
  const char* env = std::getenv("DOL_AURORA_FORCE_ANISO");
  const long samples = env != nullptr ? std::strtol(env, nullptr, 10) : 1L;
  return static_cast<std::uint16_t>(std::clamp(samples, 1L, 16L));
}
std::atomic<std::uint16_t> g_forcedAnisotropy{initial_forced_anisotropy()};

wgpu::AddressMode to_address_mode(std::uint8_t wrap) {
  switch (wrap) {
  case 1:
    return wgpu::AddressMode::Repeat;
  case 2:
    return wgpu::AddressMode::MirrorRepeat;
  case 0:
  default:
    return wgpu::AddressMode::ClampToEdge;
  }
}

wgpu::SamplerDescriptor sampler_descriptor(const gxc::PlanSampler& sampler) {
  const bool mipmaps = sampler.mipmap_filter != 0u;
  std::uint16_t maxAnisotropy = 1;
  if (mipmaps && (sampler.max_aniso == 1u || sampler.max_aniso == 2u)) {
    maxAnisotropy = sampler.max_aniso == 1u
                        ? std::max<std::uint16_t>(
                              webgpu::g_graphicsConfig.textureAnisotropy / 2u,
                              1u)
                        : std::max<std::uint16_t>(
                              webgpu::g_graphicsConfig.textureAnisotropy, 1u);
  }
  // The player's forced anisotropy (aurora_set_forced_anisotropy), with
  // Dolphin's rule: every texture except one filtered nearest both ways (pixel
  // art, fonts), all filters linear. Most of Wind Waker's textures have no
  // mips (TX_SETMODE0 filter 4), so the anisotropic taps are what sharpen the
  // ground and the walls at a glancing angle.
  const std::uint16_t forced = g_forcedAnisotropy.load(std::memory_order_relaxed);
  if (forced > 1u && (sampler.min_filter != 0u || sampler.mag_filter != 0u))
    maxAnisotropy = std::max(maxAnisotropy, forced);
  auto magFilter = sampler.mag_filter != 0u ? wgpu::FilterMode::Linear
                                             : wgpu::FilterMode::Nearest;
  auto minFilter = sampler.min_filter != 0u ? wgpu::FilterMode::Linear
                                             : wgpu::FilterMode::Nearest;
  auto mipFilter = sampler.mipmap_filter == 2u
                       ? wgpu::MipmapFilterMode::Linear
                       : wgpu::MipmapFilterMode::Nearest;
  if (maxAnisotropy > 1u) {
    magFilter = wgpu::FilterMode::Linear;
    minFilter = wgpu::FilterMode::Linear;
    mipFilter = wgpu::MipmapFilterMode::Linear;
  }
  return {
      .label = "GXCore Sampler",
      .addressModeU = to_address_mode(sampler.wrap_s),
      .addressModeV = to_address_mode(sampler.wrap_t),
      .addressModeW = wgpu::AddressMode::Repeat,
      .magFilter = magFilter,
      .minFilter = minFilter,
      .mipmapFilter = mipFilter,
      .lodMinClamp = mipmaps ? static_cast<float>(sampler.min_lod) / 16.f : 0.f,
      .lodMaxClamp = mipmaps ? static_cast<float>(sampler.max_lod) / 16.f : 0.f,
      .maxAnisotropy = maxAnisotropy,
  };
}

// Texture bind group layout for a used-texmap set (63/Mfin multi-texmap): texmap
// t occupies binding 2t (texture) + 2t+1 (sampler), matching the WGSL. Cached per
// mask; used_mask=1 (texmap 0 only) reproduces the pre-Mfin single-texmap layout.
wgpu::BindGroupLayout texture_bind_group_layout(uint32_t used_mask = 1u) {
  // Both the pipeline compiler and FIFO submission use this cache. A lookup
  // concurrent with flat_hash_map growth can read an invalid layout handle.
  static std::mutex mutex;
  std::lock_guard lock{mutex};
  static absl::flat_hash_map<uint32_t, wgpu::BindGroupLayout> cache;
  // Retaining the owner also prevents pointer reuse from matching a dead
  // device after renderer reinitialization.
  static wgpu::Device owner;
  if (owner.Get() != g_device.Get()) {
    cache.clear();
    owner = g_device;
  }
  auto it = cache.find(used_mask);
  if (it != cache.end())
    return it->second;
  std::vector<wgpu::BindGroupLayoutEntry> entries;
  for (uint32_t t = 0; t < 8u; ++t) {
    if ((used_mask & (1u << t)) == 0u)
      continue;
    entries.push_back(wgpu::BindGroupLayoutEntry{
        .binding = 2u * t,
        .visibility = wgpu::ShaderStage::Fragment,
        .texture =
            wgpu::TextureBindingLayout{
                .sampleType = wgpu::TextureSampleType::Float,
                .viewDimension = wgpu::TextureViewDimension::e2D,
            },
    });
    entries.push_back(wgpu::BindGroupLayoutEntry{
        .binding = 2u * t + 1u,
        .visibility = wgpu::ShaderStage::Fragment,
        .sampler =
            wgpu::SamplerBindingLayout{
                .type = wgpu::SamplerBindingType::Filtering,
            },
    });
  }
  const wgpu::BindGroupLayoutDescriptor descriptor{
      .label = "GXCore Texture Bind Group Layout",
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  auto layout = g_device.CreateBindGroupLayout(&descriptor);
  cache.emplace(used_mask, layout);
  return layout;
}

// Guest-identity texture cache (S13 A3). Keyed by the texture's guest identity
// AND the TLUT identity it indexes: the same CI image bytes re-palettized to a
// different TLUT decode to different pixels, so the palette address/format/
// entries are part of the key. 63/Mfin adds `content_hash` (xxh3 of the actual
// source texels + TLUT): the live game reuses one guest texture-buffer address
// for different images across screens (stadium select overwrites the captain
// art) and streams movie frames through a single buffer, so identity alone would
// bind the stale decode. Hashing the content makes a content change under an
// unchanged identity a distinct entry (Dolphin/Aurora texel-hash model). Static
// replay fixtures never rewrite an address, so the no-reconvert property holds.
struct TextureKey {
  uint32_t address;
  uint32_t size;
  uint32_t format;
  uint32_t width;
  uint32_t height;
  uint32_t tlut_address;
  uint32_t tlut_format;
  uint32_t tlut_entries;
  uint64_t content_hash;
  bool operator==(const TextureKey&) const = default;
  template <typename H>
  friend H AbslHashValue(H h, const TextureKey& key) {
    return H::combine(std::move(h), key.address, key.size, key.format,
                      key.width, key.height, key.tlut_address, key.tlut_format,
                      key.tlut_entries, key.content_hash);
  }
};

absl::flat_hash_map<TextureKey, TextureHandle> g_textureCache;
// Original textures drawn while their HD replacement decodes on a background
// thread. They stay out of g_textureCache so the next draw asks again; the
// entry goes when the replacement arrives (or with the cache's clears).
absl::flat_hash_map<TextureKey, TextureHandle> g_textureAwaitingReplacement;
// address -> the key currently cached at that guest address. When new content
// arrives at an address (movie streaming, screen transitions), the prior entry
// for that address is evicted so the content-hashed cache stays bounded to one
// live entry per buffer instead of accumulating every historical frame.
struct TextureAddressState {
  TextureKey key;
  uint64_t texel_epoch = 0;
  uint64_t tlut_epoch = 0;
  bool texel_epoch_valid = false;
  bool tlut_epoch_valid = false;
  // Small textures only (see kAlwaysRehashBytes): hash of the texels and
  // palette as last decoded, and the frame in which they were last compared.
  uint64_t small_hash = 0;
  uint64_t verified_frame = ~0ull;
  // The texture cached under `key`, so an unchanged texture takes one lookup
  // instead of two (the second, by the whole key, was a tenth of the FIFO
  // worker's time on Outset); empty once that entry is evicted.
  TextureHandle handle;
};
absl::flat_hash_map<uint32_t, TextureAddressState> g_textureAddrKey;
// Counts presented frames; a small texture is re-hashed at most once in each.
std::atomic<uint64_t> g_textureVerifyFrame{0};
// A small texture is re-hashed even when its dirty epoch has not moved: the
// game rewrites some in place with plain CPU stores that no dirty mark sees
// (Wind Waker's A/B action labels, 80x24 IA4 at 0x01673420 and 0x01674C60,
// showed "Charts" for "Choose"/"Return" in the pause menu). Dolphin reads
// memory coherently, so the reference shows the new text. Once a frame per
// texture, up to 4 KB of texels plus the palette.
constexpr uint32_t kAlwaysRehashBytes = 4096u;
uint64_t small_texture_hash(const uint8_t* bytes, uint32_t size,
                            const void* tlut, uint32_t tlut_size) {
  uint64_t h = XXH3_64bits(bytes, size);
  if (tlut != nullptr && tlut_size != 0u)
    h ^= XXH3_64bits(tlut, tlut_size) * 0x9E3779B97F4A7C15ull;
  return h;
}
TextureCacheStats g_textureCacheStats;
TextureDirtyEpochObserver g_textureDirtyEpochObserver = nullptr;

// EFB copy allocations mirror GXState::copyTextureCache: one guest destination
// may be reused with different dimensions or formats, which require distinct GPU
// textures. g_efbCopyTextures mirrors GXState::copyTextures and selects the most
// recent allocation when that guest destination is sampled.
struct EfbCopyKey {
  uint32_t address;
  uint32_t width;
  uint32_t height;
  uint32_t format;
  bool opaque;  // the view reads alpha as one (EFB without alpha)
  bool operator==(const EfbCopyKey&) const = default;
  template <typename H>
  friend H AbslHashValue(H h, const EfbCopyKey& key) {
    return H::combine(std::move(h), key.address, key.width, key.height,
                      key.format, key.opaque);
  }
};
absl::flat_hash_map<EfbCopyKey, TextureHandle> g_efbCopyCache;
struct EfbCopyBinding {
  TextureHandle handle;
  uint32_t byte_size = 0;
  uint64_t memory_epoch = 0;
  bool memory_epoch_valid = false;
};
absl::flat_hash_map<uint32_t, EfbCopyBinding> g_efbCopyTextures;

} // namespace

bool needs_early_depth_emulation(const gxc::PipelineKey& key) {
  if (key.depth_test == 0u || key.depth_update == 0u ||
      key.early_depth_test == 0u || key.shader.tev_valid == 0u) {
    return false;
  }

  // Mirror AlphaTest::TestResult enough to omit the extra pass when the test is
  // statically guaranteed to pass. Fail and undetermined both need an early
  // depth write; the ordinary color shader will discard failures afterward.
  const bool c0_never = key.shader.alpha_comp0 == 0u;
  const bool c1_never = key.shader.alpha_comp1 == 0u;
  const bool c0_always = key.shader.alpha_comp0 == 7u;
  const bool c1_always = key.shader.alpha_comp1 == 7u;
  bool always_passes = false;
  switch (key.shader.alpha_logic) {
  case 0u: // And
    always_passes = c0_always && c1_always;
    break;
  case 1u: // Or
    always_passes = c0_always || c1_always;
    break;
  case 2u: // Xor
    always_passes = (c0_always && c1_never) ||
                    (c0_never && c1_always);
    break;
  case 3u: // Xnor
    always_passes = (c0_always && c1_always) ||
                    (c0_never && c1_never);
    break;
  }
  return !always_passes;
}

wgpu::RenderPipeline create_pipeline(const PipelineConfig& config) {
  const gxc::PipelineKey& key = config.key;
  const bool depthOnly = config.depthOnly != 0u;
  CHECK(key.shader.use_dst_alpha == 0 ||
            webgpu::g_dualSourceBlendingSupported,
        "GX destination alpha requires WebGPU dual-source blending");
  std::string wgsl = gxc::generate_wgsl(key.shader);
  if (depthOnly) {
    wgsl += "\n@fragment\nfn fs_depth_only() -> @location(0) vec4f {\n"
            "    return vec4f(0.0);\n}\n";
  }
  wgpu::ShaderSourceWGSL sourceDescriptor{};
  sourceDescriptor.code = wgsl.c_str();
  const wgpu::ShaderModuleDescriptor moduleDescriptor{
      .nextInChain = &sourceDescriptor,
      .label = "GXCore Shader Module",
  };
  const auto module = g_device.CreateShaderModule(&moduleDescriptor);

  // Group 0 keeps the pass preamble's static bind group compatible (the shader
  // never references it); 1 = shared dynamic VS uniform. On the TEV path (S14)
  // 2 = shared dynamic PS uniform and 3 = texture; else 2 = texture. Putting
  // the PS uniform before the texture keeps an untextured TEV draw gap-free.
  const bool tev = key.shader.tev_valid != 0;
  const bool textured = key.shader.textured != 0;
  // Multi-texmap: the texture group's layout matches the set of texmaps the WGSL
  // declares (derived identically from the shader key), so pipeline and bind
  // group agree. Untextured draws never use the texture group (layoutCount below).
  const uint32_t tex_mask = textured ? gxc::used_texmap_mask(key.shader) : 1u;
  std::array<wgpu::BindGroupLayout, 4> bindGroupLayouts{
      g_staticBindGroupLayout,
      g_uniformBindGroupLayout,
      tev ? g_uniformBindGroupLayout : texture_bind_group_layout(tex_mask),
      texture_bind_group_layout(tex_mask),
  };
  const size_t layoutCount =
      depthOnly ? 2 : tev ? (textured ? 4 : 3) : (textured ? 3 : 2);
  const wgpu::PipelineLayoutDescriptor layoutDescriptor{
      .label = "GXCore Pipeline Layout",
      .bindGroupLayoutCount = layoutCount,
      .bindGroupLayouts = bindGroupLayouts.data(),
  };
  const auto pipelineLayout = g_device.CreatePipelineLayout(&layoutDescriptor);

  // Fixed decoded-vertex layout (gxruntime/gxcore/shader.hpp). The normal
  // (location 8) is only declared by the lit shader, so add it to the pipeline
  // only when the key is lit — WGSL requires the vertex layout to satisfy every
  // shader input.
  std::vector<wgpu::VertexAttribute> attributes{
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Float32x3,
          .offset = gxc::kVertexPosOffset,
          .shaderLocation = 0,
      },
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Uint32,
          .offset = gxc::kVertexPosMtxOffset,
          .shaderLocation = 1,
      },
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Float32x4,
          .offset = gxc::kVertexColor0Offset,
          .shaderLocation = 2,
      },
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Float32x4,
          .offset = gxc::kVertexColor1Offset,
          .shaderLocation = 3,
      },
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Float32x2,
          .offset = gxc::kVertexUvOffset,
          .shaderLocation = 4,
      },
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Float32x2,
          .offset = gxc::kVertexUvOffset + 8,
          .shaderLocation = 5,
      },
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Float32x2,
          .offset = gxc::kVertexUvOffset + 16,
          .shaderLocation = 6,
      },
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Float32x2,
          .offset = gxc::kVertexUvOffset + 24,
          .shaderLocation = 7,
      },
  };
  // Item-5 texgen inputs must mirror the generator's VertexIn: emboss needs the
  // NBT normal/binormal/tangent + light dir; a Color1/emboss key that is unlit
  // still declares the normal (location 8). Detect emboss demand from the key.
  bool has_emboss = false;
  bool has_normal_source = false;
  for (std::uint32_t i = 0; i < key.shader.num_tex_gens; ++i) {
    const auto type =
        static_cast<gxc::TexGenType>(key.shader.tex_gens[i].texgentype);
    if (type == gxc::TexGenType::EmbossMap)
      has_emboss = true;
    if (type == gxc::TexGenType::Regular &&
        static_cast<gxc::TexSourceRow>(key.shader.tex_gens[i].sourcerow) ==
            gxc::TexSourceRow::Normal)
      has_normal_source = true;
  }
  // Mirror the generator exactly: locations 8/10/11 are declared only when the
  // vertex FORMAT carries that attribute. A lit/emboss draw whose format omits
  // it reads the cached fallback from the uniform instead (I_CACHED_NORMAL), so
  // the shader does not declare the input and the layout must not provide it.
  if ((key.shader.lit_valid != 0 || has_emboss || has_normal_source) &&
      key.shader.has_vertex_normal != 0) {
    attributes.push_back(wgpu::VertexAttribute{
        .format = wgpu::VertexFormat::Float32x3,
        .offset = gxc::kVertexNormalOffset,
        .shaderLocation = 8,
    });
  }
  if (key.shader.has_tex_mtx_idx != 0) {
    attributes.push_back(wgpu::VertexAttribute{
        .format = wgpu::VertexFormat::Uint32,
        .offset = gxc::kVertexTexMtxIdxOffset,
        .shaderLocation = 9,
    });
  }
  if (has_emboss && key.shader.has_vertex_binormal != 0) {
    attributes.push_back(wgpu::VertexAttribute{
        .format = wgpu::VertexFormat::Float32x3,
        .offset = gxc::kVertexBinormalOffset,
        .shaderLocation = 10,
    });
  }
  if (has_emboss && key.shader.has_vertex_tangent != 0) {
    attributes.push_back(wgpu::VertexAttribute{
        .format = wgpu::VertexFormat::Float32x3,
        .offset = gxc::kVertexTangentOffset,
        .shaderLocation = 11,
    });
  }
  if ((key.shader.uv_mask & (1u << 4u)) != 0u) {
    attributes.push_back(wgpu::VertexAttribute{
        .format = wgpu::VertexFormat::Float32x2,
        .offset = gxc::kVertexUvOffset + 32u,
        .shaderLocation = 12,
    });
  }
  if (key.shader.has_tex_mtx_idx != 0 &&
      (key.shader.tex_mtx_idx_mask & 0xF0u) != 0u) {
    attributes.push_back(wgpu::VertexAttribute{
        .format = wgpu::VertexFormat::Uint32,
        .offset = gxc::kVertexTexMtxIdxHiOffset,
        .shaderLocation = 13,
    });
  }
  const wgpu::VertexBufferLayout vertexLayout{
      .arrayStride = gxc::kVertexStrideBytes,
      .stepMode = wgpu::VertexStepMode::Vertex,
      .attributeCount = attributes.size(),
      .attributes = attributes.data(),
  };

  const bool emulateEarlyDepth = needs_early_depth_emulation(key);
  const bool depthCompare = key.depth_test != 0;
  const wgpu::DepthStencilState depthStencil{
      .format = g_graphicsConfig.depthFormat,
      .depthWriteEnabled =
          depthOnly ? depthCompare && key.depth_update != 0
                    : depthCompare && key.depth_update != 0 &&
                          !emulateEarlyDepth,
      .depthCompare =
          emulateEarlyDepth && !depthOnly
              ? wgpu::CompareFunction::Equal
              : depthCompare ? to_compare(static_cast<gxc::CompareMode>(
                                   key.depth_func))
                             : wgpu::CompareFunction::Always,
  };

  // GC subtract mode forces ONE/ONE with dst - src (Dolphin RenderState).
  const bool dual_source = key.shader.use_dst_alpha != 0;
  wgpu::BlendState blendState{};
  if (key.blend_subtract != 0) {
    blendState.color = {
        .operation = wgpu::BlendOperation::ReverseSubtract,
        .srcFactor = wgpu::BlendFactor::One,
        .dstFactor = wgpu::BlendFactor::One,
    };
    blendState.alpha = dual_source
                           ? wgpu::BlendComponent{
                                 .operation = wgpu::BlendOperation::Add,
                                 .srcFactor = wgpu::BlendFactor::One,
                                 .dstFactor = wgpu::BlendFactor::Zero,
                             }
                           : blendState.color;
  } else {
    blendState.color = {
        .operation = wgpu::BlendOperation::Add,
        .srcFactor = to_blend_factor_src(
            static_cast<gxc::SrcBlendFactor>(key.src_factor), dual_source),
        .dstFactor = to_blend_factor_dst(
            static_cast<gxc::DstBlendFactor>(key.dst_factor), dual_source),
    };
    blendState.alpha = {
        .operation = wgpu::BlendOperation::Add,
        .srcFactor = to_blend_factor_src(
            static_cast<gxc::SrcBlendFactor>(key.src_factor_alpha),
            dual_source),
        .dstFactor = to_blend_factor_dst(
            static_cast<gxc::DstBlendFactor>(key.dst_factor_alpha),
            dual_source),
    };
  }
  auto writeMask = wgpu::ColorWriteMask::None;
  if (key.color_update != 0) {
    writeMask |= wgpu::ColorWriteMask::Red | wgpu::ColorWriteMask::Green |
                 wgpu::ColorWriteMask::Blue;
  }
  if (key.alpha_update != 0)
    writeMask |= wgpu::ColorWriteMask::Alpha;
  const bool blending =
      key.blend_enable != 0 || key.blend_subtract != 0;
  const wgpu::ColorTargetState colorTarget{
      .format = g_graphicsConfig.surfaceConfiguration.format,
      .blend = blending ? &blendState : nullptr,
      .writeMask = writeMask,
  };
  const wgpu::ColorTargetState depthOnlyTarget{
      .format = g_graphicsConfig.surfaceConfiguration.format,
      .blend = nullptr,
      .writeMask = wgpu::ColorWriteMask::None,
  };
  const wgpu::FragmentState fragmentState{
      .module = module,
      .entryPoint = depthOnly ? "fs_depth_only" : "fs_main",
      .targetCount = 1,
      .targets = depthOnly ? &depthOnlyTarget : &colorTarget,
  };

  auto cullMode = wgpu::CullMode::None;
  switch (static_cast<gxc::CullMode>(key.cull_mode)) {
  case gxc::CullMode::Back:
    cullMode = wgpu::CullMode::Back;
    break;
  case gxc::CullMode::Front:
    cullMode = wgpu::CullMode::Front;
    break;
  default:
    break; // None here; All was skipped at plan time
  }
  auto topology = wgpu::PrimitiveTopology::TriangleList;
  if (key.primitive_topology == 1u) {
    topology = wgpu::PrimitiveTopology::LineList;
    cullMode = wgpu::CullMode::None;
  } else if (key.primitive_topology == 2u) {
    topology = wgpu::PrimitiveTopology::PointList;
    cullMode = wgpu::CullMode::None;
  }

  const wgpu::RenderPipelineDescriptor descriptor{
      .label = "GXCore Pipeline",
      .layout = pipelineLayout,
      .vertex =
          wgpu::VertexState{
              .module = module,
              .entryPoint = "vs_main",
              .bufferCount = 1,
              .buffers = &vertexLayout,
          },
      .primitive =
          wgpu::PrimitiveState{
              .topology = topology,
              // Substrate winding convention (gx/gx.cpp to_primitive_state).
              .frontFace = wgpu::FrontFace::CW,
              .cullMode = cullMode,
          },
      .depthStencil = &depthStencil,
      .multisample =
          wgpu::MultisampleState{
              .count = config.msaaSamples,
          },
      .fragment = &fragmentState,
  };
  return g_device.CreateRenderPipeline(&descriptor);
}

// What render() last bound in the pass being encoded, so a draw sets only
// what differs from the draw before it. Every draw set its vertex and index
// ranges and its bind groups again, six or seven commands a draw, and the
// in-between frames encode every draw once more each (three times at 120 Hz):
// the render worker's encoding was most of a game frame in a busy scene. The
// vertex and index buffers are bound whole once (a draw's range is its base
// vertex and first index), and a bind group is set again only when it or its
// offset changed or the pipeline did (a tev pipeline's group 2 is another
// layout than an untextured one's).
namespace {
struct PassState {
  WGPUBuffer vertexBuffer = nullptr; // bound whole, or a particle's range
  uint64_t vertexOffset = 0;
  bool indexBound = false;
  PipelineRef pipeline = 0;
  WGPUBindGroup group1 = nullptr;
  uint32_t offset1 = UINT32_MAX;
  WGPUBindGroup group2 = nullptr;
  uint32_t offset2 = UINT32_MAX;
  WGPUBindGroup group3 = nullptr;
};
PassState g_pass;
} // namespace

void reset_pass_state() { g_pass = PassState{}; }

// What a draw, or one of a batch's (submit_draw_plan) draws, is encoded with:
// its constants and a particle's in-between vertices (empty: the frame's
// own), its indices, and its first vertex from the batch's.
namespace {
struct DrawPart {
  Range uniform;
  Range verts;
  uint32_t firstIndex;
  uint32_t indexCount;
  uint32_t firstVertex;
};
} // namespace

void render(const DrawData& data, const wgpu::RenderPassEncoder& pass) {
  const auto bind = [&](PipelineRef ref) {
    if (!bind_pipeline(ref, pass))
      return false;
    if (ref != g_pass.pipeline) {
      g_pass.pipeline = ref;
      g_pass.group1 = g_pass.group2 = g_pass.group3 = nullptr;
    }
    return true;
  };
  // Ensure both asynchronous pipelines are ready before the prepass writes any
  // depth. Binding the color pipeline first is harmless; it is rebound below.
  if (!bind(data.pipeline)) {
    return;
  }
  if (data.depthPipeline != 0 && !bind(data.depthPipeline))
    return;
  // The in-between frame being encoded: its step's block and vertices (a
  // traced frame's own, matched while recording, are one step's).
  const bool interpolated = frame_interp::encoding_interpolated();
  const int step = interpolated ? interp_replay_step() : 0;
  const auto job_ranges = [&](uint32_t job, Range& uniform, Range& verts) {
    uniform = verts = {};
    if (!interpolated)
      return;
    if (data.interpJob != UINT32_MAX) {
      const InterpRanges& ranges = interp_job_ranges(job);
      uniform = ranges.uniform[step];
      verts = ranges.verts[step];
    } else if (step == 0) {
      uniform = data.interpUniformRange;
      verts = data.interpVertRange;
    }
  };
  const uint32_t firstIndex = data.idxRange.offset / sizeof(uint16_t);
  // One draw, or a batch as one; a batch whose draws' in-between blocks (or a
  // particle's vertices, laid out apart from the batch's) differ, one by one.
  static std::vector<DrawPart> parts;
  parts.clear();
  DrawPart whole{.firstIndex = firstIndex, .indexCount = data.indexCount, .firstVertex = 0};
  job_ranges(data.interpJob, whole.uniform, whole.verts);
  bool asOne = true;
  if (interpolated && data.batchSize > 1 && data.interpJob != UINT32_MAX) {
    for (uint32_t i = 1; i < data.batchSize && asOne; ++i) {
      Range uniform, verts;
      job_ranges(data.interpJob + i, uniform, verts);
      const uint32_t first = batch_draw(data.batch + i).firstVertex;
      asOne = uniform.offset == whole.uniform.offset && uniform.size == whole.uniform.size &&
              (whole.verts.size == 0 ? verts.size == 0
                                     : verts.size != 0 && verts.offset == whole.verts.offset +
                                                                              first * gxc::kVertexStrideBytes);
    }
    if (!asOne) {
      uint32_t index = firstIndex;
      for (uint32_t i = 0; i < data.batchSize; ++i) {
        const BatchDraw& part = batch_draw(data.batch + i);
        DrawPart& draw = parts.emplace_back(
            DrawPart{.firstIndex = index, .indexCount = part.indexCount, .firstVertex = part.firstVertex});
        job_ranges(data.interpJob + i, draw.uniform, draw.verts);
        index += part.indexCount;
      }
    }
  }
  if (asOne)
    parts.push_back(whole);

  const auto setGroup1 = [&](const DrawPart& part) {
    const bool blended = part.uniform.size != 0;
    const auto& vsGroup = blended ? g_interpUniformBindGroup : g_uniformBindGroup;
    const uint32_t vsOffset = blended ? part.uniform.offset : data.uniformRange.offset;
    if (vsGroup.Get() != g_pass.group1 || vsOffset != g_pass.offset1) {
      pass.SetBindGroup(1, vsGroup, 1, &vsOffset);
      g_pass.group1 = vsGroup.Get();
      g_pass.offset1 = vsOffset;
    }
  };
  // The vertices: from the whole buffer at the draw's (or batch's) base
  // vertex, or a particle's blended ones from the in-between vertex area,
  // bound where the part's first index (counted from the batch's first
  // vertex) finds them.
  const auto setVertices = [&](const DrawPart& part) -> int32_t {
    const uint64_t back = uint64_t{part.firstVertex} * gxc::kVertexStrideBytes;
    if (part.verts.size != 0 && part.verts.offset >= back) {
      const uint64_t offset = part.verts.offset - back;
      if (g_pass.vertexBuffer != g_interpVertexBuffer.Get() || g_pass.vertexOffset != offset) {
        pass.SetVertexBuffer(0, g_interpVertexBuffer, offset);
        g_pass.vertexBuffer = g_interpVertexBuffer.Get();
        g_pass.vertexOffset = offset;
      }
      return 0;
    }
    if (data.vertRange.offset % gxc::kVertexStrideBytes == 0) {
      if (g_pass.vertexBuffer != g_vertexBuffer.Get() || g_pass.vertexOffset != 0) {
        pass.SetVertexBuffer(0, g_vertexBuffer);
        g_pass.vertexBuffer = g_vertexBuffer.Get();
        g_pass.vertexOffset = 0;
      }
      return static_cast<int32_t>(data.vertRange.offset / gxc::kVertexStrideBytes);
    }
    pass.SetVertexBuffer(0, g_vertexBuffer, data.vertRange.offset, data.vertRange.size);
    g_pass.vertexBuffer = g_vertexBuffer.Get();
    g_pass.vertexOffset = data.vertRange.offset;
    return 0;
  };
  if (!g_pass.indexBound) {
    pass.SetIndexBuffer(g_indexBuffer, wgpu::IndexFormat::Uint16);
    g_pass.indexBound = true;
  }
  const auto draw = [&] {
    for (const DrawPart& part : parts) {
      setGroup1(part);
      const int32_t baseVertex = setVertices(part);
      pass.DrawIndexed(part.indexCount, 1, part.firstIndex, baseVertex);
    }
  };
  if (data.depthPipeline != 0) {
    draw();
    if (!bind(data.pipeline))
      return;
  }
  if (data.tev) {
    // group 2 = PS uniform (same dynamic-uniform bind group, its own offset);
    // group 3 = texture.
    const uint32_t psOffset = data.pixelUniformRange.offset;
    if (g_uniformBindGroup.Get() != g_pass.group2 || psOffset != g_pass.offset2) {
      pass.SetBindGroup(2, g_uniformBindGroup, 1, &psOffset);
      g_pass.group2 = g_uniformBindGroup.Get();
      g_pass.offset2 = psOffset;
    }
    if (data.textureBindGroup != 0) {
      const auto& group = find_bind_group(data.textureBindGroup);
      if (group.Get() != g_pass.group3) {
        pass.SetBindGroup(3, group);
        g_pass.group3 = group.Get();
      }
    }
  } else if (data.textureBindGroup != 0) {
    const auto& group = find_bind_group(data.textureBindGroup);
    if (group.Get() != g_pass.group2 || g_pass.offset2 != UINT32_MAX) {
      pass.SetBindGroup(2, group);
      g_pass.group2 = group.Get();
      g_pass.offset2 = UINT32_MAX;
    }
  }
  draw();
}

void note_frame_presented() {
  g_textureVerifyFrame.fetch_add(1u, std::memory_order_relaxed);
  frame_interp::end_game_frame();
}

void reset_texture_cache() {
  g_textureCache.clear();
  g_textureAwaitingReplacement.clear();
  g_textureAddrKey.clear();
  g_efbCopyCache.clear();
  g_efbCopyTextures.clear();
  g_textureCacheStats = {};
}

void set_texture_dirty_epoch_observer(TextureDirtyEpochObserver observer) {
  g_textureDirtyEpochObserver = observer;
}

const TextureCacheStats& texture_cache_stats() { return g_textureCacheStats; }
unsigned long long texture_upload_count() { return g_textureCacheStats.uploads; }

// Resolve the current EFB region into a texture the guest-identity path can bind
// later, keyed by the copy's destination address (63/S16). Mirrors aurora
// lib/dolphin/gx/GXFrameBuffer.cpp copy_tex on the same substrate: map the EFB
// source rect through the logical->render scaling, allocate (or reuse) a resolve
// target, then gfx::resolve_pass — which converts per format (tex_copy_conv,
// incl. depth targets) and optionally clears the EFB to the game's copy-clear
// color/Z afterwards. The bound draw's geometry is already in the pass (the
// sink flushed the pending draw before firing this).
void copy_efb_to_texture(const gxc::EfbCopyCommand& cmd) {
  // Mirror the copy-clear color/Z (BP 0x4F-0x51 at this copy) into the gx
  // state GXSetCopyClear would have written: begin_frame's pass-0 EFB clear
  // reads g_gxState.clearColor/clearDepth, and in gxcore mode the live gx
  // layer that normally maintains them is bypassed. Without this every frame
  // cleared to the default alpha=1 and e.g. the Strikers shadow-grab alpha
  // background inverted (glxSwap display-copy clears to {0,0,0,0}).
  gx::g_gxState.clearColor = {
      static_cast<float>(cmd.clear_r) / 255.f,
      static_cast<float>(cmd.clear_g) / 255.f,
      static_cast<float>(cmd.clear_b) / 255.f,
      static_cast<float>(cmd.clear_a) / 255.f,
  };
  gx::g_gxState.clearDepth = cmd.clear_z;
  if (cmd.format == 0xFu) {
    // Display copy (GXCopyDisp): no texture destination; its requested clear
    // becomes the next frame's EFB clear via the state mirrored above.
    return;
  }
  const auto fmt = static_cast<GXTexFmt>(cmd.format);
  const gfx::ClipRect srcRect = gx::map_logical_scissor(gfx::ClipRect{
      .x = static_cast<int32_t>(cmd.src_x),
      .y = static_cast<int32_t>(cmd.src_y),
      .width = static_cast<int32_t>(cmd.width),
      .height = static_cast<int32_t>(cmd.height),
  });
  // The copy is made at the render target's scale, as Aurora's own
  // GXCopyTex does (dolphin/gx/GXFrameBuffer.cpp scale_copy_dst) and as
  // Dolphin's scaled EFB copies do. Allocated at the guest's size, a game
  // that copies the whole EFB and draws it back (Wind Waker's depth of field
  // and blur passes, every frame) replaced its high-resolution scene with a
  // 640x480 image stretched over it. Sampling uses normalized coordinates,
  // so a larger texture needs nothing else.
  uint32_t dstWidth = std::max(cmd.destination_width, static_cast<uint32_t>(1));
  uint32_t dstHeight = std::max(cmd.destination_height, static_cast<uint32_t>(1));
  static const bool scale_copies = [] {
    const char* env = std::getenv("DOL_GXCORE_COPY_SCALE");
    return env == nullptr || env[0] != '0';
  }();
  if (scale_copies && gx::g_gxState.viewportPolicy != AURORA_VIEWPORT_NATIVE) {
    const auto [logicalW, logicalH] = gx::logical_fb_size();
    const auto [targetW, targetH] = gfx::get_render_target_size();
    if (logicalW != 0 && logicalH != 0 && targetW != 0 && targetH != 0) {
      const float sx = static_cast<float>(targetW) / static_cast<float>(logicalW);
      const float sy = static_cast<float>(targetH) / static_cast<float>(logicalH);
      dstWidth = std::max<uint32_t>(static_cast<uint32_t>(std::lround(dstWidth * sx)), 1u);
      dstHeight = std::max<uint32_t>(static_cast<uint32_t>(std::lround(dstHeight * sy)), 1u);
    }
  }

  const EfbCopyKey key{cmd.dest_address, dstWidth, dstHeight, cmd.format, !cmd.efb_has_alpha};
  auto it = g_efbCopyCache.find(key);
  if (it == g_efbCopyCache.end() || !it->second) {
    TextureHandle handle;
    if (gfx::tex_copy_conv::needs_conversion(fmt)) {
      handle = gfx::new_conv_texture(dstWidth, dstHeight, fmt, "GXCore Copy Conv");
    } else {
      // As Aurora's GXCopyTex: an RGB565 target, or an EFB without alpha,
      // gets a view whose alpha reads one.
      const auto viewFmt = fmt == GX_TF_RGB565 || !cmd.efb_has_alpha ? GX_TF_RGB565 : GX_TF_RGBA8;
      handle = gfx::new_render_texture(dstWidth, dstHeight, viewFmt, "GXCore Copy");
    }
    it = g_efbCopyCache.insert_or_assign(key, handle).first;
  }
  if (!it->second) {
    return;
  }
  uint64_t memoryEpoch = 0;
  const bool memoryEpochValid =
      g_textureDirtyEpochObserver != nullptr && cmd.byte_size != 0u &&
      g_textureDirtyEpochObserver(cmd.dest_address, cmd.byte_size, &memoryEpoch);
  g_efbCopyTextures.insert_or_assign(
      cmd.dest_address,
      EfbCopyBinding{it->second, cmd.byte_size, memoryEpoch, memoryEpochValid});
  float clearDepthValue = static_cast<float>(cmd.clear_z) / 16777215.f;
  if (gx::UseReversedZ) {
    clearDepthValue = 1.f - clearDepthValue;
  }
  gfx::resolve_pass(it->second, srcRect, cmd.clear && cmd.color_update,
                    cmd.clear && cmd.alpha_update,
                    cmd.clear && cmd.depth_update,
                    Vec4<float>{static_cast<float>(cmd.clear_r) / 255.f,
                                static_cast<float>(cmd.clear_g) / 255.f,
                                static_cast<float>(cmd.clear_b) / 255.f,
                                static_cast<float>(cmd.clear_a) / 255.f},
                    clearDepthValue, fmt);
}

// Uniform de-duplication across consecutive draws.
//
// Each draw carries the full vertex-constant block (every position, texture
// and normal matrix, about 2 KB) and a pixel-constant block, and consecutive
// draws of one model often carry byte-identical blocks. Pushing them anyway
// filled Aurora's 24 MB uniform staging area in the middle of Wind Waker's
// Outset frames, and each split waited on the GPU. A draw whose block matches
// the previous draw's, within the same frame packet, reuses that range.
// On Apple GPUs the block a cache last took is compared where it was staged
// (uniform_bytes, or interp_uniform_bytes for the in-between frames'): the
// staging area is ordinary memory there, and a copy of each block kept aside
// for the comparison was a tenth of the FIFO worker's copying. Elsewhere
// (Direct3D 12, Vulkan) the staging area is the GPU's upload memory, which the
// CPU reads uncached: comparing there took Wind Waker's Outset frames (14,000
// draws) 400 ms on the FIFO worker, so the block is kept aside.
#if defined(__APPLE__)
#define AURORA_UNIFORM_COMPARE_STAGED 1
#else
#define AURORA_UNIFORM_COMPARE_STAGED 0
#endif
struct UniformCache {
  Range range{};
  uint64_t frameId = 0;
  uint64_t hits = 0;
  uint64_t pushes = 0;
#if !AURORA_UNIFORM_COMPARE_STAGED
  std::vector<uint8_t> bytes; // the block range holds
#endif
};
static UniformCache g_vertexUniformCache;
static UniformCache g_interpUniformCache;
static UniformCache g_pixelUniformCache;

// The block the cache last took: `staged` where it was staged, or the copy.
static const uint8_t* cached_bytes(const UniformCache& cache, const uint8_t* staged) {
#if AURORA_UNIFORM_COMPARE_STAGED
  return staged;
#else
  (void)staged;
  return cache.bytes.data();
#endif
}

// The cache took `data` (staged at cache.range).
static void keep_cached(UniformCache& cache, const uint8_t* data, size_t length) {
#if AURORA_UNIFORM_COMPARE_STAGED
  (void)cache;
  (void)data;
  (void)length;
#else
  cache.bytes.assign(data, data + length);
#endif
}

// Whether `data` is the block the cache last took, in this frame packet.
static bool repeats_cached(const UniformCache& cache, const uint8_t* data, size_t length) {
  const uint64_t frameId = current_frame_id();
  return frameId != 0 && cache.frameId == frameId && cache.range.size == length &&
         std::memcmp(cached_bytes(cache, AURORA_UNIFORM_COMPARE_STAGED ? uniform_bytes(cache.range) : nullptr), data,
                     length) == 0;
}

// The in-between frame's blocks, de-duplicated the same way. repeated: the
// in-between frame made this block the way
// it made the last one, from the same constants (frame_interp::last_blend_repeated),
// so its bytes are the same.
static Range push_interp_uniform_dedup(UniformCache& cache, uint64_t frameId, size_t slot, const uint8_t* data,
                                       size_t length, bool repeated) {
  if (frameId != 0 && cache.frameId == frameId && cache.range.size == length &&
      (repeated ||
       std::memcmp(cached_bytes(cache, AURORA_UNIFORM_COMPARE_STAGED ? interp_uniform_bytes(slot, cache.range)
                                                                     : nullptr),
                   data, length) == 0)) {
    ++cache.hits;
    return cache.range;
  }
  ++cache.pushes;
  cache.range = push_interp_uniform(slot, data, length);
  cache.frameId = frameId;
  keep_cached(cache, data, length);
  return cache.range;
}

// --- In-between frames on a helper thread -----------------------------------
//
// Matching each draw to its counterpart in the frame before and blending its
// constants (frame_interp::blend_draw), then staging the blended block, was a
// fifth of the FIFO worker's time. That worker is what holds the game below
// 30 FPS in the busiest scenes (Outset's village, 10,000 draws a frame, on a
// Windows PC: 2026-09-29), so a helper thread does it, in draw order, while
// the worker goes on. Each draw records its job; end_frame waits for the
// helper before a frame packet is handed on, and the render worker reads the
// finished ranges (interp_job_ranges). One recording thread at a time queues
// jobs (the recording lock serialises them), so the queue is single-producer.
namespace {
struct InterpJob {
  frame_interp::DrawInput input; // its key, samples and a particle's positions
  std::vector<float> vertices;   // a particle's decoded vertices, for its blended copy
  uint64_t frameId;
  size_t slot;
  bool repeatsLastDraw; // the constants are the draw before's (repeats_cached)
  // The constants, copied only where they differ from the job before's
  // (constantsCopied); a job that repeats them leaves the 2.8 KB copy to the
  // helper, which keeps the last it was given (InterpHelper::constants).
  bool constantsCopied;
  gxc::VertexShaderConstants constants;
};

// A particle's in-between vertices: its own, with the blended positions, in
// the in-between vertex area of `slot`.
Range push_blended_vertices(size_t slot, const std::vector<float>& vertices, const float* positions,
                            std::vector<float>& scratch) {
  scratch.assign(vertices.begin(), vertices.end());
  const size_t count = scratch.size() / gxc::kVertexFloats;
  for (size_t i = 0; i < count; ++i)
    std::memcpy(scratch.data() + i * gxc::kVertexFloats + gxc::kVertexPosOffset / sizeof(float), positions + i * 3u,
                sizeof(float) * 3u);
  return push_interp_vertices(slot, reinterpret_cast<const uint8_t*>(scratch.data()), scratch.size() * sizeof(float));
}

struct InterpHelper {
  static constexpr uint64_t Capacity = 1024;
  // A sleeping helper is woken for this many jobs at once (or by
  // wait_interp_jobs at the frame's end), not for every draw.
  static constexpr uint64_t WakeBatch = 32;
  std::unique_ptr<InterpJob[]> ring{new InterpJob[Capacity]};
  std::atomic<uint64_t> produced{0};
  std::atomic<uint64_t> consumed{0};
  std::atomic<bool> sleeping{false};
  std::mutex mutex;
  std::condition_variable work;
  std::condition_variable idle;
  // Recording thread: the frame packet being queued and its job count.
  uint64_t packet = 0;
  uint32_t jobs = 0;
  // Helper thread: the last block it staged per step, and a particle's vertices.
  UniformCache cache[frame_interp::kMaxSteps];
  std::vector<float> vertices;
  // Helper thread: the constants of the last job that carried them, which the
  // jobs after it that repeat them use.
  gxc::VertexShaderConstants constants;
};

// Made with the thread and never destroyed: the detached thread may still be
// waiting on it while statics are torn down at exit.
std::atomic<InterpHelper*> g_interpHelper{nullptr};

void interp_helper_main(InterpHelper* h) {
  thread_cpu::register_current(thread_cpu::Role::InterpHelper);
  for (;;) {
    const uint64_t tail = h->consumed.load(std::memory_order_relaxed);
    if (tail == h->produced.load(std::memory_order_seq_cst)) {
      // Draws come microseconds apart while a frame records: wait a little
      // before sleeping. Woken for every draw it was 300,000 wake-ups a second
      // (more CPU than the work); spinning 4,096 times before each sleep
      // instead was an eighth of its time. It spins briefly, and a sleeping
      // helper is woken for a batch of jobs (WakeBatch).
      bool arrived = false;
      for (int spin = 0; spin < 256 && !arrived; ++spin) {
#if defined(__x86_64__) || defined(_M_X64)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        __asm__ __volatile__("yield");
#endif
        arrived = h->produced.load(std::memory_order_acquire) != tail;
      }
      if (arrived)
        continue;
      std::unique_lock lock{h->mutex};
      h->sleeping.store(true, std::memory_order_seq_cst);
      h->idle.notify_all();
      h->work.wait(lock, [h] {
        return h->produced.load(std::memory_order_seq_cst) != h->consumed.load(std::memory_order_relaxed);
      });
      h->sleeping.store(false, std::memory_order_relaxed);
      continue;
    }
    const InterpJob& job = h->ring[tail % InterpHelper::Capacity];
    InterpRanges ranges{};
    const int steps = frame_interp::frame_steps();
    if (job.constantsCopied)
      std::memcpy(&h->constants, &job.constants, sizeof(h->constants));
    if (frame_interp::blend_draw(job.input, h->constants, job.repeatsLastDraw) != nullptr) {
      const bool repeated = frame_interp::last_blend_repeated();
      for (int step = 0; step < steps; ++step)
        ranges.uniform[step] =
            push_interp_uniform_dedup(h->cache[step], job.frameId, job.slot,
                                      reinterpret_cast<const uint8_t*>(frame_interp::blended_step(step)),
                                      sizeof(gxc::VertexShaderConstants), repeated);
    }
    for (int step = 0; step < steps; ++step) {
      if (const float* positions = frame_interp::blended_positions(step))
        ranges.verts[step] = push_blended_vertices(job.slot, job.vertices, positions, h->vertices);
    }
    resolve_interp_job(job.slot, ranges);
    h->consumed.store(tail + 1, std::memory_order_release);
  }
}

uint32_t queue_interp_job(const gxc::DrawPlan& plan, bool repeatsLastDraw) {
  InterpHelper* h = g_interpHelper.load(std::memory_order_acquire);
  if (h == nullptr) {
    h = new InterpHelper;
    g_interpHelper.store(h, std::memory_order_release);
    std::thread(interp_helper_main, h).detach();
  }
  const uint64_t frameId = current_frame_id();
  if (h->packet != frameId) {
    h->packet = frameId;
    h->jobs = 0;
  }
  const uint64_t head = h->produced.load(std::memory_order_relaxed);
  while (head - h->consumed.load(std::memory_order_acquire) >= InterpHelper::Capacity)
    std::this_thread::yield();
  InterpJob& job = h->ring[head % InterpHelper::Capacity];
  frame_interp::capture_draw(plan, job.input);
  if (job.input.positions.empty())
    job.vertices.clear();
  else
    job.vertices.assign(plan.vertices.begin(), plan.vertices.end());
  job.frameId = frameId;
  job.slot = recording_frame_slot();
  job.repeatsLastDraw = repeatsLastDraw;
  // repeatsLastDraw: the constants are those last pushed in this frame packet,
  // and every push while frames are interpolated here queues a job, so they
  // are those of this packet's job before (the helper's copy). The first job
  // of a packet carries its own.
  job.constantsCopied = !repeatsLastDraw || h->jobs == 0u;
  if (job.constantsCopied)
    std::memcpy(&job.constants, &plan.constants, sizeof(plan.constants));
  h->produced.store(head + 1, std::memory_order_release);
  // Asleep, it consumes nothing, so a sleeping helper has WakeBatch or more
  // waiting before it is woken. Only then is `sleeping` read, after a full
  // fence: the helper sets it before its last look at `produced`, so one of
  // the two sees the other. Every job paid for a sequentially consistent store
  // and load here, which waited for the job's copies to leave the store buffer
  // (4 percent of the translation worker).
  if (head + 1 - h->consumed.load(std::memory_order_acquire) >= InterpHelper::WakeBatch) {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (h->sleeping.load(std::memory_order_relaxed)) {
      std::lock_guard lock{h->mutex};
      h->work.notify_one();
    }
  }
  return h->jobs++;
}

// The job queue_interp_job gives the recording frame's next draw.
uint32_t next_interp_job() {
  const InterpHelper* h = g_interpHelper.load(std::memory_order_acquire);
  return h == nullptr || h->packet != current_frame_id() ? 0u : h->jobs;
}
} // namespace

void wait_interp_jobs() {
  InterpHelper* h = g_interpHelper.load(std::memory_order_acquire);
  if (h == nullptr ||
      h->consumed.load(std::memory_order_acquire) == h->produced.load(std::memory_order_acquire))
    return;
  std::unique_lock lock{h->mutex};
  // The last jobs, fewer than a batch, are waiting on this.
  h->work.notify_one();
  h->idle.wait(lock, [h] {
    return h->consumed.load(std::memory_order_acquire) == h->produced.load(std::memory_order_acquire);
  });
}

// known: 1 or 0 when the caller has already compared `data` with the cache's
// block (repeats_cached), -1 to compare here.
static Range push_uniform_dedup(UniformCache& cache, const uint8_t* data,
                                size_t length, int known = -1) {
  static const bool enabled = [] {
    const char* env = std::getenv("DOL_AURORA_UNIFORM_DEDUP");
    return env == nullptr || env[0] != '0';
  }();
  const uint64_t frameId = current_frame_id();
  if (enabled && frameId != 0 && cache.frameId == frameId && cache.range.size == length &&
      (known >= 0 ? known == 1
                  : std::memcmp(cached_bytes(cache, AURORA_UNIFORM_COMPARE_STAGED ? uniform_bytes(cache.range) : nullptr),
                                data, length) == 0)) {
    ++cache.hits;
    return cache.range;
  }
  ++cache.pushes;
  if (((cache.hits + cache.pushes) & 0x3FFFFu) == 0)
    Log.info("GXCore uniform reuse: {} of {} blocks reused",
             cache.hits, cache.hits + cache.pushes);
  cache.range = push_uniform(data, length);
  cache.frameId = frameId;
  keep_cached(cache, data, length);
  return cache.range;
}

// DOL_GXCORE_DRAW_DUMP=<game frame>: every draw of that frame, its TEV and
// indirect stages, texgens, textures and blending, on stderr (debug).
static void dump_draw(const gxc::DrawPlan& plan) {
  static const uint64_t frame = [] {
    const char* env = std::getenv("DOL_GXCORE_DRAW_DUMP");
    return env != nullptr ? std::strtoull(env, nullptr, 10) : 0ull;
  }();
  if (frame == 0 || frame_interp::game_frame_number() != frame)
    return;
  static uint32_t index = 0;
  const auto& key = plan.pipeline;
  const auto& sh = key.shader;
  const auto& m = plan.constants.posnormalmatrix;
  std::fprintf(stderr,
               "[draw-dump] #%u verts=%u prim=0x%02X t=(%.0f,%.0f,%.0f) blend=%u src=%u dst=%u z=%u/%u/%u tev=%u "
               "stages=%u ind=%u texgens=%u chans=%u lit=%u alpha=%u/%u/%u fog=%u tag=%06X\n",
               index++, plan.vertex_count, plan.match_primitive, m[0][3], m[1][3], m[2][3], key.blend_enable,
               key.src_factor, key.dst_factor, key.depth_test, key.depth_func, key.depth_update, sh.tev_valid,
               sh.num_tev_stages, sh.num_ind_stages, sh.num_tex_gens, sh.num_color_chans, sh.lit_valid,
               sh.alpha_comp0, sh.alpha_comp1, sh.alpha_logic, sh.fog_fsel, plan.draw_tag);
  for (uint32_t i = 0; i < sh.num_tev_stages && i < gxc::kMaxTevStages; ++i) {
    const auto& t = sh.tev_stages[i];
    std::fprintf(stderr,
                 "[draw-dump]   s%u tc%u map%u en%u ras%u  C=%u,%u,%u,%u op%u b%u s%u cl%u ->%u  "
                 "A=%u,%u,%u,%u op%u b%u s%u cl%u ->%u  k=%u/%u  ind: st%u fmt%u bias%u mtx%u id%u wrap%u,%u add%u\n",
                 i, t.tevorders_texcoord, t.tevorders_texmap, t.tevorders_enable, t.tevorders_colorchan, t.cc_a,
                 t.cc_b, t.cc_c, t.cc_d, t.cc_op, t.cc_bias, t.cc_scale, t.cc_clamp, t.cc_dest, t.ac_a, t.ac_b, t.ac_c,
                 t.ac_d, t.ac_op, t.ac_bias, t.ac_scale, t.ac_clamp, t.ac_dest, t.ksel_kc, t.ksel_ka, t.ind_stage,
                 t.ind_format, t.ind_bias, t.ind_matrix_index, t.ind_matrix_id, t.ind_wrap_s, t.ind_wrap_t,
                 t.ind_add_prev);
  }
  for (uint32_t i = 0; i < sh.num_ind_stages && i < 4u; ++i)
    std::fprintf(stderr, "[draw-dump]   ind%u map%u tc%u scale=%u,%u\n", i, sh.ind_stages[i].texmap,
                 sh.ind_stages[i].texcoord, sh.ind_stages[i].scale_s, sh.ind_stages[i].scale_t);
  for (uint32_t i = 0; i < sh.num_tex_gens && i < gxc::kMaxTexGens; ++i) {
    std::fprintf(stderr, "[draw-dump]   tg%u type%u src%u form%u proj%u row%u (MatrixIndexA %08X)\n", i,
                 sh.tex_gens[i].texgentype, sh.tex_gens[i].sourcerow, sh.tex_gens[i].inputform,
                 sh.tex_gens[i].projection, plan.texgen_row[i], plan.matrix_index_a);
    for (uint32_t r = 0; r < 3u; ++r) {
      const float* row = plan.constants.texmatrices[i * 3u + r];
      std::fprintf(stderr, "[draw-dump]     tm%u.%u %.4f %.4f %.4f %.4f\n", i, r, row[0], row[1], row[2], row[3]);
    }
  }
  if (plan.vertices.size() >= 3)
    std::fprintf(stderr, "[draw-dump]   v0 %.2f %.2f %.2f\n", plan.vertices[0], plan.vertices[1], plan.vertices[2]);
  for (uint32_t r = 0; r < 3u; ++r)
    std::fprintf(stderr, "[draw-dump]   pos%u %.4f %.4f %.4f %.4f\n", r, m[r][0], m[r][1], m[r][2], m[r][3]);
  for (uint32_t i = 0; i < 8u; ++i)
    if (plan.textures[i].valid)
      std::fprintf(stderr, "[draw-dump]   tex%u @%08X fmt%u %ux%u dims=%d,%d,%d,%d\n", i, plan.textures[i].address,
                   plan.textures[i].format, plan.textures[i].width, plan.textures[i].height,
                   plan.pixel_constants.texdims[i][0], plan.pixel_constants.texdims[i][1],
                   plan.pixel_constants.texdims[i][2], plan.pixel_constants.texdims[i][3]);
  if (sh.num_ind_stages != 0)
    for (uint32_t i = 0; i < 6u; ++i)
      std::fprintf(stderr, "[draw-dump]   indmtx%u %d %d %d %d\n", i, plan.pixel_constants.indtexmtx[i][0],
                   plan.pixel_constants.indtexmtx[i][1], plan.pixel_constants.indtexmtx[i][2],
                   plan.pixel_constants.indtexmtx[i][3]);
  for (uint32_t i = 0; i < 4u; ++i)
    std::fprintf(stderr, "[draw-dump]   c%u=%d,%d,%d,%d k%u=%d,%d,%d,%d\n", i, plan.pixel_constants.colors[i][0],
                 plan.pixel_constants.colors[i][1], plan.pixel_constants.colors[i][2], plan.pixel_constants.colors[i][3],
                 i, plan.pixel_constants.kcolors[i][0], plan.pixel_constants.kcolors[i][1],
                 plan.pixel_constants.kcolors[i][2], plan.pixel_constants.kcolors[i][3]);
}

bool submit_draw_plan(const gxc::DrawPlan& plan) {
  if (!plan.ok || plan.vertex_count == 0 || plan.indices.empty()) {
    return false;
  }
  dump_draw(plan);

  if (plan.viewport_valid) {
    gx::set_logical_viewport(retail_viewport(plan.viewport));
  }
  if (plan.scissor_valid) {
    // Wind Waker draws its 2D layer (HUD, menus and the fade to and from
    // black) with a 640x479 scissor, so the frame's last line keeps the
    // previous scene through every fade. A television hides that line in its
    // overscan; an iPad shows it as a strip of the old area along the bottom
    // during transitions. Extend that full-width scissor to the full frame.
    uint32_t scissorHeight = plan.scissor_height;
    static const bool s_keepLastLine = std::getenv("DOL_GXCORE_KEEP_SCISSOR_479") != nullptr;
    if (!s_keepLastLine && plan.scissor_x == 0 && plan.scissor_y == 0 && plan.scissor_width == 640 &&
        plan.scissor_height == 479) {
      scissorHeight = 480;
    }
    gx::set_logical_scissor({
        .x = plan.scissor_x,
        .y = plan.scissor_y,
        .width = plan.scissor_width,
        .height = static_cast<int32_t>(scissorHeight),
    });
  }

  // Resolve one texmap's texture to a GPU handle: EFB-copy shadow first, else the
  // guest-identity + content-hash decode cache. Shared by the single-texmap fast
  // path and the multi-texmap path (63/Mfin, e.g. THP YUV Y/U/V on texmap 0/1/2).
  auto resolve_texture_handle =
      [&](uint32_t address, uint32_t tsize, uint32_t format, uint32_t width,
          uint32_t height, const void* data, uint32_t available, bool has_tlut,
          uint32_t tlut_address, uint32_t tlut_format, uint32_t tlut_entries,
          const void* tlut_data, uint32_t tlut_available) -> TextureHandle {
    // Only CI formats read a palette. The frontend reports whatever TLUT was
    // last bound for every texture, so without this a non-CI image drawn under
    // two different palettes had two cache identities at one address, evicted
    // itself and was decoded and uploaded again on every alternation: Wind
    // Waker's HUD did that about 23 times a frame (I4/IA4 images, same texels).
    if (!gxc::is_ci_format(format)) {
      has_tlut = false;
      tlut_address = 0u;
      tlut_format = 0u;
      tlut_entries = 0u;
      tlut_data = nullptr;
      tlut_available = 0u;
    }
    auto efbIt = g_efbCopyTextures.find(address);
    if (efbIt != g_efbCopyTextures.end() && efbIt->second.handle) {
      uint64_t memoryEpoch = 0;
      const bool memoryUnchanged =
          !efbIt->second.memory_epoch_valid ||
          (g_textureDirtyEpochObserver != nullptr &&
           g_textureDirtyEpochObserver(address, efbIt->second.byte_size,
                                       &memoryEpoch) &&
           memoryEpoch == efbIt->second.memory_epoch);
      if (memoryUnchanged) {
        ++g_textureCacheStats.hits;
        return efbIt->second.handle;
      }
      g_efbCopyTextures.erase(efbIt);
    }
    if (data == nullptr)
      return TextureHandle{};
    const auto* bytes = static_cast<const uint8_t*>(data);
    const uint32_t size = std::min(tsize, available);
    uint64_t texelEpoch = 0;
    uint64_t tlutEpoch = 0;
    const bool texelEpochValid =
        g_textureDirtyEpochObserver != nullptr &&
        g_textureDirtyEpochObserver(address, size, &texelEpoch);
    const uint32_t tlutSize =
        gxc::is_ci_format(format) && has_tlut && tlut_data != nullptr
            ? std::min(tlut_available, tlut_entries * 2u)
            : 0u;
    const bool tlutEpochValid =
        tlutSize == 0u ||
        (g_textureDirtyEpochObserver != nullptr &&
         g_textureDirtyEpochObserver(tlut_address, tlutSize, &tlutEpoch));
    auto addrIt = g_textureAddrKey.find(address);
    if (texelEpochValid && tlutEpochValid && addrIt != g_textureAddrKey.end()) {
      const TextureAddressState& previous = addrIt->second;
      const TextureKey& priorKey = previous.key;
      const bool sameIdentity =
          priorKey.address == address && priorKey.size == tsize &&
          priorKey.format == format && priorKey.width == width &&
          priorKey.height == height && priorKey.tlut_address == tlut_address &&
          priorKey.tlut_format == tlut_format &&
          priorKey.tlut_entries == tlut_entries;
      const uint64_t frame = g_textureVerifyFrame.load(std::memory_order_relaxed);
      bool small_changed = false;
      if (sameIdentity && size <= kAlwaysRehashBytes && previous.verified_frame != frame) {
        const uint64_t h = small_texture_hash(bytes, size, tlut_data, tlutSize);
        if (h != previous.small_hash)
          small_changed = true;
        else
          addrIt->second.verified_frame = frame;
      }
      if (sameIdentity && !small_changed && previous.texel_epoch_valid &&
          previous.tlut_epoch_valid && previous.texel_epoch == texelEpoch &&
          previous.tlut_epoch == tlutEpoch) {
        const TextureHandle* cached = previous.handle ? &previous.handle : nullptr;
        if (cached == nullptr) {
          if (auto found = g_textureCache.find(priorKey); found != g_textureCache.end())
            cached = &found->second;
        }
        if (cached != nullptr) {
          // DOL_GXCORE_TEX_VERIFY=1: re-hash every generation hit and report the
          // ones whose texels changed without a dirty mark (a write path the
          // dirty tracking does not see).
          static const bool s_verify = std::getenv("DOL_GXCORE_TEX_VERIFY") != nullptr;
          if (s_verify) {
            const auto* vbytes = static_cast<const uint8_t*>(data);
            if (vbytes != nullptr &&
                XXH3_64bits(vbytes, std::min(tsize, available)) != priorKey.content_hash &&
                !gxc::is_ci_format(format))
              std::fprintf(stderr, "[tex-stale] addr=%08X fmt=%u %ux%u size=%u epoch=%llu\n",
                           address, format, width, height, tsize,
                           (unsigned long long)texelEpoch);
          }
          ++g_textureCacheStats.hits;
          ++g_textureCacheStats.generation_hits;
          return *cached;
        }
      }
    } else if (!texelEpochValid || !tlutEpochValid) {
      ++g_textureCacheStats.generation_fallbacks;
    }
    ++g_textureCacheStats.hashed_lookups;
    uint64_t content_hash = XXH3_64bits(bytes, size);
    if (gxc::is_ci_format(format) && has_tlut && tlut_data != nullptr) {
      ++g_textureCacheStats.palette_hashes;
      // Resolvers report bytes available to the end of their mapped range. Only
      // the declared GX palette belongs to this texture cache identity.
      content_hash = XXH3_64bits_withSeed(tlut_data, tlutSize, content_hash);
    }
    const uint64_t small_hash =
        size <= kAlwaysRehashBytes ? small_texture_hash(bytes, size, tlut_data, tlutSize) : 0u;
    const uint64_t verify_frame = size <= kAlwaysRehashBytes
                                      ? g_textureVerifyFrame.load(std::memory_order_relaxed)
                                      : ~0ull;
    const TextureKey key{address,     tsize,        format,      width,
                         height,      tlut_address, tlut_format, tlut_entries,
                         content_hash};
    auto it = g_textureCache.find(key);
    if (it != g_textureCache.end()) {
      ++g_textureCacheStats.hits;
      g_textureAddrKey[address] = TextureAddressState{
          key, texelEpoch, tlutEpoch, texelEpochValid, tlutEpochValid,
          small_hash, verify_frame, it->second};
      return it->second;
    }
    // New content for this identity: evict any prior entry cached at the same
    // guest address (its buffer was overwritten) to keep the cache bounded.
    addrIt = g_textureAddrKey.find(address);
    if (addrIt != g_textureAddrKey.end()) {
      g_textureCache.erase(addrIt->second.key);
      addrIt->second.handle = {};
    }
    // HD texture packs (Dolphin's tex1_WxH_hash[_tlut]_fmt names): the first
    // time these guest bytes are seen, look for a replacement before decoding.
    // The handle is cached under the same content key as a decode would be,
    // so later draws of the same texture never repeat the lookup.
    bool replacementPending = false;
    if (texture_replacement::has_source_replacements()) {
      static unsigned long long s_lookups;
      static const bool s_log_misses = std::getenv("DOL_TEXREP_LOG_MISSES") != nullptr;
      ++s_lookups;
      const auto replacement = texture_replacement::find_replacement_for_guest(
          width, height, format, bytes, size, static_cast<const uint8_t*>(tlut_data),
          tlut_data != nullptr ? std::min(tlut_available, tlut_entries * 2u) : 0u,
          &replacementPending);
      if (s_log_misses && !replacementPending && !(replacement.has_value() && *replacement))
        std::fprintf(stderr, "[mods] texture miss #%llu addr=%08X fmt=%u %ux%u size=%u tlut=%u\n", s_lookups,
                     address, format, width, height, size, tlut_entries);
      if (replacement.has_value() && *replacement) {
        g_textureAwaitingReplacement.erase(key);
        ++g_textureCacheStats.uploads;
        static unsigned long long s_replaced;
        if (++s_replaced <= 3u || (s_replaced % 200u) == 0u)
          std::fprintf(stderr, "[mods] texture replaced #%llu addr=%08X fmt=%u %ux%u -> %ux%u\n",
                       s_replaced, address, format, width, height, (*replacement)->size.width,
                       (*replacement)->size.height);
        g_textureCache.emplace(key, *replacement);
        g_textureAddrKey[address] = TextureAddressState{
            key, texelEpoch, tlutEpoch, texelEpochValid, tlutEpochValid,
            small_hash, verify_frame, *replacement};
        return *replacement;
      }
      if (replacementPending) {
        if (auto it = g_textureAwaitingReplacement.find(key); it != g_textureAwaitingReplacement.end())
          return it->second;
      } else {
        g_textureAwaitingReplacement.erase(key);
      }
    }
    // gxcore owns the decode for every format: produce tightly-packed RGBA8 and
    // upload it as a pre-decoded PC texture (no substrate re-conversion).
    std::vector<uint8_t> decoded;
    if (gxc::is_ci_format(format)) {
      if (has_tlut && tlut_data != nullptr)
        decoded = gxc::decode_ci(format, width, height, bytes, size, tlut_format,
                                 tlut_entries,
                                 static_cast<const uint8_t*>(tlut_data),
                                 tlut_available);
    } else {
      decoded = gxc::decode_texture(format, width, height, bytes, size);
    }
    {
      static const bool s_upload_log = std::getenv("DOL_GXCORE_TEX_UPLOAD_LOG") != nullptr;
      if (s_upload_log)
        std::fprintf(stderr, "[tex-upload] addr=%08X fmt=%u %ux%u size=%u efb=%d prior=%d tlut=%08X hash=%016llX pfmt=%u p%ux%u psize=%u\n",
                     address, format, width, height, tsize,
                     g_efbCopyTextures.find(address) != g_efbCopyTextures.end() ? 1 : 0,
                     addrIt != g_textureAddrKey.end() ? 1 : 0, tlut_address,
                     (unsigned long long)content_hash,
                     addrIt != g_textureAddrKey.end() ? addrIt->second.key.format : 0u,
                     addrIt != g_textureAddrKey.end() ? addrIt->second.key.width : 0u,
                     addrIt != g_textureAddrKey.end() ? addrIt->second.key.height : 0u,
                     addrIt != g_textureAddrKey.end() ? addrIt->second.key.size : 0u);
    }
    {
      // DOL_GXCORE_DUMP_TEX=<hex address>: write the decoded RGBA of that
      // texture as /tmp/gxcore-tex-<address>-<n>.pam (diagnostics).
      static const long long s_dump = [] {
        const char* env = std::getenv("DOL_GXCORE_DUMP_TEX");
        return env != nullptr ? std::strtoll(env, nullptr, 16) : -1ll;
      }();
      static int s_dumped = 0;
      if (s_dump >= 0 && (address & 0x3FFFFFFFu) == (static_cast<uint32_t>(s_dump) & 0x3FFFFFFFu) &&
          !decoded.empty() && s_dumped < 4) {
        char path[128];
        std::snprintf(path, sizeof(path), "/tmp/gxcore-tex-%08X-%d.pam", address, s_dumped++);
        if (FILE* f = std::fopen(path, "wb")) {
          std::fprintf(f, "P7\nWIDTH %u\nHEIGHT %u\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n", width, height);
          std::fwrite(decoded.data(), 1, decoded.size(), f);
          std::fclose(f);
        }
      }
    }
    TextureHandle handle;
    if (!decoded.empty()) {
      handle = new_static_texture_2d(
          width, height, 1, GX_TF_RGBA8_PC,
          ArrayRef<uint8_t>{decoded.data(), decoded.size()}, false,
          "GXCore Texture");
      ++g_textureCacheStats.uploads;
      if (gxc::is_ci_format(format))
        ++g_textureCacheStats.ci_uploads;
    } else {
      // CI without a resolved palette, or a format gxcore does not decode: upload
      // the raw GX bytes under the original format (old behavior).
      handle = new_static_texture_2d(width, height, 1, format,
                                     ArrayRef<uint8_t>{bytes, size}, false,
                                     "GXCore Texture");
      ++g_textureCacheStats.uploads;
      ++g_textureCacheStats.raw_fallback;
    }
    if (replacementPending) {
      g_textureAwaitingReplacement[key] = handle;
      return handle;
    }
    g_textureCache.emplace(key, handle);
    g_textureAddrKey[address] = TextureAddressState{
        key, texelEpoch, tlutEpoch, texelEpochValid, tlutEpochValid,
        small_hash, verify_frame, handle};
    return handle;
  };

  BindGroupRef textureBindGroup = 0;
  if (plan.pipeline.shader.textured != 0) {
    if (plan.texmap_mask == 0u) {
      // Single-texmap fast path: primary texture at binding 0/1 (unchanged).
      if (plan.has_texture) {
        TextureHandle bound = resolve_texture_handle(
            plan.tex_address, plan.tex_size, plan.tex_format, plan.tex_width,
            plan.tex_height, plan.tex_data, plan.tex_available, plan.has_tlut,
            plan.tlut_address, plan.tlut_format, plan.tlut_entries,
            plan.tlut_data, plan.tlut_available);
        if (bound) {
          // Draws mostly bind the texture and sampler of the draw before. In
          // one frame (the bind group cache keeps what a frame used) the same
          // view and sampler state are the same bind group, without the
          // layout's, the sampler's and the bind group's hash and lock. The
          // generation changes when the renderer is set up again, whose frame
          // count starts over.
          struct LastBind {
            WGPUTextureView view = nullptr;
            gxc::PlanSampler sampler{};
            uint32_t frame = UINT32_MAX;
            uint32_t generation = 0;
            BindGroupRef ref = 0;
          };
          thread_local LastBind s_last;
          const gxc::PlanSampler& samplerState = plan.samplers[plan.tex_slot & 7u];
          const WGPUTextureView view = bound->sampleTextureView.Get();
          const uint32_t frame = current_frame();
          const uint32_t generation = pipeline_cache_generation();
          if (s_last.frame == frame && s_last.generation == generation && s_last.view == view &&
              std::memcmp(&s_last.sampler, &samplerState, sizeof samplerState) == 0) {
            textureBindGroup = s_last.ref;
          } else {
            const auto sampler = sampler_ref(sampler_descriptor(samplerState));
            const std::array entries{
                WGPUBindGroupEntry{.binding = 0, .textureView = view},
                WGPUBindGroupEntry{.binding = 1, .sampler = sampler.Get()},
            };
            const WGPUBindGroupDescriptor descriptor{
                .label = {"GXCore Texture Bind Group", WGPU_STRLEN},
                .layout = texture_bind_group_layout(1u).Get(),
                .entryCount = entries.size(),
                .entries = entries.data(),
            };
            textureBindGroup = bind_group_ref(descriptor);
            s_last = LastBind{view, samplerState, frame, generation, textureBindGroup};
          }
        }
      }
    } else {
      // Multi-texmap: bind each used texmap at 2t (texture) / 2t+1 (sampler),
      // matching the WGSL declarations. If any referenced texmap fails to
      // resolve the draw is not drawable (avoids a missing-bind-group error).
      // At most eight texmaps, two entries each: fixed arrays, so a draw does
      // not allocate (the title screen issues about 20,000 draws a frame).
      std::array<TextureHandle, 8> held{};
      std::array<wgpu::Sampler, 8> heldSamplers{};
      std::array<WGPUBindGroupEntry, 16> entries{};
      size_t heldCount = 0;
      size_t entryCount = 0;
      bool complete = true;
      for (uint32_t t = 0; t < 8u; ++t) {
        if ((plan.texmap_mask & (1u << t)) == 0u)
          continue;
        const gxc::PlanTexture& pt = plan.textures[t];
        TextureHandle bound;
        if (pt.valid)
          bound = resolve_texture_handle(
              pt.address, pt.size, pt.format, pt.width, pt.height, pt.data,
              pt.available, pt.has_tlut, pt.tlut_address, pt.tlut_format,
              pt.tlut_entries, pt.tlut_data, pt.tlut_available);
        if (!bound) {
          complete = false;
          break;
        }
        held[heldCount] = bound;
        heldSamplers[heldCount] = sampler_ref(sampler_descriptor(plan.samplers[t]));
        entries[entryCount++] = WGPUBindGroupEntry{
            .binding = 2u * t, .textureView = bound->sampleTextureView.Get()};
        entries[entryCount++] = WGPUBindGroupEntry{
            .binding = 2u * t + 1u, .sampler = heldSamplers[heldCount].Get()};
        ++heldCount;
      }
      if (!complete)
        return false;
      const WGPUBindGroupDescriptor descriptor{
          .label = {"GXCore Texture Bind Group", WGPU_STRLEN},
          .layout = texture_bind_group_layout(plan.texmap_mask).Get(),
          .entryCount = entryCount,
          .entries = entries.data(),
      };
      textureBindGroup = bind_group_ref(descriptor);
    }
  }

  const bool tev = plan.pipeline.shader.tev_valid != 0;
  const size_t vertBytes = plan.vertices.size() * sizeof(float);
  const size_t indexBytes = plan.indices.size() * sizeof(uint16_t);
  const size_t pixelUniformBytes = tev ? sizeof(plan.pixel_constants) : 0;
  // One comparison for the three de-duplications below (the vertex block, the
  // in-between frame's pool and its block): most draws repeat the constants of
  // the draw before them, 96 percent of the Forsaken Fortress's 17,500 a frame,
  // and each comparison of equal blocks reads all 2.8 KB of both.
  const bool repeatsLast = repeats_cached(g_vertexUniformCache, reinterpret_cast<const uint8_t*>(&plan.constants),
                                          sizeof(plan.constants));
  // In-between frames: matched on the helper thread (queued below), or here
  // while a traced frame reports each draw's outcome. Here it is matched
  // before a staging segment can split the frame; a split frame is not
  // interpolated. A frame the pacing gave no in-between frames
  // (frame_skipped) queues nothing: matching its draws would only cost the
  // CPU it was dropped for, and the frame after has no previous to blend from
  // (as after a cut).
  const bool interpolating = frame_interp::enabled() && !frame_interp::frame_skipped();
  const bool matchHere = interpolating && frame_interp::tracing();
  if (matchHere)
    wait_interp_jobs();
  static frame_interp::DrawInput tracedInput;
  if (matchHere)
    frame_interp::capture_draw(plan, tracedInput);
  const gxc::VertexShaderConstants* interpConstants =
      matchHere ? frame_interp::blend_draw(tracedInput, plan.constants, repeatsLast) : nullptr;
  // The helper is idle while a frame is traced, so its areas are ours.
  static std::vector<float> tracedVertices;
  const float* tracedPositions = matchHere ? frame_interp::blended_positions() : nullptr;
  const Range interpVertRange =
      tracedPositions != nullptr
          ? push_blended_vertices(recording_frame_slot(), plan.vertices, tracedPositions, tracedVertices)
          : Range{};
  if (frame_interp::tracing()) {
    const auto& m = plan.constants.posnormalmatrix;
    std::fprintf(stderr,
                 "[frame-interp-trace] frame=%llu %s key=%016llx prim=0x%02X fmt=%u verts=%u payload=%u idx=%d "
                 "t=(%.1f,%.1f,%.1f) s=%.3f proj00=%.3f proj32=%.1f tex=%08X bt=(%.1f,%.1f,%.1f) bt0=(%.1f,%.1f,%.1f) "
                 "direct=%d tag=%06X age=%u positions=%d\n",
                 static_cast<unsigned long long>(frame_interp::game_frame_number()), frame_interp::last_outcome(),
                 static_cast<unsigned long long>(frame_interp::draw_key(plan)),
                 plan.match_primitive, plan.match_vtx_fmt, plan.vertex_count, plan.match_payload_size,
                 plan.pipeline.shader.has_pos_mtx_idx, m[0][3], m[1][3], m[2][3],
                 std::sqrt(m[0][0] * m[0][0] + m[1][0] * m[1][0] + m[2][0] * m[2][0]), plan.constants.projection[0][0],
                 plan.constants.projection[3][2], plan.tex_address,
                 interpConstants ? interpConstants->posnormalmatrix[0][3] : 0.f,
                 interpConstants ? interpConstants->posnormalmatrix[1][3] : 0.f,
                 interpConstants ? interpConstants->posnormalmatrix[2][3] : 0.f,
                 interpConstants ? interpConstants->transformmatrices[0][3] - plan.constants.transformmatrices[0][3] : 0.f,
                 interpConstants ? interpConstants->transformmatrices[1][3] - plan.constants.transformmatrices[1][3] : 0.f,
                 interpConstants ? interpConstants->transformmatrices[2][3] - plan.constants.transformmatrices[2][3] : 0.f,
                 plan.match_direct_position ? 1 : 0, plan.draw_tag, plan.draw_tag_age,
                 tracedPositions != nullptr ? 1 : 0);
  }
  if (!staging_has_capacity(vertBytes + gxc::kVertexStrideBytes, indexBytes, sizeof(plan.constants),
                            pixelUniformBytes)) {
    if (!segment_frame() ||
        !staging_has_capacity(vertBytes + gxc::kVertexStrideBytes, indexBytes, sizeof(plan.constants),
                              pixelUniformBytes)) {
      Log.error("GXCore draw exceeds an empty Aurora staging segment");
      return false;
    }
  }

  const auto uniformRange = push_uniform_dedup(
      g_vertexUniformCache, reinterpret_cast<const uint8_t*>(&plan.constants),
      sizeof(plan.constants), repeatsLast ? 1 : 0);
  Range pixelUniformRange{};
  if (tev) {
    pixelUniformRange = push_uniform_dedup(
        g_pixelUniformCache,
        reinterpret_cast<const uint8_t*>(&plan.pixel_constants),
        sizeof(plan.pixel_constants));
  }
  const PipelineConfig colorConfig{
      .version = GXCorePipelineConfigVersion,
      .key = plan.pipeline,
      .msaaSamples = get_sample_count(),
  };
  const PipelineRef pipeline = pipeline_ref(colorConfig);
  PipelineRef depthPipeline = 0;
  if (needs_early_depth_emulation(plan.pipeline)) {
    PipelineConfig depthConfig = colorConfig;
    depthConfig.depthOnly = 1u;
    depthPipeline = pipeline_ref(depthConfig);
  }
  const bool ownVertices = plan.draw_tag != 0 || plan.draw_scope_part != 0;

  // Batching: a draw of the state of the pass's last command (pipeline,
  // constants, pixel constants, textures), with nothing between them, whose
  // vertices and indices follow its, extends it, its indices counted from the
  // batch's first vertex. Grass, rocks, a particle system's quads and a wake's
  // strips are runs of such draws: 65 percent of Adanmae's 10,000 draws a
  // frame repeat the constants of the draw before. In-between frames draw a
  // batch as one where its draws' blocks came out the same (render()).
  static const bool batching = [] {
    const char* env = std::getenv("DOL_AURORA_GXCORE_BATCH");
    return env == nullptr || env[0] != '0';
  }();
  const size_t vertexOffset = next_vertex_offset(gxc::kVertexStrideBytes);
  DrawData* batch = nullptr;
  uint32_t firstVertex = 0;
  if (batching && !matchHere) {
    DrawData* last = last_recorded_draw();
    if (last != nullptr && last->pipeline == pipeline && last->depthPipeline == depthPipeline &&
        last->uniformRange.offset == uniformRange.offset && last->tev == tev &&
        (!tev || last->pixelUniformRange.offset == pixelUniformRange.offset) &&
        last->textureBindGroup == textureBindGroup && last->ownVertices == ownVertices &&
        last->interpUniformRange.size == 0 && last->interpVertRange.size == 0 &&
        (interpolating ? last->interpJob != UINT32_MAX && last->interpJob + last->batchSize == next_interp_job()
                       : last->interpJob == UINT32_MAX) &&
        last->vertRange.offset % gxc::kVertexStrideBytes == 0 &&
        last->vertRange.offset + last->vertRange.size == vertexOffset &&
        last->idxRange.offset + last->idxRange.size == next_index_offset()) {
      const size_t first = (vertexOffset - last->vertRange.offset) / gxc::kVertexStrideBytes;
      if (first + plan.vertex_count <= 65536u) {
        batch = last;
        firstVertex = static_cast<uint32_t>(first);
      }
    }
  }

  // On a whole vertex: render() draws it at a base vertex of the whole buffer.
  const auto vertRange = push_verts_strided(
      reinterpret_cast<const uint8_t*>(plan.vertices.data()),
      vertBytes, gxc::kVertexStrideBytes);
  Range idxRange;
  if (batch != nullptr) {
    static std::vector<uint16_t> rebased;
    rebased.resize(plan.indices.size());
    for (size_t i = 0; i < rebased.size(); ++i)
      rebased[i] = static_cast<uint16_t>(plan.indices[i] + firstVertex);
    idxRange = push_indices(reinterpret_cast<const uint8_t*>(rebased.data()), indexBytes, 2);
  } else {
    idxRange = push_indices(reinterpret_cast<const uint8_t*>(plan.indices.data()), indexBytes, 2);
  }
  Range interpUniformRange{};
  uint32_t interpJob = UINT32_MAX;
  if (interpConstants != nullptr)
    interpUniformRange = push_interp_uniform_dedup(
        g_interpUniformCache, current_frame_id(), recording_frame_slot(), reinterpret_cast<const uint8_t*>(interpConstants),
        sizeof(*interpConstants), frame_interp::last_blend_repeated());
  else if (interpolating && !matchHere)
    interpJob = queue_interp_job(plan, repeatsLast);

  const auto indexCount = static_cast<uint32_t>(plan.indices.size());
  if (batch != nullptr) {
    if (batch->batchSize == 1)
      batch->batch = push_batch_draw({.indexCount = batch->indexCount, .firstVertex = 0});
    push_batch_draw({.indexCount = indexCount, .firstVertex = firstVertex});
    ++batch->batchSize;
    batch->indexCount += indexCount;
    batch->idxRange.size += idxRange.size;
    batch->vertRange.size = static_cast<uint32_t>(vertRange.offset + vertRange.size - batch->vertRange.offset);
    ++g_mergedDrawCallCount;
    return true;
  }
  push_draw_command(DrawData{
      .pipeline = pipeline,
      .depthPipeline = depthPipeline,
      .vertRange = vertRange,
      .idxRange = idxRange,
      .uniformRange = uniformRange,
      .interpUniformRange = interpUniformRange,
      .interpVertRange = interpVertRange,
      .interpJob = interpJob,
      .pixelUniformRange = pixelUniformRange,
      .indexCount = indexCount,
      .textureBindGroup = textureBindGroup,
      .tev = tev,
      .ownVertices = ownVertices,
  });
  return true;
}

} // namespace aurora::gfx::gxcore

void aurora_set_forced_anisotropy(unsigned samples) {
  aurora::gfx::gxcore::g_forcedAnisotropy.store(
      static_cast<std::uint16_t>(std::clamp(samples, 1u, 16u)), std::memory_order_relaxed);
}
