#pragma once

// gxcore substrate integration (63/S12): the 4th per-type draw module beside
// clear/gx/rmlui. Pipelines and WGSL come from the headless
// gxruntime::gxcore lib; this file owns only the wgpu plumbing (pipeline
// descriptor, bind groups, buffer pushes, draw submission).

#include "common.hpp"

#include <gxruntime/gxcore/shader.hpp>

#include <webgpu/webgpu_cpp.h>

namespace aurora::gfx::gxcore {

struct DrawData {
  PipelineRef pipeline;
  PipelineRef depthPipeline; // early-Z depth-only pass, 0 when unnecessary
  Range vertRange;
  Range idxRange;
  Range uniformRange;       // VertexShaderConstants (group 1)
  // The constants blended toward the previous frame, used while the
  // in-between frame is encoded (frame_interp.hpp); uniformRange otherwise.
  Range interpUniformRange;
  // Or the helper thread's job that makes them (interp_job_range), when the
  // draw was matched off the recording thread; UINT32_MAX when not.
  uint32_t interpJob = UINT32_MAX;
  Range pixelUniformRange;  // PixelShaderConstants (group 2), TEV path only
  uint32_t indexCount;
  uint32_t vertexStride = gxruntime::gxcore::kVertexStrideBytes;
  bool packedVertices = false;
  BindGroupRef textureBindGroup; // 0 when untextured
  bool tev = false; // TEV path: PS uniform at group 2, texture at group 3
};

inline bool can_merge_draws(const DrawData& a, const DrawData& b) {
  const uint32_t stride = a.vertexStride;
  return stride != 0 && a.vertexStride == b.vertexStride && a.packedVertices == b.packedVertices &&
         a.depthPipeline == 0 && b.depthPipeline == 0 && a.pipeline == b.pipeline &&
         a.textureBindGroup == b.textureBindGroup && a.tev == b.tev && a.uniformRange == b.uniformRange &&
         a.interpUniformRange == b.interpUniformRange && a.pixelUniformRange == b.pixelUniformRange &&
         // A helper-thread in-between job makes one draw's blended constants.
         a.interpJob == UINT32_MAX && b.interpJob == UINT32_MAX &&
         a.vertRange.offset + a.vertRange.size == b.vertRange.offset &&
         a.idxRange.offset + a.idxRange.size == b.idxRange.offset && a.vertRange.size % stride == 0 &&
         b.vertRange.size % stride == 0 && (uint64_t(a.vertRange.size) + b.vertRange.size) / stride <= 65535u;
}

// Bump when generate_wgsl output or the DrawData/vertex layout changes: the
// persisted pipeline cache precompiles stored configs at startup, and a
// layout/semantics drift under an unchanged key is the S8 poisoned-cache
// failure shape. v2: S14 TEV path (PS uniform group 2, texture group 3).
// v3: S15 lit vertex layout. v4: S16 fog (PixelShaderConstants grows +
// fog fragment on the TEV path). v5: Mfin multi-texmap (a TEV combining >1 texmap
// now emits per-texmap samplers + a wider texture bind group under an unchanged
// key, so a persisted v4 pipeline for such a draw is stale). v6: indirect TEV
// stages add shader structure, pixel constants, and indirect texmap bindings.
// v7: fifth texgen adds UV4 and a high per-vertex matrix-index word. v8:
// destination-alpha override adds a dual-source fragment output/blend state.
// v9: GXSetZCompLoc early depth adds a depth-only pipeline variant. v10:
// RGB and alpha blending carry donor-exact independent factors. v11: BP SU
// texture-coordinate scales enlarge pixel constants and alter TEV WGSL.
// v12: shared texture layouts. v13: optional packed-vertex storage fetch.
constexpr uint32_t GXCorePipelineConfigVersion = 13;

struct PipelineConfig {
  uint32_t version = GXCorePipelineConfigVersion;
  gxruntime::gxcore::PipelineKey key;
  uint32_t msaaSamples = 1;
  uint32_t depthOnly = 0;
};
static_assert(std::has_unique_object_representations_v<PipelineConfig>);

wgpu::RenderPipeline create_pipeline(const PipelineConfig& config);
bool needs_early_depth_emulation(
    const gxruntime::gxcore::PipelineKey& key);
void render(const DrawData& data, const wgpu::RenderPassEncoder& pass);

// Perform one EFB copy-to-texture (63/S16): resolve the current EFB region into
// a texture keyed by the copy destination address, so a later draw binding that
// address samples the copied EFB instead of stale guest memory. Called by the
// GxCoreSink copy observer at the copy's stream position (pending draw flushed).
void copy_efb_to_texture(const gxruntime::gxcore::EfbCopyCommand& cmd);

// Submission layer: turn one headless DrawPlan into buffer pushes, texture
// upload (guest-identity cache keyed incl. TLUT identity; gxcore decodes every
// format to RGBA8, S13 A3), viewport state, and a queued draw command on the
// current pass. False = plan not drawable.
bool submit_draw_plan(const gxruntime::gxcore::DrawPlan& plan);
// Waits until the helper thread has matched and blended every draw submitted
// so far (end_frame calls it before a frame packet is handed on).
void wait_interp_jobs();
// Drop the texture cache + reset its stats (start of a replay run).
void reset_texture_cache();
// Once per presented frame: small textures are re-hashed at most once a frame.
void note_frame_presented();

using TextureDirtyEpochObserver =
    bool (*)(uint32_t address, uint32_t size, uint64_t* epoch);
void set_texture_dirty_epoch_observer(TextureDirtyEpochObserver observer);

// Texture-cache telemetry (S13 A3): proves the no-reconvert property. `uploads`
// counts actual decode+upload (cache miss); `hits` counts cache reuse (no
// re-decode) — on a static replay scene `uploads` must go to 0 after warm-up.
// `ci_uploads` = CI textures decoded through a TLUT; `raw_fallback` = decode
// produced nothing (CI without a resolved palette / unsupported) so the raw GX
// bytes were uploaded under the original format instead. Hash counters expose
// live cache-key work separately from decode/upload work.
struct TextureCacheStats {
  unsigned long long uploads = 0;
  unsigned long long hits = 0;
  unsigned long long ci_uploads = 0;
  unsigned long long raw_fallback = 0;
  unsigned long long hashed_lookups = 0;
  unsigned long long palette_hashes = 0;
  unsigned long long generation_hits = 0;
  unsigned long long generation_fallbacks = 0;
};
const TextureCacheStats& texture_cache_stats();

} // namespace aurora::gfx::gxcore
