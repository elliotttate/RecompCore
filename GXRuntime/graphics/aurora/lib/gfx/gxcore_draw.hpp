#pragma once

// gxcore substrate integration (63/S12): the 4th per-type draw module beside
// clear/gx/rmlui. Pipelines and WGSL come from the headless
// gxruntime::gxcore lib; this file owns only the wgpu plumbing (pipeline
// descriptor, bind groups, buffer pushes, draw submission).

#include "common.hpp"

#include <gxruntime/gxcore/shader.hpp>

#include <webgpu/webgpu_cpp.h>

namespace aurora::gfx::gxcore {

// Retail GXSetViewport writes a 342-pixel XF origin bias. Aurora's native
// GX API has a paired 340 encoder/decoder convention; do not use that
// convention for FIFO written by the original game. A two-pixel error here
// shifts geometry once, then shifts an EFB-copy postprocess over it again.
inline Viewport retail_viewport(const float (&raw)[6]) {
  return {
      .left = raw[3] - 342.f - raw[0],
      .top = raw[4] - 342.f + raw[1],
      .width = raw[0] * 2.f,
      .height = -raw[1] * 2.f,
      .znear = (raw[5] - raw[2]) / 16777215.f,
      .zfar = raw[5] / 16777215.f,
  };
}

struct DrawData {
  PipelineRef pipeline;
  PipelineRef depthPipeline; // early-Z depth-only pass, 0 when unnecessary
  Range vertRange;
  Range idxRange;
  VertexUniformRanges uniformRange; // VertexShaderConstants' parts (group 1)
  // The constants blended toward the previous frame, used while the
  // in-between frame is encoded (frame_interp.hpp); uniformRange otherwise.
  VertexUniformRanges interpUniformRange;
  // And its vertices, for a particle whose positions came in its payload,
  // blended toward the previous frame's (in the in-between vertex area);
  // vertRange otherwise.
  Range interpVertRange;
  // Or the helper thread's job that makes both (interp_job_ranges), when the
  // draw was matched off the recording thread; UINT32_MAX when not.
  uint32_t interpJob = UINT32_MAX;
  Range pixelUniformRange;  // PixelShaderConstants (group 2), TEV path only
  uint32_t indexCount;
  BindGroupRef textureBindGroup; // 0 when untextured
  bool tev = false; // TEV path: PS uniform at group 2, texture at group 3
  // A tagged particle's, whose in-between vertices are its own (interpVertRange).
  bool ownVertices = false;
  // Its vertices are in the full layout (gxruntime/gxcore/shader.hpp); the
  // compact one otherwise, with the full layout's other inputs read from
  // the default stream.
  bool fullVertices = false;
  // The ubershader's pipeline for this draw's fixed-function state, its
  // pixel block (the pixel constants and the shader key) and a bind group with
  // all eight texmaps, when the draw's own pipeline was still compiling as it
  // was recorded: render() draws it with them if its own is not ready yet.
  PipelineRef uberPipeline = 0;
  Range uberPixelRange;
  BindGroupRef uberTextureBindGroup = 0;
  // Consecutive draws of one state whose vertices and indices follow on are
  // one draw (submit_draw_plan); an in-between frame whose blocks for them
  // differ draws them one by one: their parts are batch_draw(batch + i), their
  // jobs interpJob + i.
  uint32_t batch = UINT32_MAX;
  uint32_t batchSize = 1;
};

// Recording: the pass's last command, when it is a GXCore draw a following
// draw may extend; nullptr otherwise.
DrawData* last_recorded_draw() noexcept;

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
constexpr uint32_t GXCorePipelineConfigVersion = 12;

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
// At a render pass's start, and after anything else draws in it: what
// render() last bound is not known.
void reset_pass_state();

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
