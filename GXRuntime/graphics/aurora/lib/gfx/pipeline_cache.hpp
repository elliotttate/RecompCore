#pragma once

#include "common.hpp"

#include <functional>

namespace aurora::gfx::clear {
struct PipelineConfig;
} // namespace aurora::gfx::clear

namespace aurora::gx {
struct PipelineConfig;
} // namespace aurora::gx

namespace aurora::rmlui {
struct PipelineConfig;
} // namespace aurora::rmlui

namespace aurora::gfx {

using NewPipelineCallback = std::function<wgpu::RenderPipeline()>;

void initialize_pipeline_cache();
void shutdown_pipeline_cache();
// Changes each time the cache is initialized or shut down: a pipeline reference
// remembered from before names nothing the cache still knows.
uint32_t pipeline_cache_generation();
void begin_pipeline_frame();
void end_pipeline_frame();

template <typename Config>
PipelineRef find_pipeline(ShaderType type, const Config& config, NewPipelineCallback&& cb);

bool get_pipeline(PipelineRef ref, wgpu::RenderPipeline& pipeline);
// Whether a pipeline find_pipeline() returned is compiled (cheap when none is
// compiling).
bool pipeline_ready(PipelineRef ref);
// A gxcore draw whose own pipeline was not ready: drawn with the ubershader,
// or left out of its frame (for the once-a-second report; a batch counts its
// draws).
void note_ubershader_draw();
void note_draw_left_out(uint32_t draws);
// The ubershader (gxcore_uber.cpp): 0 off, 1 for the draws whose own pipeline
// is still compiling, 2 for every draw (testing). DOL_AURORA_UBERSHADER sets
// it; the default is 1 on D3D12, where it was tested, and 0 elsewhere.
int ubershader_mode();

} // namespace aurora::gfx
