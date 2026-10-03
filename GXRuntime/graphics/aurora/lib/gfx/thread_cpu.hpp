#pragma once

#include <cstdint>

// The CPU time of the graphics threads, for the per-second frame diagnostics:
// which of them is the one holding the game back on a slower PC. Each thread
// registers itself when it starts; the reads come from another thread.
namespace aurora::gfx::thread_cpu {

enum class Role : uint32_t {
  GxWorker,     // the GX FIFO translation worker
  InterpHelper, // Smooth Motion's matching and blending
  RenderWorker, // Aurora's encoding and presenting
  GxSubmit,     // the GX worker's second stage: the draw plans into Aurora
  Count,
};

void register_current(Role role);
// Microseconds of CPU time the thread in that role has used, 0 if none has
// registered.
uint64_t cpu_us(Role role);

} // namespace aurora::gfx::thread_cpu
