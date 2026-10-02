#pragma once

// In-between frames for a 30 FPS game on a 60 Hz display.
//
// The game keeps its own 30 steps a second; only the renderer changes. Every
// gxcore draw carries its complete transform state (VertexShaderConstants:
// position, normal, projection and texture matrices, lights), and its vertices
// stay in object space. So an in-between frame is the finished frame drawn a
// second time with each draw's constants blended halfway toward the same draw
// in the previous frame, and nothing else recorded again.
//
// Recording (the GX translation thread): each draw is keyed by its raw vertex
// stream (the indices or direct attributes of the FIFO payload, which a model
// sends identically every frame) and its occurrence of that key in the frame.
// blend_draw() matches it to the previous frame, rejects a pair that is not
// plausibly the same object a frame apart, and returns the blended constants,
// which the draw pushes as a second uniform block. A frame where too many
// matches are rejected is a camera cut and is not interpolated. Copies of one
// model (grass, bushes, trees) share a key and are culled one by one, so
// their order shifts: they are matched by where the camera's motion, which
// the draws with a unique key vote for, carries each copy of the frame before.
//
// Presenting (the render worker): the finished frame is copied aside, its
// passes are encoded once more with the blended blocks and presented at once,
// and the real frame follows half a game frame later.

#include <array>
#include <cstdint>
#include <vector>

#include <gxruntime/gxcore/shader.hpp>

namespace aurora::gfx::frame_interp {

bool enabled() noexcept;
void set_enabled(bool enabled) noexcept;

// In-between frames per game frame: 1 (60 Hz from 30) or 3 (120 Hz), at
// t = step / (steps + 1). A change takes effect at the next game frame;
// frame_steps() is the recording frame's.
// In-between frames a game frame, at most: 7 shows 240 a second (a 240 Hz
// display), 3 shows 120 and 1 shows 60.
constexpr int kMaxSteps = 7;
void set_steps(int steps) noexcept;
int steps() noexcept;
int frame_steps() noexcept;
// Rendering fell behind (the GPU a game frame behind, the game waiting on the
// render worker, a long wait for a drawable): the next game frames get fewer
// in-between frames, or none (frame_skipped), until it has been calm a while.
void note_overload(const char* why) noexcept;
bool frame_skipped() noexcept;
// A save state was loaded: the next game frame gets no in-between frames, as
// the frame before it is not the one it follows.
void request_cut() noexcept;

// The key for a draw's vertex stream; 0 when the plan carries no payload.
uint64_t draw_key(const gxruntime::gxcore::DrawPlan& plan) noexcept;

// For a draw with per-vertex matrix indices (a skinned model): bit r set when
// a vertex uses the position matrix starting at XF row r. 0 otherwise.
uint64_t used_matrix_rows(const gxruntime::gxcore::DrawPlan& plan) noexcept;

// What blend_draw() needs of a draw besides its constants, captured from its
// plan on the recording thread (the draw may be matched later, on another
// thread): its key and indexed matrix rows, three of its vertices (for a
// model the game skins on the CPU; see vertex_motion()), and a tagged
// particle's positions (x, y, z per decoded vertex) and age.
struct DrawInput {
  uint64_t key = 0;
  uint64_t usedMatrixRows = 0;
  bool haveSamples = false;
  std::array<float, 9> samples{};
  std::vector<float> positions;
  uint32_t age = 0;
};
void capture_draw(const gxruntime::gxcore::DrawPlan& plan, DrawInput& out) noexcept;

// Once per submitted gxcore draw, in draw order, on one thread at a time.
// Returns the constants the draw uses in the in-between frame, or nullptr
// when they are its own (no match, an implausible match, or nothing moved).
// The pointer is valid until the next call. repeats_last_draw: `current` is
// the constants of the draw before this one (the caller already compared
// them), so they are not compared again, and an in-between block made from
// the same match is reused. pixel: a TEV draw's pixel constants, whose
// colours are blended with its counterpart's (blended_pixel()).
const gxruntime::gxcore::VertexShaderConstants* blend_draw(const DrawInput& input,
                                                           const gxruntime::gxcore::VertexShaderConstants& current,
                                                           bool repeats_last_draw = false,
                                                           const gxruntime::gxcore::PixelShaderConstants* pixel = nullptr);
// A draw known by its key and rows alone (no samples or positions).
const gxruntime::gxcore::VertexShaderConstants* blend_draw(
    uint64_t key, uint64_t used_matrix_rows,
    const gxruntime::gxcore::VertexShaderConstants& current, bool repeats_last_draw = false);
// After a blend_draw() that returned a block: each step's (the returned
// pointer is step 0's), valid until the next call.
const gxruntime::gxcore::VertexShaderConstants* blended_step(int step) noexcept;

// For a draw whose positions came in its payload (particles, the sword's
// trail): after blend_draw(), a step's in-between positions (x, y, z per
// decoded vertex), valid until the next call, or nullptr when it draws its own.
const float* blended_positions(int step = 0) noexcept;

// After a blend_draw() given pixel constants: a step's, with the TEV colour
// and konst registers and the fog colour blended toward the counterpart's
// (a fade, a particle's colour over its life, a flash), valid until the next
// call, or nullptr when they are its own (no counterpart, or the same).
const gxruntime::gxcore::PixelShaderConstants* blended_pixel(int step) noexcept;

// Whether the last blend_draw() returned the same in-between block as the call
// before it (its bytes unchanged), so the caller need not compare them.
bool last_blend_repeated() noexcept;

// Why the last blend_draw() returned what it did, for the per-draw trace
// (DOL_AURORA_FRAME_INTERP_TRACE=game frame number).
const char* last_outcome() noexcept;
bool tracing() noexcept;

// Recording thread, when a presenting frame ends: whether it gets an
// in-between frame.
bool frame_verdict() noexcept;

// At every present (a game frame boundary), with recording stopped.
void end_game_frame() noexcept;

// Render worker: set while the in-between frame's passes are encoded.
bool encoding_interpolated() noexcept;
void set_encoding_interpolated(bool value) noexcept;

// Debug dumps (DOL_AURORA_FRAME_INTERP_DUMP=dir, _FROM/_TO = game frame
// numbers): the render worker writes the real and in-between images there.
const char* dump_directory() noexcept;
bool dump_frame(uint64_t frame) noexcept;
uint64_t game_frame_number() noexcept;

} // namespace aurora::gfx::frame_interp
