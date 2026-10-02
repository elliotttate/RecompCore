// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "gxruntime/aurora_recomp/render_sink.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

extern "C" {
#include "gxruntime/gx_recomp.h"
}

namespace gxruntime::aurora_recomp {

struct DrawTransformSnapshot {
  std::uint32_t transform_flags = 0;
  std::uint32_t current_pn_matrix = 0;
  std::uint32_t payload_pn_matrix_mask = 0;
  std::uint32_t position_matrix_valid_mask = 0;
  float viewport[6]{};
  float projection[6]{};
  std::uint32_t projection_type = 0;
  float position_matrices[DOL_GX_RECOMP_POSITION_MATRIX_COUNT]
                         [DOL_GX_RECOMP_POSITION_MATRIX_WORDS]{};
  float normal_matrices[DOL_GX_RECOMP_NORMAL_MATRIX_COUNT]
                       [DOL_GX_RECOMP_NORMAL_MATRIX_WORDS]{};
  std::uint16_t normal_matrix_word_mask[DOL_GX_RECOMP_NORMAL_MATRIX_COUNT]{};
  // XF lighting state at draw time (raw words; per-word validity in the
  // masks). Same layout as DolGxRecompState — see gx_recomp.h.
  std::uint32_t light_words[DOL_GX_RECOMP_LIGHT_COUNT]
                           [DOL_GX_RECOMP_LIGHT_WORDS]{};
  std::uint16_t light_word_mask[DOL_GX_RECOMP_LIGHT_COUNT]{};
  std::uint32_t chan_regs[DOL_GX_RECOMP_CHAN_REG_COUNT]{};
  std::uint32_t chan_reg_mask = 0;
  // Texture matrices + raw XF register window at draw time (gxcore
  // texgen inputs). Layouts match DolGxRecompState — see gx_recomp.h.
  float tex_matrices[DOL_GX_RECOMP_TEX_MATRIX_COUNT]
                    [DOL_GX_RECOMP_TEX_MATRIX_WORDS]{};
  std::uint16_t tex_matrix_word_mask[DOL_GX_RECOMP_TEX_MATRIX_COUNT]{};
  std::uint32_t xf_regs[DOL_GX_RECOMP_XF_REG_COUNT]{};
  std::uint64_t xf_reg_mask = 0;
  // Dual-texture post-transform at draw time (see RenderDrawPacket).
  std::uint8_t post_tex_mask = 0;
  std::uint8_t post_tex_normalize = 0;
  float post_tex_rows[8][12]{};
};

// XF post-transform ("dual texture") matrix memory, 0x500..0x5FF: 64 rows of
// four floats. Kept beside DolGxRecompState rather than in it, so the saved
// front-end state keeps its size; after a load the rows read as identity
// until the game loads them again.
struct PostTexMatrices {
  float rows[64][4]{};
  std::uint64_t written = 0;  // rows the game has loaded
  // identity[k] bit r: row r is row k of the identity (or was never loaded),
  // so a texgen whose three post rows all read as identity needs no transform.
  std::uint64_t identity[3] = {~0ull, ~0ull, ~0ull};
  bool off = false; // XF 0x1012 written 0; GXInit writes 1
};

class RetailGxFrontend {
public:
  RetailGxFrontend();
  explicit RetailGxFrontend(const DolGuestAddressResolver& resolver);

  void reset(const DolGuestAddressResolver* resolver = nullptr);

  bool set_vertex_layout(std::uint8_t vtx_fmt, std::uint32_t vertex_size);
  bool set_indexed_attr(std::uint8_t vtx_fmt, std::uint8_t attr,
                        std::uint32_t vertex_offset, std::uint8_t index_size,
                        std::uint32_t element_size,
                        std::uint32_t element_bias);
  bool derive_vertex_layout(std::uint8_t vtx_fmt);
  bool set_cp_array(std::uint8_t attr, std::uint32_t physical_base,
                    std::uint8_t stride);
  bool load_cp_reg(std::uint8_t reg, std::uint32_t value);
  void set_packet_drain_enabled(bool enabled);
  // Stop a FIFO parse right after a display copy (GXCopyDisp), leaving the
  // rest buffered for the next flush. A host that presents at the display copy
  // needs the frame to end exactly there: draws after it belong to the next
  // frame. display_copy_stopped() reports that the last flush stopped so.
  void set_stop_at_display_copy(bool enabled) { stop_at_display_copy_ = enabled; }
  bool display_copy_stopped() const { return display_copy_stopped_; }

  // Read-only tap on the decode event stream (histograms): invoked
  // for every trace event a successful flush/write_display_list/replay_fifo
  // produced, before drain reclaims the ring. Never affects decode.
  using TraceEventObserver = void (*)(const DolGxRecompTraceEvent& event,
                                      void* user);
  void set_event_observer(TraceEventObserver observer, void* user) {
    event_observer_ = observer;
    event_observer_user_ = user;
  }

  bool write_fifo(std::span<const std::uint8_t> bytes);
  bool flush(AuroraRenderSink* sink = nullptr);
  // Parse a display list from caller-provided bytes through the SAME internal
  // path as the in-stream 0x40 CALL_DL opcode, minus the guest resolution
  // (the bytes are already host-visible). Entry point for trace replay
  // (CALL_DL records) and the backend's DL mirror; keep it
  // byte-path-identical to the opcode branch.
  bool write_display_list(std::span<const std::uint8_t> bytes,
                          AuroraRenderSink* sink = nullptr);
  std::size_t pending_fifo_size() const { return fifo_buffer_.size(); }

  bool replay_fifo(std::span<const std::uint8_t> bytes,
                   AuroraRenderSink* sink = nullptr);

  const DolGxRecompState& state() const { return state_; }
  DolGxRecompState& state() { return state_; }

  // Save states. save_state is the register model (CP/XF/BP, TMEM palettes,
  // texture and copy bindings) plus the bytes of a command the guest has not
  // finished writing; take it with every buffered byte parsed (the backend
  // drains its worker first). load_state puts it back under this front end's
  // own resolver, resolving the texture, palette and copy ranges again against
  // the (restored) guest memory; a range that no longer resolves is dropped.
  std::vector<std::uint8_t> save_state() const;
  bool load_state(const std::uint8_t* data, std::size_t size);

  std::span<const DolGxRecompTraceEvent> trace_events() const;
  // Cumulative zero-vertex draw headers consumed as no-ops (s56 conformance:
  // the frontend emits no Draw packet for them, but Aurora's live decoder
  // counts each unmerged one in drawCallCount). Replay adds this to its draw
  // count when gating against recorded PRESENT_STATS.
  std::uint64_t zero_vertex_draws() const { return zero_vertex_draws_; }
  std::uint64_t display_copies() const { return display_copies_.load(std::memory_order_relaxed); }
  const char* last_error() const { return last_error_; }
  std::size_t last_error_offset() const { return last_error_offset_; }
  std::uint8_t last_error_opcode() const { return last_error_opcode_; }
  std::uint32_t last_error_a() const { return last_error_a_; }
  std::uint32_t last_error_b() const { return last_error_b_; }
  std::uint32_t last_error_c() const { return last_error_c_; }
  std::uint32_t last_error_d() const { return last_error_d_; }

private:
  bool fail_parse(const char* reason, std::uint8_t opcode,
                  std::size_t offset, std::uint32_t a = 0,
                  std::uint32_t b = 0, std::uint32_t c = 0,
                  std::uint32_t d = 0);
  bool parse_stream(std::span<const std::uint8_t> bytes, bool allow_partial,
                    bool record_fifo_bytes, std::uint32_t depth,
                    std::size_t* consumed);
  bool handle_bp(std::uint32_t raw);
  bool maybe_resolve_texture(std::uint8_t slot);
  bool handle_copy_trigger(std::uint32_t value);
  bool handle_draw(std::uint8_t command,
                   std::span<const std::uint8_t> vertex_data,
                   std::uint16_t vertex_count,
                   std::size_t command_offset);
  bool emit_new_packets(AuroraRenderSink& sink, std::uint32_t first_event);
  void drain_emitted_packets(std::uint32_t emitted_count);
  // Drain mode only: hand every pending event to the sink and reclaim the
  // trace, so one large batch cannot overflow it (see parse_stream).
  bool emit_and_drain(AuroraRenderSink& sink);
  AuroraRenderSink* parse_sink_ = nullptr;
  bool stop_at_display_copy_ = false;
  bool display_copy_stopped_ = false;

  DolGxRecompState state_{};
  PostTexMatrices post_tex_{};
  std::vector<std::uint8_t> fifo_buffer_;
  // Raw per-vertex payload bytes for each parsed draw, in trace order. Popped in
  // lockstep as Draw events are emitted (emit_new_packets) so each Draw packet
  // references its own bytes; cleared on full drain/reset. Each element is its
  // own buffer, so the outer vector reallocating during parse does not move the
  // inner heap storage the emitted packets point at.
  std::vector<std::vector<std::uint8_t>> draw_payload_queue_;
  std::vector<DrawTransformSnapshot> draw_transform_queue_;
  std::size_t draw_queue_count_ = 0; // live slots; the vectors keep capacity
  std::size_t draw_payload_head_ = 0;
  std::size_t draw_transform_head_ = 0;
  void notify_events(std::uint32_t first_event);

  bool packet_drain_enabled_ = false;
  std::uint64_t zero_vertex_draws_ = 0;
  std::uint64_t unknown_opcodes_skipped_ = 0;
  std::uint64_t display_lists_skipped_ = 0;
  std::atomic<std::uint64_t> display_copies_{0};
  TraceEventObserver event_observer_ = nullptr;
  void* event_observer_user_ = nullptr;
  std::uint32_t emitted_trace_count_ = 0;
  // Texture slots (bit per slot) whose palette TMEM was reloaded after the
  // texture event was emitted; re-resolved at the next draw.
  std::uint8_t tlut_stale_mask_ = 0;
  std::uint64_t next_packet_sequence_ = 0;
  // emit_new_packets' reused packet (see there).
  RenderPacket scratch_packet_{};
  bool scratch_draw_dirty_ = false;
  const char* last_error_ = nullptr;
  std::size_t last_error_offset_ = 0;
  std::uint8_t last_error_opcode_ = 0;
  std::uint32_t last_error_a_ = 0;
  std::uint32_t last_error_b_ = 0;
  std::uint32_t last_error_c_ = 0;
  std::uint32_t last_error_d_ = 0;
};

} // namespace gxruntime::aurora_recomp
