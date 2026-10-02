// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstddef>
#include <vector>

// gxcore consumer side: accumulates the normalized packet stream's
// register state (BP regs, CP VCD/VAT) and turns each span-complete
// ConsumedDraw into a DrawPlan — decoded vertices (Dolphin VertexLoader
// semantics), topology indices, packed uniforms, pipeline key. Headless: the
// GPU submission layer lives in the fork (lib/gfx/gxcore_draw.*).

#include "gxruntime/gxcore/shader.hpp"

#include "gxruntime/aurora_recomp/render_sink.hpp"

#include <cstdint>

namespace gxruntime::gxcore {

// Loud-counter taxonomy for everything the slice does not implement yet.
// Consumers print non-zero counters after a replay; a growing counter is the
// demand signal that schedules the S13-S16 module work.
struct GapCounters {
  unsigned long long draws_planned = 0;
  unsigned long long draws_skipped = 0;         // plan.ok == false
  unsigned long long draws_noop = 0;            // valid incomplete primitive
  unsigned long long cull_all_draws = 0;        // culled by state, not a gap
  unsigned long long missing_vcd = 0;           // draw before VCD/VAT seen
  unsigned long long vertex_decode_failures = 0;
  unsigned long long vertex_projection_missing = 0;
  unsigned long long vertex_payload_empty = 0;
  unsigned long long vertex_walk_underivable = 0;
  unsigned long long vertex_stride_mismatch = 0;
  unsigned long long vertex_topology_unsupported = 0;
  unsigned long long topology_zero_quads = 0;
  unsigned long long topology_zero_triangles = 0;
  unsigned long long topology_zero_triangle_strip = 0;
  unsigned long long topology_zero_triangle_fan = 0;
  unsigned long long topology_zero_lines = 0;
  unsigned long long topology_zero_line_strip = 0;
  unsigned long long topology_zero_points = 0;
  unsigned long long topology_zero_unknown = 0;
  unsigned long long vertex_payload_overrun = 0;
  unsigned long long vertex_array_unresolved = 0;
  unsigned long long vertex_array_out_of_bounds = 0;
  unsigned long long unsupported_texgen = 0;    // sum of actual texgen gaps below
  unsigned long long texgen_count_overflow = 0; // XF count > captured shader cap
  unsigned long long texgen_count_5 = 0;
  unsigned long long texgen_count_6 = 0;
  unsigned long long texgen_count_7 = 0;
  unsigned long long texgen_count_8plus = 0;
  unsigned long long texgen_emboss_cached_nbt = 0; // supported cached N/B/T fallback
  unsigned long long per_vertex_normal_matrix = 0; // PNMTXIDX selects normal bank
  unsigned long long texgen_source_normal = 0;  // supported regular normal source
  unsigned long long texgen_source_normal_default = 0; // no normal attribute
  unsigned long long texgen_source_colors = 0;  // regular source row not emitted
  // Toon/ramp diagnostics: color texgens and the channel they read.
  unsigned long long texgen_color_lit = 0;      // Color0/1 texgen, channel lit
  unsigned long long texgen_color_unlit = 0;    // Color0/1 texgen, channel unlit
  unsigned long long lit_light_missing = 0;     // lit channel names a light never loaded
  unsigned long long texgen_source_binormal = 0; // regular T/B source not emitted
  unsigned long long texgen_source_tex47 = 0;   // regular Tex4..Tex7 not captured
  unsigned long long texgen_source_unknown = 0; // regular source row > Tex7
  unsigned long long per_vertex_tex_mtx = 0;    // TEXMTXIDX attrs (stubbed)
  unsigned long long unresolved_tex_matrix = 0; // matrix rows never written
  unsigned long long normals_ignored = 0;       // decoded past, not lit (S15)
  unsigned long long lighting_ignored = 0;      // chanctrl wants lighting (S15)
  unsigned long long tlut_texture = 0;          // C4/C8/C14X2 binds (S13)
  unsigned long long alpha_compare_ignored = 0; // BP 0xF3 non-always, no TEV
  unsigned long long tev_stages_over = 0;        // numtevstages > kMaxTevStages
  unsigned long long tev_multi_texmap = 0;       // stage reads texmap != 0
  unsigned long long texcoord_scale_active = 0;  // textured TEV uses BP SU scale
  unsigned long long texcoord_scale_mismatch = 0; // SU scale != sampled image
  unsigned long long efb_copy_ignored = 0;      // BP 0x52 copies, no observer
  unsigned long long efb_copies = 0;            // EFB copies performed (S16)
  unsigned long long efb_copy_depth = 0;        // of which Z-source (PE Z24)
  unsigned long long efb_display_copies = 0;    // GXCopyDisp (clear-only, 0xF)
  unsigned long long fog_ignored = 0;           // fog enabled on a non-TEV draw
  unsigned long long indirect_active = 0;       // plans using an indirect stage
  unsigned long long indirect_ignored = 0;      // invalid/undefined indirect state
  unsigned long long logic_op_ignored = 0;      // cmode0 logic-op enable
  unsigned long long dst_alpha_active = 0;      // BP 0x42 forced EFB alpha
  unsigned long long early_depth_active = 0;    // GXSetZCompLoc before alpha test
  unsigned long long ztexture_active = 0;       // late BP F4/F5 depth output
  unsigned long long ztexture_ignored = 0;      // unsupported/non-writing form
};

// Cross-draw cached vertex attributes (Dolphin VertexLoaderManager::
// normal_cache / tangent_cache / binormal_cache): the RAW object-space N/B/T of
// the LAST vertex of the most recent draw that decoded that attribute. A later
// draw whose vertex format omits the attribute reuses this value (Dolphin's
// I_CACHED_NORMAL fallback). Persists across draws in stream order, so it lives
// in the sink (not a per-draw state snapshot). Seeded to zero, matching
// Dolphin's zero-initialized caches before any normal is decoded.
struct CachedVertexAttrs {
  float normal[3]{0.f, 0.f, 0.f};
  float tangent[3]{0.f, 0.f, 0.f};
  float binormal[3]{0.f, 0.f, 0.f};
};

// Register-state model rebuilt from RenderStatePacket stream. Only the
// registers the slice consumes are decoded; everything arrives raw so later
// modules extend decode without frontend changes.
class GxCoreState {
public:
  void reset();
  void apply(const gxruntime::aurora_recomp::RenderStatePacket& state);

  // Build the plan for one span-complete draw. `counters` collects gap
  // signals; the plan is self-contained (vertices/indices/uniforms copied).
  // `cached` (optional) carries the cross-draw N/B/T fallback: it is READ to
  // fill the uniform when this draw's format omits an attribute, and UPDATED
  // to this draw's last-vertex value when the format carries it (Dolphin
  // VertexLoaderManager normal_cache write on m_remaining==0). nullptr in unit
  // tests that don't exercise the fallback (fields stay zero, matching a fresh
  // cache).
  DrawPlan build_draw_plan(const gxruntime::aurora_recomp::ConsumedDraw& draw,
                           GapCounters& counters,
                           CachedVertexAttrs* cached = nullptr) const;
  // As build_draw_plan, into a caller-owned plan whose vector capacity is kept
  // across draws (no per-draw allocation of the vertex and index arrays).
  void build_draw_plan_into(const gxruntime::aurora_recomp::ConsumedDraw& draw,
                            GapCounters& counters, CachedVertexAttrs* cached,
                            DrawPlan& plan) const;

  std::uint32_t bp(std::uint8_t reg) const { return bp_regs_[reg]; }
  bool bp_valid(std::uint8_t reg) const { return bp_valid_[reg]; }
  // The draw tag goes with one draw (DrawPlan::draw_tag); a scope starts at
  // one draw and covers the count of draws it gives (GxCoreSink).
  void forget_draw_tag() {
    bp_valid_[kDrawTagRegister] = bp_valid_[kDrawTagAgeRegister] = false;
    bp_valid_[kDrawScopeRegister] = bp_valid_[kDrawScopeCountRegister] = false;
  }
  static constexpr std::uint8_t kDrawTagRegister = 0x7Eu;
  static constexpr std::uint8_t kDrawTagAgeRegister = 0x7Du;
  static constexpr std::uint8_t kDrawScopeRegister = 0x7Cu;
  static constexpr std::uint8_t kDrawScopeCountRegister = 0x7Bu;
  // A register state put back from a save state takes a version no draw has
  // been derived from in this run (the saved one belongs to the run that saved
  // it).
  void renew_version() { version_ = next_version(); }

private:
  std::uint32_t bp_regs_[256]{};
  bool bp_valid_[256]{};
  // TEV color registers (BP 0xE0-0xE7). Konst and tev-color writes alias the
  // same BP address, disambiguated only by the TevRegType bit; tracking them
  // separately at write time (not from the last bp_regs_ snapshot) keeps both.
  std::int32_t tev_color_[4][4]{};  // I_COLORS: [0] prev seed, [1..3] c0/c1/c2
  std::int32_t konst_color_[4][4]{}; // I_KCOLORS: K0-K3
  std::uint32_t vcd_lo_ = 0;
  std::uint32_t vcd_hi_ = 0;
  bool vcd_lo_valid_ = false;
  bool vcd_hi_valid_ = false;
  std::uint32_t vat_[8][3]{};
  std::uint8_t vat_valid_[8]{}; // bit per group
  // Changes with every change apply() makes to the BP, VCD and VAT registers
  // (next_version), except the host's draw tag registers above, which no
  // derived state reads; build_draw_plan_into's cache of the state derived
  // from them compares it instead of the registers.
  std::uint64_t version_ = 0;
  static std::uint64_t next_version();
};

// AuroraRenderSink that owns a ConsumingAuroraRenderSink (streaming mode) and
// pairs every span-complete draw with the register state that was current at
// its Draw packet. The observer receives (plan-ready) ConsumedDraws; the
// register snapshot is taken when the Draw packet passes through — state
// packets that arrive later belong to the NEXT draw.
class GxCoreSink final : public gxruntime::aurora_recomp::AuroraRenderSink {
public:
  using PlanObserver = void (*)(const DrawPlan& plan, void* user);
  // Fires when a CopyDestination packet arrives, AFTER the pending draw is
  // flushed (its geometry is in the pass) — so a GPU observer can resolve the
  // EFB into a texture at the copy's stream position.
  using CopyObserver = void (*)(const EfbCopyCommand& cmd, void* user);

  GxCoreSink();

  void set_guest_resolver(const DolGuestAddressResolver* resolver) {
    consumer_.set_guest_resolver(resolver);
  }
  void set_plan_observer(PlanObserver observer, void* user) {
    plan_observer_ = observer;
    plan_observer_user_ = user;
  }
  void set_copy_observer(CopyObserver observer, void* user) {
    copy_observer_ = observer;
    copy_observer_user_ = user;
  }

  bool submit_packet(
      const gxruntime::aurora_recomp::RenderPacket& packet) override;
  // Frame boundary: plan the final pending draw of the frame.
  void flush_frame();

  // Save states: the register model the next draws are planned against and
  // the cross-draw attribute cache. A state is taken with the front end
  // drained; a draw still pending in the consumer is not carried (load plans
  // it first, then starts the next draw from the saved registers).
  std::vector<std::uint8_t> save_state() const;
  bool load_state(const std::uint8_t* data, std::size_t size);

  const GapCounters& counters() const { return counters_; }
  GapCounters& counters() { return counters_; }
  const gxruntime::aurora_recomp::ConsumingAuroraRenderSink& consumer() const {
    return consumer_;
  }
  const char* failure_reason() const { return consumer_.failure_reason(); }

private:
  static void on_consumed_draw(
      const gxruntime::aurora_recomp::ConsumedDraw& draw,
      unsigned long long cumulative_draw, void* user);

  gxruntime::aurora_recomp::ConsumingAuroraRenderSink consumer_;
  GxCoreState live_state_;    // updated by every state packet
  GxCoreState pending_state_; // snapshot paired with the pending draw
  // State packets applied to live_state_ since the last draw. At a draw they
  // are replayed onto pending_state_, which then equals live_state_ exactly
  // (apply depends only on the state and the packet) without copying the
  // whole state per draw. Past kMaxReplay the next draw copies instead.
  static constexpr std::size_t kMaxReplay = 256u;
  std::vector<gxruntime::aurora_recomp::RenderStatePacket> since_draw_;
  bool replay_overflow_ = false;
  CachedVertexAttrs cached_attrs_{}; // cross-draw N/B/T fallback (stream order)
  DrawPlan scratch_plan_{};          // reused by on_consumed_draw
  // The draw scope under way: its emitter, the draws left and the next's place,
  // and whether its draws index their positions (cloth) rather than send them
  // (a wake's).
  std::uint32_t scope_ = 0;
  std::uint32_t scope_left_ = 0;
  std::uint32_t scope_part_ = 0;
  bool scope_indexed_ = false;
  PlanObserver plan_observer_ = nullptr;
  void* plan_observer_user_ = nullptr;
  CopyObserver copy_observer_ = nullptr;
  void* copy_observer_user_ = nullptr;
  GapCounters counters_{};
};

} // namespace gxruntime::gxcore
