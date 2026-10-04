// SPDX-License-Identifier: GPL-3.0-or-later
#include "gxruntime/gxcore/gxcore.hpp"
#include "gxruntime/gxcore/shader.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>

namespace gxruntime::gxcore {

namespace {

namespace ar = gxruntime::aurora_recomp;

std::uint32_t bits(std::uint32_t value, std::uint32_t count,
                   std::uint32_t shift) {
  return (value >> shift) & ((1u << count) - 1u);
}

std::int32_t sx11(std::uint32_t value) {
  return static_cast<std::int32_t>(value << 21) >> 21;
}

// FogParam0/FogParam3 FloatValue (Dolphin BPMemory.cpp): mant(11)/exp(8)/sign(1)
// re-expanded to an IEEE-754 f32 (mantissa scaled from 11 to 23 bits).
float fog_param_float(std::uint32_t raw) {
  const std::uint32_t integral =
      (bits(raw, 1, 19) << 31) | (bits(raw, 8, 11) << 23) | (bits(raw, 11, 0) << 12);
  float f;
  std::memcpy(&f, &integral, sizeof f);
  return f;
}

void decode_scissor(DrawPlan& plan, std::uint32_t tl, std::uint32_t br,
                    std::uint32_t offset) {
  // Dolphin BPFunctions::ScissorResult: coordinates are inclusive, offsets
  // wrap every 1024 pixels, and hardware chooses the rectangle with the most
  // viewport overlap (then the largest area).
  const int left = static_cast<int>(bits(tl, 11, 12));
  const int top = static_cast<int>(bits(tl, 11, 0));
  const int right = static_cast<int>(bits(br, 11, 12));
  const int bottom = static_cast<int>(bits(br, 11, 0));
  const int x_offset = static_cast<int>(bits(offset, 9, 0)) << 1;
  const int y_offset = static_cast<int>(bits(offset, 9, 10)) << 1;
  plan.scissor_valid = true;
  if (left > right || top > bottom) {
    return;
  }

  struct Range {
    int offset;
    int start;
    int end;
  };
  auto ranges = [](int start, int end, int base_offset, int dimension) {
    std::array<Range, 9> result{};
    std::size_t count = 0;
    for (int extra = -4096; extra <= 4096; extra += 1024) {
      const int wrapped_offset = base_offset + extra;
      const int clipped_start =
          std::clamp(start - wrapped_offset, 0, dimension);
      const int clipped_end =
          std::clamp(end - wrapped_offset + 1, 0, dimension);
      if (clipped_start < clipped_end)
        result[count++] = {wrapped_offset, clipped_start, clipped_end};
    }
    return std::pair{result, count};
  };

  const auto [x_ranges, x_count] = ranges(left, right, x_offset, 640);
  const auto [y_ranges, y_count] = ranges(top, bottom, y_offset, 528);
  if (x_count == 0 || y_count == 0)
    return;

  int viewport_left = 0;
  int viewport_right = 640;
  int viewport_top = 0;
  int viewport_bottom = 528;
  if (plan.viewport_valid) {
    const float x0 = plan.viewport[3] - plan.viewport[0];
    const float x1 = plan.viewport[3] + plan.viewport[0];
    const float y0 = plan.viewport[4] - plan.viewport[1];
    const float y1 = plan.viewport[4] + plan.viewport[1];
    viewport_left = static_cast<int>(std::min(x0, x1));
    viewport_right = static_cast<int>(std::max(x0, x1));
    viewport_top = static_cast<int>(std::min(y0, y1));
    viewport_bottom = static_cast<int>(std::max(y0, y1));
  }

  int best_viewport_area = -1;
  int best_area = -1;
  for (std::size_t xi = 0; xi < x_count; ++xi) {
    for (std::size_t yi = 0; yi < y_count; ++yi) {
      const Range& xr = x_ranges[xi];
      const Range& yr = y_ranges[yi];
      const int vx0 =
          std::clamp(xr.start + xr.offset, viewport_left, viewport_right);
      const int vx1 =
          std::clamp(xr.end + xr.offset, viewport_left, viewport_right);
      const int vy0 =
          std::clamp(yr.start + yr.offset, viewport_top, viewport_bottom);
      const int vy1 =
          std::clamp(yr.end + yr.offset, viewport_top, viewport_bottom);
      const int viewport_area = (vx1 - vx0) * (vy1 - vy0);
      const int area = (xr.end - xr.start) * (yr.end - yr.start);
      if (viewport_area < best_viewport_area ||
          (viewport_area == best_viewport_area && area <= best_area)) {
        continue;
      }
      best_viewport_area = viewport_area;
      best_area = area;
      plan.scissor_x = xr.start;
      plan.scissor_y = yr.start;
      plan.scissor_width = xr.end - xr.start;
      plan.scissor_height = yr.end - yr.start;
    }
  }
}

} // namespace

// --- Register state ----------------------------------------------------------

void GxCoreState::reset() { *this = GxCoreState{}; }

// A version no other state, and no earlier change of this one, has: the
// thread's own count, tagged with the thread, so no atomic operation is needed
// and none is ever reused. A copy of a state carries its version with its
// registers; a reset state, like a new one, is version 0 with every register
// at its default.
std::uint64_t GxCoreState::next_version() {
  static std::atomic<std::uint32_t> s_threads{0};
  static thread_local const std::uint64_t s_tag =
      static_cast<std::uint64_t>(s_threads.fetch_add(1, std::memory_order_relaxed) + 1u) << 40;
  static thread_local std::uint64_t s_count = 0;
  return s_tag | ++s_count;
}

void GxCoreState::apply(const ar::RenderStatePacket& state) {
  using ar::RenderStateKind;
  switch (state.kind) {
  case RenderStateKind::BpReg:
    if (state.index < 256u) {
      // The draw tag registers change at nearly every particle draw and feed
      // no derived state, so they leave the version alone.
      const bool tag = state.index >= kDrawScopeCountRegister && state.index <= kDrawTagRegister;
      if (!tag && (!bp_valid_[state.index] || bp_regs_[state.index] != state.value))
        version_ = next_version();
      bp_regs_[state.index] = state.value;
      bp_valid_[state.index] = true;
      // TEV color/konst registers (BP 0xE0-0xE7, TevReg RA/BG). The TevRegType
      // bit (23) selects konst vs tev-color; the two alias the same address, so
      // route each half to its own store to keep both (BPMemory.h TevReg).
      if (state.index >= 0xE0u && state.index <= 0xE7u) {
        const std::uint32_t idx = (state.index - 0xE0u) / 2u;
        const bool is_bg = ((state.index - 0xE0u) & 1u) != 0u;
        auto sx11 = [](std::uint32_t v) {
          return static_cast<std::int32_t>(v << 21) >> 21; // sign-extend 11 bits
        };
        std::int32_t(&dst)[4] =
            bits(state.value, 1, 23) != 0u ? konst_color_[idx] : tev_color_[idx];
        if (!is_bg) {
          dst[0] = sx11(bits(state.value, 11, 0));  // red
          dst[3] = sx11(bits(state.value, 11, 12)); // alpha
        } else {
          dst[2] = sx11(bits(state.value, 11, 0));  // blue
          dst[1] = sx11(bits(state.value, 11, 12)); // green
        }
      }
    }
    break;
  case RenderStateKind::CpVcd:
    if (state.index == 0u) {
      if (!vcd_lo_valid_ || vcd_lo_ != state.value)
        version_ = next_version();
      vcd_lo_ = state.value;
      vcd_lo_valid_ = true;
    } else {
      if (!vcd_hi_valid_ || vcd_hi_ != state.value)
        version_ = next_version();
      vcd_hi_ = state.value;
      vcd_hi_valid_ = true;
    }
    break;
  case RenderStateKind::CpVat:
    if (state.index < 8u && state.aux0 < 3u) {
      if ((vat_valid_[state.index] & (1u << state.aux0)) == 0u || vat_[state.index][state.aux0] != state.value)
        version_ = next_version();
      vat_[state.index][state.aux0] = state.value;
      vat_valid_[state.index] |= static_cast<std::uint8_t>(1u << state.aux0);
    }
    break;
  default:
    break; // arrays/cull/etc. are handled by the inner consumer
  }
}

// --- Vertex decode (Dolphin VertexLoader semantics) ---------------------------

namespace {

// One payload element in GC attribute order. attr uses CP array numbering
// (0 pos, 1 nrm, 2/3 colors, 4-11 tex); matrix-index bytes use kMatIdx.
struct WalkEntry {
  enum Kind : std::uint8_t {
    kPosMtxIdx,
    kTexMtxIdx,
    kPos,
    kNormal,
    kColor,
    kTex,
  };
  Kind kind = kPos;
  std::uint8_t attr = 0;      // CP array index for indexed fetch
  std::uint8_t vcd_type = 0;  // 1 direct, 2 idx8, 3 idx16
  std::uint8_t format = 0;    // ComponentFormat / ColorFormat
  std::uint8_t count = 0;     // components (pos 2/3, tex 1/2, normal 3/9)
  std::uint8_t frac = 0;
  std::uint8_t out_slot = 0;  // color: 0/1; tex: 0-7
  std::uint32_t element_size = 0; // bytes of one element in the source array
};

bool component_scalar_size(std::uint32_t format, std::uint32_t* out) {
  switch (format) {
  case 0u: // u8
  case 1u: // s8
    *out = 1u;
    return true;
  case 2u: // u16
  case 3u: // s16
    *out = 2u;
    return true;
  case 4u: // f32
    *out = 4u;
    return true;
  default:
    return false;
  }
}

bool color_element_size(std::uint32_t format, std::uint32_t* out) {
  switch (format) {
  case 0u: // RGB565
  case 3u: // RGBA4444
    *out = 2u;
    return true;
  case 1u: // RGB888
  case 4u: // RGBA6666
    *out = 3u;
    return true;
  case 2u: // RGB888x
  case 5u: // RGBA8888
    *out = 4u;
    return true;
  default:
    return false;
  }
}

struct WalkLayout {
  WalkEntry entries[24];
  std::uint32_t entry_count = 0;
  std::uint32_t vertex_size = 0;
  bool has_pos_mtx_idx = false;
  bool has_tex_mtx_idx = false;
  bool has_normal = false;
  bool has_nbt = false; // normal attr carries binormal+tangent (emboss inputs)
  bool has_color[2] = {false, false};
  std::uint8_t uv_mask = 0;          // tex0-7 presence
  std::uint8_t tex_mtx_idx_mask = 0; // per-vertex TEXMTXIDX per texgen (0..4)
};

// Derive the payload walk from raw VCD/VAT (CPMemory.h bit positions).
bool derive_walk(std::uint32_t vcd_lo, std::uint32_t vcd_hi,
                 const std::uint32_t vat[3], WalkLayout& out) {
  out = WalkLayout{};
  std::uint32_t offset = 0;
  auto add = [&](const WalkEntry& entry, std::uint32_t payload_size) {
    out.entries[out.entry_count++] = entry;
    offset += payload_size;
  };

  if (bits(vcd_lo, 1, 0) != 0u) { // PosMatIdx
    out.has_pos_mtx_idx = true;
    add({.kind = WalkEntry::kPosMtxIdx, .vcd_type = 1}, 1u);
  }
  for (std::uint32_t t = 0; t < 8; ++t) { // TexMatIdx0-7
    if (bits(vcd_lo, 1, 1 + t) != 0u) {
      out.has_tex_mtx_idx = true;
      if (t < kMaxTexGens)
        out.tex_mtx_idx_mask |= static_cast<std::uint8_t>(1u << t);
      add({.kind = WalkEntry::kTexMtxIdx,
           .vcd_type = 1,
           .out_slot = static_cast<std::uint8_t>(t)},
          1u);
    }
  }

  auto index_size = [](std::uint32_t type) -> std::uint32_t {
    return type == 2u ? 1u : 2u;
  };

  // Position (VCD bits 9-10; VAT g0 bits 0-8).
  {
    const std::uint32_t type = bits(vcd_lo, 2, 9);
    if (type == 0u)
      return false; // a draw without position is malformed
    const std::uint32_t elements = bits(vat[0], 1, 0) != 0u ? 3u : 2u;
    const std::uint32_t format = bits(vat[0], 3, 1);
    const std::uint32_t frac = bits(vat[0], 5, 4);
    std::uint32_t scalar = 0;
    if (!component_scalar_size(format, &scalar))
      return false;
    WalkEntry entry{.kind = WalkEntry::kPos,
                    .attr = 0,
                    .vcd_type = static_cast<std::uint8_t>(type),
                    .format = static_cast<std::uint8_t>(format),
                    .count = static_cast<std::uint8_t>(elements),
                    .frac = static_cast<std::uint8_t>(frac),
                    .element_size = elements * scalar};
    add(entry, type == 1u ? elements * scalar : index_size(type));
  }

  // Normal (VCD bits 11-12; VAT g0 bits 9-12). Decoded for stride only.
  {
    const std::uint32_t type = bits(vcd_lo, 2, 11);
    if (type != 0u) {
      out.has_normal = true;
      const bool nbt = bits(vat[0], 1, 9) != 0u;
      out.has_nbt = nbt;
      const std::uint32_t format = bits(vat[0], 3, 10);
      const bool index3 = bits(vat[0], 1, 31) != 0u;
      std::uint32_t scalar = 0;
      if (!component_scalar_size(format, &scalar))
        return false;
      const std::uint32_t elements = nbt ? 9u : 3u;
      // Normals use a FIXED fractional scale (not the VAT frac field): s8 => 6,
      // s16 => 14, f32 => 0 (GX/Dolphin VertexLoader normal decode).
      const std::uint32_t nfrac = format == 1u ? 6u : format == 3u ? 14u : 0u;
      WalkEntry entry{.kind = WalkEntry::kNormal,
                      .attr = 1,
                      .vcd_type = static_cast<std::uint8_t>(type),
                      .format = static_cast<std::uint8_t>(format),
                      .count = static_cast<std::uint8_t>(elements),
                      .frac = static_cast<std::uint8_t>(nfrac),
                      .element_size = nbt && index3 ? 3u * scalar
                                                    : elements * scalar};
      if (type == 1u) {
        add(entry, elements * scalar);
      } else if (nbt && index3) {
        // Three separate indices, one per 3-component normal.
        for (std::uint32_t n = 0; n < 3; ++n) {
          WalkEntry part = entry;
          part.count = 3;
          add(part, index_size(type));
        }
      } else {
        add(entry, index_size(type));
      }
    }
  }

  // Colors (VCD bits 13-14 / 15-16; VAT g0 bits 13-16 / 17-20).
  for (std::uint32_t c = 0; c < 2; ++c) {
    const std::uint32_t type = bits(vcd_lo, 2, 13 + 2 * c);
    if (type == 0u)
      continue;
    out.has_color[c] = true;
    const std::uint32_t format = bits(vat[0], 3, 14 + 4 * c);
    std::uint32_t element = 0;
    if (!color_element_size(format, &element))
      return false;
    WalkEntry entry{.kind = WalkEntry::kColor,
                    .attr = static_cast<std::uint8_t>(2 + c),
                    .vcd_type = static_cast<std::uint8_t>(type),
                    .format = static_cast<std::uint8_t>(format),
                    .out_slot = static_cast<std::uint8_t>(c),
                    .element_size = element};
    add(entry, type == 1u ? element : index_size(type));
  }

  // Tex coords 0-7 (VCD hi 2 bits each; VAT g0/g1/g2 per CPMemory.h).
  for (std::uint32_t t = 0; t < 8; ++t) {
    const std::uint32_t type = bits(vcd_hi, 2, 2 * t);
    if (type == 0u)
      continue;
    std::uint32_t elements_bit = 0, format = 0, frac = 0;
    switch (t) {
    case 0:
      elements_bit = bits(vat[0], 1, 21);
      format = bits(vat[0], 3, 22);
      frac = bits(vat[0], 5, 25);
      break;
    case 1:
      elements_bit = bits(vat[1], 1, 0);
      format = bits(vat[1], 3, 1);
      frac = bits(vat[1], 5, 4);
      break;
    case 2:
      elements_bit = bits(vat[1], 1, 9);
      format = bits(vat[1], 3, 10);
      frac = bits(vat[1], 5, 13);
      break;
    case 3:
      elements_bit = bits(vat[1], 1, 18);
      format = bits(vat[1], 3, 19);
      frac = bits(vat[1], 5, 22);
      break;
    case 4:
      elements_bit = bits(vat[1], 1, 27);
      format = bits(vat[1], 3, 28);
      frac = bits(vat[2], 5, 0);
      break;
    case 5:
      elements_bit = bits(vat[2], 1, 5);
      format = bits(vat[2], 3, 6);
      frac = bits(vat[2], 5, 9);
      break;
    case 6:
      elements_bit = bits(vat[2], 1, 14);
      format = bits(vat[2], 3, 15);
      frac = bits(vat[2], 5, 18);
      break;
    case 7:
      elements_bit = bits(vat[2], 1, 23);
      format = bits(vat[2], 3, 24);
      frac = bits(vat[2], 5, 27);
      break;
    }
    const std::uint32_t elements = elements_bit != 0u ? 2u : 1u;
    std::uint32_t scalar = 0;
    if (!component_scalar_size(format, &scalar))
      return false;
    out.uv_mask |= static_cast<std::uint8_t>(1u << t);
    WalkEntry entry{.kind = WalkEntry::kTex,
                    .attr = static_cast<std::uint8_t>(4 + t),
                    .vcd_type = static_cast<std::uint8_t>(type),
                    .format = static_cast<std::uint8_t>(format),
                    .count = static_cast<std::uint8_t>(elements),
                    .frac = static_cast<std::uint8_t>(frac),
                    .out_slot = static_cast<std::uint8_t>(t),
                    .element_size = elements * scalar};
    add(entry, type == 1u ? elements * scalar : index_size(type));
  }

  out.vertex_size = offset;
  return out.entry_count > 0;
}

std::uint32_t read_be(const std::uint8_t* p, std::uint32_t size) {
  std::uint32_t v = 0;
  for (std::uint32_t i = 0; i < size; ++i)
    v = (v << 8) | p[i];
  return v;
}

std::uint16_t load_be16(const std::uint8_t* p) {
  std::uint16_t v;
  std::memcpy(&v, p, sizeof v);
  return __builtin_bswap16(v);
}

std::uint32_t load_be32(const std::uint8_t* p) {
  std::uint32_t v;
  std::memcpy(&v, p, sizeof v);
  return __builtin_bswap32(v);
}

// One component of an attribute: fixed-point formats times the entry's
// 1 / 2^frac, which is exactly dividing by 2^frac as this did (both are the
// one rounding of the same real number, and an 8- or 16-bit value over 2^31
// at most is far from the subnormals).
inline float decode_scaled(const std::uint8_t* p, std::uint32_t format,
                           float scale) {
  switch (format) {
  case 0u: // u8
    return static_cast<float>(p[0]) * scale;
  case 1u: // s8
    return static_cast<float>(static_cast<std::int8_t>(p[0])) * scale;
  case 2u: // u16
    return static_cast<float>(load_be16(p)) * scale;
  case 3u: // s16
    return static_cast<float>(static_cast<std::int16_t>(load_be16(p))) *
           scale;
  case 4u: { // f32 big-endian
    const std::uint32_t v = load_be32(p);
    float f;
    std::memcpy(&f, &v, sizeof f);
    return f;
  }
  default:
    return 0.f;
  }
}

std::uint8_t expand5(std::uint32_t v) {
  return static_cast<std::uint8_t>((v << 3) | (v >> 2));
}
std::uint8_t expand6(std::uint32_t v) {
  return static_cast<std::uint8_t>((v << 2) | (v >> 4));
}
std::uint8_t expand4(std::uint32_t v) {
  return static_cast<std::uint8_t>(v * 17u);
}

void decode_color(const std::uint8_t* p, std::uint32_t format, float out[4]) {
  std::uint8_t r = 255, g = 255, b = 255, a = 255;
  switch (format) {
  case 0u: { // RGB565
    const std::uint32_t v = read_be(p, 2);
    r = expand5(bits(v, 5, 11));
    g = expand6(bits(v, 6, 5));
    b = expand5(bits(v, 5, 0));
    break;
  }
  case 1u: // RGB888
    r = p[0];
    g = p[1];
    b = p[2];
    break;
  case 2u: // RGB888x
    r = p[0];
    g = p[1];
    b = p[2];
    break;
  case 3u: { // RGBA4444
    const std::uint32_t v = read_be(p, 2);
    r = expand4(bits(v, 4, 12));
    g = expand4(bits(v, 4, 8));
    b = expand4(bits(v, 4, 4));
    a = expand4(bits(v, 4, 0));
    break;
  }
  case 4u: { // RGBA6666
    const std::uint32_t v = read_be(p, 3);
    r = expand6(bits(v, 6, 18));
    g = expand6(bits(v, 6, 12));
    b = expand6(bits(v, 6, 6));
    a = expand6(bits(v, 6, 0));
    break;
  }
  case 5u: // RGBA8888
    r = p[0];
    g = p[1];
    b = p[2];
    a = p[3];
    break;
  }
  // The 256 quotients x / 255 once, rather than four divisions a colour.
  static const auto unorm8 = [] {
    std::array<float, 256> table{};
    for (std::uint32_t i = 0; i < 256u; ++i)
      table[i] = static_cast<float>(i) / 255.f;
    return table;
  }();
  out[0] = unorm8[r];
  out[1] = unorm8[g];
  out[2] = unorm8[b];
  out[3] = unorm8[a];
}

const ar::ConsumedArrayInput* find_array(const ar::ConsumedDraw& draw,
                                         std::uint32_t attr) {
  for (std::uint32_t i = 0; i < draw.array_input_count; ++i) {
    if (draw.arrays[i].attr == attr)
      return &draw.arrays[i];
  }
  return nullptr;
}

// Matrix-memory row -> 4 floats, from the draw's captured XF matrix state.
// Rows 0-29 live in position_matrices (3 rows per matrix), rows 30-62 in
// tex_matrices. Returns false when any word of the row was never written.
bool load_matrix_row(const ar::ConsumedDraw& draw, std::uint32_t row,
                     float out[4]) {
  if (row < 30u) {
    const std::uint32_t matrix = row / 3u;
    const std::uint32_t base = (row % 3u) * 4u;
    const std::uint16_t mask = draw.position_matrix_valid_mask;
    for (std::uint32_t w = 0; w < 4; ++w)
      out[w] = draw.position_matrices[matrix][base + w];
    return (mask & (1u << matrix)) != 0u;
  }
  if (row < 63u) {
    const std::uint32_t rel = row - 30u;
    const std::uint32_t matrix = rel / 3u;
    const std::uint32_t base = (rel % 3u) * 4u;
    bool valid = true;
    for (std::uint32_t w = 0; w < 4; ++w) {
      out[w] = draw.tex_matrices[matrix][base + w];
      if ((draw.tex_matrix_word_mask[matrix] & (1u << (base + w))) == 0u)
        valid = false;
    }
    return valid;
  }
  out[0] = out[1] = out[2] = out[3] = 0.f;
  return false;
}

void identity_rows(float rows[3][4]) {
  std::memset(rows, 0, sizeof(float) * 12);
  rows[0][0] = 1.f;
  rows[1][1] = 1.f;
  rows[2][2] = 1.f;
}

} // namespace

DrawPlan GxCoreState::build_draw_plan(const ar::ConsumedDraw& draw,
                                      GapCounters& counters,
                                      CachedVertexAttrs* cached) const {
  DrawPlan plan;
  build_draw_plan_into(draw, counters, cached, plan);
  return plan;
}

// The same plan, written into a caller-owned object. The sink builds one plan
// per draw (about 20,000 a frame on Wind Waker's title), and returning a fresh
// DrawPlan allocated and freed its vertex and index arrays every time; reusing
// one object keeps their capacity. Every other field is reset to its default.
void GxCoreState::build_draw_plan_into(const ar::ConsumedDraw& draw,
                                       GapCounters& counters,
                                       CachedVertexAttrs* cached,
                                       DrawPlan& plan) const {
  {
    // Every field to its default, from one copy made once; the vertex and
    // index arrays keep their capacity, and the vertex constants stay as they
    // are until the draw's inputs are known (see "Uniforms" below).
    static const DrawPlanFields kDefaults{};
    static_cast<DrawPlanFields&>(plan) = kDefaults;
    plan.vertices.clear();
    plan.indices.clear();
  }
  plan.match_payload = draw.vertex_payload.data();
  plan.match_payload_size = static_cast<std::uint32_t>(draw.vertex_payload.size());
  plan.match_primitive = draw.primitive;
  plan.match_vtx_fmt = draw.vtx_fmt;
  plan.match_vertex_stride = draw.vertex_size;
  if (bp_valid_[kDrawTagRegister]) {
    plan.draw_tag = bp_regs_[kDrawTagRegister];
    plan.draw_tag_age = bp_valid_[kDrawTagAgeRegister] ? bp_regs_[kDrawTagAgeRegister] : 0u;
  }
  auto skip = [&](const char* reason) {
    plan.ok = false;
    plan.skip_reason = reason;
    ++counters.draws_skipped;
  };
  auto noop = [&](const char* reason) {
    plan.ok = false;
    plan.skip_reason = reason;
    ++counters.draws_noop;
  };

  if (draw.cull_all) {
    ++counters.cull_all_draws;
    return skip("cull-all state");
  }
  if (!vcd_lo_valid_ || !vcd_hi_valid_ || draw.vtx_fmt >= 8u ||
      (vat_valid_[draw.vtx_fmt] & 0x7u) != 0x7u) {
    ++counters.missing_vcd;
    return skip("VCD/VAT not yet seen");
  }
  if ((draw.transform_flags &
       ar::kDrawTransformProjectionValid) == 0u) {
    ++counters.vertex_decode_failures;
    ++counters.vertex_projection_missing;
    return skip("projection never captured");
  }
  if (draw.vertex_payload.empty()) {
    ++counters.vertex_decode_failures;
    ++counters.vertex_payload_empty;
    return skip("draw carried no payload");
  }

  plan.match_direct_position = bits(vcd_lo_, 2, 9) == 1u; // VCD position: direct
  for (std::uint32_t uv = 0; uv < kMaxTexGens; ++uv)
    if (bits(vcd_hi_, 2, uv * 2) == 1u)
      plan.match_direct_texcoord_mask |= 1u << uv;
  if (!plan.match_direct_position) {
    if (const auto* positions = find_array(draw, 0)) {
      plan.match_position_offset = positions->vertex_offset;
      plan.match_position_size = positions->index_size;
      plan.match_position_base = positions->base;
      if (positions->resolved && positions->host_data != nullptr) {
        plan.match_position_span = std::min(positions->span_size, positions->host_available);
        plan.match_position_readable = positions->host_available;
        plan.match_position_data = static_cast<const std::uint8_t*>(positions->host_data);
      }
    }
  }

  // BP-derived pipeline state.
  const std::uint32_t gen_mode = bp_valid_[0x00] ? bp_regs_[0x00] : 0u;
  const std::uint32_t cull = bits(gen_mode, 2, 14);
  if (cull == static_cast<std::uint32_t>(CullMode::All)) {
    ++counters.cull_all_draws;
    return skip("genMode culls all");
  }
  // Alpha test is consumed by the TEV path (below); only count it as ignored
  // when no combiner was captured and we fall back to the passthrough fragment.
  if (!bp_valid_[0xC0] && bp_valid_[0xF3]) {
    const std::uint32_t ac = bp_regs_[0xF3];
    if (bits(ac, 3, 16) != 7u || bits(ac, 3, 19) != 7u)
      ++counters.alpha_compare_ignored;
  }

  // Payload walk from raw VCD/VAT.
  struct WalkCacheEntry {
    std::array<std::uint32_t, 5> key{};
    WalkLayout walk{};
    bool valid = false;
  };
  static thread_local std::array<WalkCacheEntry, 16> walk_cache{};
  const std::array walk_key{vcd_lo_, vcd_hi_, vat_[draw.vtx_fmt][0],
                            vat_[draw.vtx_fmt][1], vat_[draw.vtx_fmt][2]};
  auto &entry = walk_cache[(vcd_lo_ ^ vcd_hi_ ^ vat_[draw.vtx_fmt][0] ^
                            vat_[draw.vtx_fmt][1] ^ vat_[draw.vtx_fmt][2]) %
                           walk_cache.size()];
  if (!entry.valid || entry.key != walk_key) {
    entry.valid = derive_walk(vcd_lo_, vcd_hi_, vat_[draw.vtx_fmt], entry.walk);
    entry.key = walk_key;
  }
  const WalkLayout &walk = entry.walk;
  if (!entry.valid) {
    ++counters.vertex_decode_failures;
    ++counters.vertex_walk_underivable;
    return skip("VCD/VAT walk underivable");
  }
  if (walk.vertex_size != draw.vertex_size) {
    // Layout disagreement with the frontend is a correctness bug, not data.
    ++counters.vertex_decode_failures;
    ++counters.vertex_stride_mismatch;
    return skip("walk stride != frontend stride");
  }
  // Cache only register-derived pipeline state. Geometry, mutable guest
  // arrays, uniforms, texture bytes and cross-draw NBT remain live each draw.
  // Thread-local storage keeps separate FIFO consumers independent. The key is
  // the register state's version (next_version: never reused, carried by a
  // copy) with the vertex format it reads, and the draw's own transform,
  // channel, light and texture values - where it was every register, 2 KB
  // copied and compared at each of thousands of draws a frame.
  struct DerivedCache {
    std::uint64_t version = 0;
    std::uint32_t vtx_fmt = 0;
    std::uint32_t xf_regs[DOL_GX_RECOMP_XF_REG_COUNT]{};
    decltype(draw.xf_reg_mask) xf_reg_mask{};
    std::uint32_t chan_regs[DOL_GX_RECOMP_CHAN_REG_COUNT]{};
    decltype(draw.chan_reg_mask) chan_reg_mask{};
    std::uint16_t light_word_mask[DOL_GX_RECOMP_LIGHT_COUNT]{};
    std::array<std::uint32_t, 9 * 6> textures{};
    PipelineKey pipeline{};
    GapCounters delta{};
    bool valid = false;
    unsigned long long hits = 0, misses = 0, verified = 0, mismatches = 0;
    ~DerivedCache() {
      if (std::getenv("DOL_GXCORE_OPT_LOG") || verified != 0)
        std::fprintf(stderr, "[gx-derived] hits=%llu misses=%llu verified=%llu mismatches=%llu\n", hits,
                     misses, verified, mismatches);
    }
  };
  static thread_local DerivedCache derived;
  static const bool cache_enabled = [] {
    const char *value = std::getenv("DOL_GXCORE_DERIVED_CACHE");
    return !value || std::strcmp(value, "0") != 0;
  }();
  static_assert(sizeof(draw.xf_regs) == sizeof(derived.xf_regs) &&
                    sizeof(draw.chan_regs) == sizeof(derived.chan_regs) &&
                    sizeof(draw.light_word_mask) == sizeof(derived.light_word_mask),
                "the derived cache mirrors the draw's register arrays");
  std::array<std::uint32_t, 9 * 6> textures;
  {
    std::uint32_t *out = textures.data();
    auto texture_key = [&](const ar::ConsumedTexture &t) {
      out[0] = t.valid;
      out[1] = t.resolved;
      out[2] = t.slot;
      out[3] = t.format;
      out[4] = t.width;
      out[5] = t.height;
      out += 6;
    };
    texture_key(draw.texture);
    for (const auto &t : draw.textures)
      texture_key(t);
  }
  static constexpr auto counter_members = std::array{
      &GapCounters::dst_alpha_active,
      &GapCounters::early_depth_active,
      &GapCounters::indirect_active,
      &GapCounters::indirect_ignored,
      &GapCounters::lighting_ignored,
      &GapCounters::lit_light_missing,
      &GapCounters::logic_op_ignored,
      &GapCounters::normals_ignored,
      &GapCounters::per_vertex_normal_matrix,
      &GapCounters::tev_multi_texmap,
      &GapCounters::texcoord_scale_active,
      &GapCounters::texcoord_scale_mismatch,
      &GapCounters::texgen_color_lit,
      &GapCounters::texgen_color_unlit,
      &GapCounters::texgen_count_5,
      &GapCounters::texgen_count_6,
      &GapCounters::texgen_count_7,
      &GapCounters::texgen_count_8plus,
      &GapCounters::texgen_count_overflow,
      &GapCounters::texgen_emboss_cached_nbt,
      &GapCounters::texgen_source_binormal,
      &GapCounters::texgen_source_colors,
      &GapCounters::texgen_source_normal,
      &GapCounters::texgen_source_normal_default,
      &GapCounters::texgen_source_tex47,
      &GapCounters::texgen_source_unknown,
      &GapCounters::tlut_texture,
      &GapCounters::unsupported_texgen,
      &GapCounters::ztexture_active,
      &GapCounters::ztexture_ignored,
  };
  // DOL_GXCORE_DERIVED_VERIFY=1: a hit derives the state again and compares
  // it with the cached (a mismatch is reported and the fresh state used).
  static const bool verify = [] {
    const char *value = std::getenv("DOL_GXCORE_DERIVED_VERIFY");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  ShaderKey& key = plan.pipeline.shader;
  PipelineKey &pipe = plan.pipeline;
  const bool hit = cache_enabled && derived.valid && derived.version == version_ &&
      derived.vtx_fmt == draw.vtx_fmt && derived.xf_reg_mask == draw.xf_reg_mask &&
      derived.chan_reg_mask == draw.chan_reg_mask && derived.textures == textures &&
      std::memcmp(derived.xf_regs, draw.xf_regs, sizeof(derived.xf_regs)) == 0 &&
      std::memcmp(derived.chan_regs, draw.chan_regs, sizeof(derived.chan_regs)) == 0 &&
      std::memcmp(derived.light_word_mask, draw.light_word_mask, sizeof(derived.light_word_mask)) == 0;
  if (hit && !verify) {
    plan.pipeline = derived.pipeline;
    for (auto member : counter_members)
      counters.*member += derived.delta.*member;
    ++derived.hits;
  } else {
    GapCounters before = counters;
    // Shader key.
    const std::uint32_t xf_numtexgens_reg = 0x103Fu - 0x1018u;
    std::uint32_t num_tex_gens = 0;
    if (draw.xf_reg_mask & (1ull << xf_numtexgens_reg))
      num_tex_gens = draw.xf_regs[xf_numtexgens_reg] & 0xFu;
    else
      num_tex_gens = bits(gen_mode, 4, 0);
    if (num_tex_gens == 5u)
      ++counters.texgen_count_5;
    else if (num_tex_gens == 6u)
      ++counters.texgen_count_6;
    else if (num_tex_gens == 7u)
      ++counters.texgen_count_7;
    else if (num_tex_gens >= 8u)
      ++counters.texgen_count_8plus;
    if (num_tex_gens > kMaxTexGens) {
      ++counters.unsupported_texgen;
      ++counters.texgen_count_overflow;
      num_tex_gens = kMaxTexGens;
    }
    key.num_tex_gens = static_cast<std::uint8_t>(num_tex_gens);
    key.has_pos_mtx_idx = walk.has_pos_mtx_idx ? 1u : 0u;
    key.has_tex_mtx_idx = walk.has_tex_mtx_idx ? 1u : 0u;
    key.tex_mtx_idx_mask = walk.tex_mtx_idx_mask;
    key.has_color0 = walk.has_color[0] ? 1u : 0u;
    key.has_color1 = walk.has_color[1] ? 1u : 0u;
    key.uv_mask =
        static_cast<std::uint8_t>(walk.uv_mask & ((1u << kMaxTexGens) - 1u));
    // Vertex-format N/B/T presence (GC packs all three in one NBT attribute, so
    // binormal/tangent presence follows has_nbt). A lit/emboss draw that omits
    // one substitutes the cached fallback from the uniform instead of a
    // per-vertex input (Dolphin I_CACHED_NORMAL, populated below from the last
    // decoded vertex of a draw that DID carry the attribute).
    key.has_vertex_normal = walk.has_normal ? 1u : 0u;
    key.has_vertex_binormal = walk.has_nbt ? 1u : 0u;
    key.has_vertex_tangent = walk.has_nbt ? 1u : 0u;

    // Lighting key (S15). LitChannel bitfields (XFMemory.h): matsource@0,
    // enablelighting@1, lightMask0_3@2, ambsource@6, diffusefunc@7, attnfunc@9,
    // lightMask4_7@11. chan_regs slot = xf_addr - 0x1009: [0]=numColorChans,
    // [1..4]=amb0/amb1/mat0/mat1, [5..8]=color0/color1/alpha0/alpha1 ctrl.
    auto lit_chan = [&](std::uint32_t slot, LightChanKey &ch) {
      if ((draw.chan_reg_mask & (1u << slot)) == 0u)
        return;
      const std::uint32_t v = draw.chan_regs[slot];
      ch.matsource = static_cast<std::uint8_t>(bits(v, 1, 0));
      ch.enablelighting = static_cast<std::uint8_t>(bits(v, 1, 1));
      ch.ambsource = static_cast<std::uint8_t>(bits(v, 1, 6));
      ch.diffusefunc = static_cast<std::uint8_t>(bits(v, 2, 7));
      ch.attnfunc = static_cast<std::uint8_t>(bits(v, 2, 9));
      const std::uint32_t mask = bits(v, 4, 2) | (bits(v, 4, 11) << 4);
      ch.light_mask = ch.enablelighting ? static_cast<std::uint8_t>(mask) : 0u;
    };
    lit_chan(5u, key.litchan[0]);
    lit_chan(6u, key.litchan[1]);
    lit_chan(7u, key.litchan[2]);
    lit_chan(8u, key.litchan[3]);
    // Record which channel-control regs were actually loaded (slots 5..8 map to
    // litchan 0..3). A captured channel selecting MatSource::Register drives
    // the material register; an unconfigured channel stays vertex-color
    // passthrough.
    for (std::uint32_t j = 0; j < 4u; ++j)
      if (draw.chan_reg_mask & (1u << (5u + j)))
        key.chan_captured_mask |= static_cast<std::uint8_t>(1u << j);
    if (draw.chan_reg_mask & 0x1u) {
      std::uint32_t n = bits(draw.chan_regs[0], 2, 0);
      if (n > 2u)
        n = 2u;
      key.num_color_chans = static_cast<std::uint8_t>(n);
    } else {
      // numColorChans reg (XF 0x1009) never captured. GX games always program
      // it; a 0 here is a capture artifact, not "no channels". Assume both
      // channels present so the WGSL numColorChans gate never spuriously blacks
      // a draw whose count we simply did not observe.
      key.num_color_chans = 2u;
    }
    key.lit_valid =
        (key.litchan[0].enablelighting || key.litchan[1].enablelighting ||
         key.litchan[2].enablelighting || key.litchan[3].enablelighting)
            ? 1u
            : 0u;
    if (key.has_pos_mtx_idx != 0u && key.lit_valid != 0u)
      ++counters.per_vertex_normal_matrix;
    for (std::uint32_t c = 0; c < 4u; ++c) {
      const std::uint8_t mask = key.litchan[c].light_mask;
      for (std::uint32_t l = 0; l < 8u; ++l)
        if ((mask & (1u << l)) && draw.light_word_mask[l] == 0u) {
          ++counters.lit_light_missing;
          c = 4u;
          break;
        }
    }
    // Only count normals/lighting as ignored when we fall back to passthrough.
    if (walk.has_normal && key.lit_valid == 0u)
      ++counters.normals_ignored;
    if (key.lit_valid == 0u && (draw.chan_reg_mask & (1u << 5)) &&
        (draw.chan_regs[5] & 0x2u) != 0u)
      ++counters.lighting_ignored;

    for (std::uint32_t i = 0; i < num_tex_gens; ++i) {
      const std::uint32_t reg = (0x1040u - 0x1018u) + i;
      TexGenKey &tg = key.tex_gens[i];
      tg.enabled = 1;
      if (draw.xf_reg_mask & (1ull << reg)) {
        const std::uint32_t info = draw.xf_regs[reg];
        // XFMemory.h TexMtxInfo bitfields.
        tg.projection = static_cast<std::uint8_t>(bits(info, 1, 1));
        tg.inputform = static_cast<std::uint8_t>(bits(info, 1, 2));
        tg.texgentype = static_cast<std::uint8_t>(bits(info, 3, 4));
        tg.sourcerow = static_cast<std::uint8_t>(bits(info, 5, 7));
        tg.embosssourceshift = static_cast<std::uint8_t>(bits(info, 3, 12));
        tg.embosslightshift = static_cast<std::uint8_t>(bits(info, 3, 15));
      } else {
        tg.sourcerow = static_cast<std::uint8_t>(TexSourceRow::Tex0) +
                       static_cast<std::uint8_t>(i);
      }
      // Regular / Color0 / Color1 / Emboss are all generated (item 5). Emboss
      // without per-vertex NBT uses the cross-draw cached tangent/binormal, so
      // it is an exercised fallback rather than an unsupported path.
      if (static_cast<TexGenType>(tg.texgentype) == TexGenType::EmbossMap &&
          !walk.has_nbt)
        ++counters.texgen_emboss_cached_nbt;
      {
        const auto t = static_cast<TexGenType>(tg.texgentype);
        if (t == TexGenType::Color0 || t == TexGenType::Color1) {
          const std::uint32_t ch = t == TexGenType::Color0 ? 0u : 1u;
          if (key.litchan[ch].enablelighting)
            ++counters.texgen_color_lit;
          else
            ++counters.texgen_color_unlit;
        }
      }

      // The current vertex layout emits position and Tex0..Tex3 source rows for
      // regular matrix texgens. Classify every other legal source instead of
      // silently leaving its coordinate at the shader default.
      if (static_cast<TexGenType>(tg.texgentype) == TexGenType::Regular) {
        const auto source = static_cast<TexSourceRow>(tg.sourcerow);
        bool unsupported_source = true;
        if (source == TexSourceRow::Geom ||
            (tg.sourcerow >= static_cast<std::uint8_t>(TexSourceRow::Tex0) &&
             tg.sourcerow <
                 static_cast<std::uint8_t>(TexSourceRow::Tex0) + kMaxTexGens)) {
          unsupported_source = false;
        } else if (source == TexSourceRow::Normal) {
          ++counters.texgen_source_normal;
          if (!walk.has_normal)
            ++counters.texgen_source_normal_default;
          unsupported_source = false;
        } else if (source == TexSourceRow::Colors) {
          ++counters.texgen_source_colors;
        } else if (source == TexSourceRow::BinormalT ||
                   source == TexSourceRow::BinormalB) {
          ++counters.texgen_source_binormal;
        } else if (tg.sourcerow <
                   static_cast<std::uint8_t>(TexSourceRow::Tex0) + 8u) {
          ++counters.texgen_source_tex47;
        } else {
          ++counters.texgen_source_unknown;
        }
        if (unsupported_source)
          ++counters.unsupported_texgen;
      }
    }
    std::uint32_t num_ind_stages = bits(gen_mode, 3, 16);
    if (num_ind_stages > kMaxIndirectStages) {
      ++counters.indirect_ignored;
      num_ind_stages = kMaxIndirectStages;
    }
    key.num_ind_stages = static_cast<std::uint8_t>(num_ind_stages);
    if (num_ind_stages != 0u)
      ++counters.indirect_active;
    const std::uint32_t indref = bp_valid_[0x27] ? bp_regs_[0x27] : 0u;
    for (std::uint32_t i = 0; i < num_ind_stages; ++i) {
      IndirectStageKey &ind = key.ind_stages[i];
      ind.texmap = static_cast<std::uint8_t>(bits(indref, 3, 6u * i));
      ind.texcoord = static_cast<std::uint8_t>(bits(indref, 3, 6u * i + 3u));
      const std::uint32_t scale =
          bp_valid_[0x25u + i / 2u] ? bp_regs_[0x25u + i / 2u] : 0u;
      const std::uint32_t shift = (i & 1u) != 0u ? 8u : 0u;
      ind.scale_s = static_cast<std::uint8_t>(bits(scale, 4, shift));
      ind.scale_t = static_cast<std::uint8_t>(bits(scale, 4, shift + 4u));
    }
    const std::uint32_t tex_format = draw.texture.format;
    if (draw.texture.valid &&
        (tex_format == 0x8u || tex_format == 0x9u || tex_format == 0xAu))
      ++counters.tlut_texture;
    key.textured =
        (draw.texture.valid && draw.texture.resolved && num_tex_gens > 0u &&
         draw.texture.width > 0u && draw.texture.height > 0u)
            ? 1u
            : 0u;

    // TEV combiner (S14). Combiner reg 0xC0 present => port the integer TEV;
    // otherwise leave tev_valid 0 and the passthrough fragment renders (used by
    // synthetic slices that never write combiner state).
    if (bp_valid_[0xC0]) {
      key.tev_valid = 1;
      std::uint32_t stages = bits(gen_mode, 4, 10) + 1u; // numtevstages + 1
      if (stages > kMaxTevStages) {
        counters.tev_stages_over += stages - kMaxTevStages;
        stages = kMaxTevStages;
      }
      key.num_tev_stages = static_cast<std::uint8_t>(stages);
      auto swap_table = [&](std::uint32_t id, std::uint8_t out[4]) {
        const std::uint32_t rg = bp_regs_[0xF6u + 2u * id];
        const std::uint32_t ba = bp_regs_[0xF6u + 2u * id + 1u];
        out[0] = static_cast<std::uint8_t>(bits(rg, 2, 0)); // swap_rb -> red
        out[1] = static_cast<std::uint8_t>(bits(rg, 2, 2)); // swap_ga -> green
        out[2] = static_cast<std::uint8_t>(bits(ba, 2, 0)); // blue
        out[3] = static_cast<std::uint8_t>(bits(ba, 2, 2)); // alpha
      };
      for (std::uint32_t n = 0; n < stages; ++n) {
        TevStageKey &ts = key.tev_stages[n];
        const std::uint32_t cc = bp_regs_[0xC0u + 2u * n]; // ColorCombiner
        const std::uint32_t ac = bp_regs_[0xC1u + 2u * n]; // AlphaCombiner
        ts.cc_d = static_cast<std::uint8_t>(bits(cc, 4, 0));
        ts.cc_c = static_cast<std::uint8_t>(bits(cc, 4, 4));
        ts.cc_b = static_cast<std::uint8_t>(bits(cc, 4, 8));
        ts.cc_a = static_cast<std::uint8_t>(bits(cc, 4, 12));
        ts.cc_bias = static_cast<std::uint8_t>(bits(cc, 2, 16));
        ts.cc_op = static_cast<std::uint8_t>(bits(cc, 1, 18));
        ts.cc_clamp = static_cast<std::uint8_t>(bits(cc, 1, 19));
        ts.cc_scale = static_cast<std::uint8_t>(bits(cc, 2, 20));
        ts.cc_dest = static_cast<std::uint8_t>(bits(cc, 2, 22));
        ts.ac_a = static_cast<std::uint8_t>(bits(ac, 3, 13));
        ts.ac_b = static_cast<std::uint8_t>(bits(ac, 3, 10));
        ts.ac_c = static_cast<std::uint8_t>(bits(ac, 3, 7));
        ts.ac_d = static_cast<std::uint8_t>(bits(ac, 3, 4));
        ts.ac_bias = static_cast<std::uint8_t>(bits(ac, 2, 16));
        ts.ac_op = static_cast<std::uint8_t>(bits(ac, 1, 18));
        ts.ac_clamp = static_cast<std::uint8_t>(bits(ac, 1, 19));
        ts.ac_scale = static_cast<std::uint8_t>(bits(ac, 2, 20));
        ts.ac_dest = static_cast<std::uint8_t>(bits(ac, 2, 22));
        // TwoTevStageOrders (BP 0x28 + n/2), even/odd halves.
        const std::uint32_t ord = bp_regs_[0x28u + n / 2u];
        const bool odd = (n & 1u) != 0u;
        ts.tevorders_texmap =
            static_cast<std::uint8_t>(odd ? bits(ord, 3, 12) : bits(ord, 3, 0));
        ts.tevorders_texcoord =
            static_cast<std::uint8_t>(odd ? bits(ord, 3, 15) : bits(ord, 3, 3));
        ts.tevorders_enable =
            static_cast<std::uint8_t>(odd ? bits(ord, 1, 18) : bits(ord, 1, 6));
        ts.tevorders_colorchan =
            static_cast<std::uint8_t>(odd ? bits(ord, 3, 19) : bits(ord, 3, 7));
        const std::uint32_t indirect =
            bp_valid_[0x10u + n] ? bp_regs_[0x10u + n] : 0u;
        ts.ind_stage = static_cast<std::uint8_t>(bits(indirect, 2, 0));
        ts.ind_format = static_cast<std::uint8_t>(bits(indirect, 2, 2));
        ts.ind_bias = static_cast<std::uint8_t>(bits(indirect, 3, 4));
        ts.ind_bump_alpha = static_cast<std::uint8_t>(bits(indirect, 2, 7));
        ts.ind_matrix_index = static_cast<std::uint8_t>(bits(indirect, 2, 9));
        ts.ind_matrix_id = static_cast<std::uint8_t>(bits(indirect, 2, 11));
        ts.ind_wrap_s = static_cast<std::uint8_t>(bits(indirect, 3, 13));
        ts.ind_wrap_t = static_cast<std::uint8_t>(bits(indirect, 3, 16));
        ts.ind_use_original_lod =
            static_cast<std::uint8_t>(bits(indirect, 1, 19));
        ts.ind_add_prev = static_cast<std::uint8_t>(bits(indirect, 1, 20));
        const bool samples_indirect =
            ts.ind_matrix_index != 0u || ts.ind_bump_alpha != 0u;
        if (samples_indirect && ts.ind_stage >= num_ind_stages)
          ++counters.indirect_ignored;
        if (ts.ind_matrix_id == 3u || ts.ind_use_original_lod != 0u)
          ++counters.indirect_ignored;
        if (ts.tevorders_enable != 0u && ts.tevorders_texmap != 0u)
          ++counters.tev_multi_texmap;
        // Konst selectors (AllTevKSels ksel[n/2]).
        const std::uint32_t ksel = bp_regs_[0xF6u + n / 2u];
        ts.ksel_kc = static_cast<std::uint8_t>(odd ? bits(ksel, 5, 14)
                                                   : bits(ksel, 5, 4));
        ts.ksel_ka = static_cast<std::uint8_t>(odd ? bits(ksel, 5, 19)
                                                   : bits(ksel, 5, 9));
        swap_table(bits(ac, 2, 0), ts.ras_swap); // rswap
        swap_table(bits(ac, 2, 2), ts.tex_swap); // tswap
      }
      if (bp_valid_[0xF3]) { // AlphaTest (BP 0xF3)
        const std::uint32_t at = bp_regs_[0xF3];
        key.alpha_comp0 = static_cast<std::uint8_t>(bits(at, 3, 16));
        key.alpha_comp1 = static_cast<std::uint8_t>(bits(at, 3, 19));
        key.alpha_logic = static_cast<std::uint8_t>(bits(at, 2, 22));
      }
    }

    if (key.tev_valid != 0u && key.textured != 0u && key.num_tex_gens != 0u) {
      ++counters.texcoord_scale_active;
      bool scale_mismatch = false;
      auto compare_scale = [&](std::uint32_t coord, std::uint32_t texmap) {
        if (coord >= key.num_tex_gens)
          coord = 0u;
        texmap &= 7u;
        const ar::ConsumedTexture *texture = &draw.textures[texmap];
        if (!texture->valid || !texture->resolved || texture->width == 0u ||
            texture->height == 0u) {
          if (!draw.texture.valid || !draw.texture.resolved ||
              (draw.texture.slot & 7u) != texmap)
            return;
          texture = &draw.texture;
        }
        const std::uint32_t sreg = 0x30u + 2u * coord;
        const std::uint32_t treg = sreg + 1u;
        const std::uint32_t scale_s =
            (bp_valid_[sreg] ? bits(bp_regs_[sreg], 16, 0) : 0u) + 1u;
        const std::uint32_t scale_t =
            (bp_valid_[treg] ? bits(bp_regs_[treg], 16, 0) : 0u) + 1u;
        scale_mismatch |=
            scale_s != texture->width || scale_t != texture->height;
      };
      for (std::uint32_t n = 0; n < key.num_tev_stages; ++n) {
        const TevStageKey &ts = key.tev_stages[n];
        if (ts.tevorders_enable != 0u)
          compare_scale(ts.tevorders_texcoord, ts.tevorders_texmap);
        if (ts.ind_stage < key.num_ind_stages &&
            (ts.ind_matrix_index != 0u || ts.ind_bump_alpha != 0u)) {
          const IndirectStageKey &ind = key.ind_stages[ts.ind_stage];
          compare_scale(ind.texcoord, ind.texmap);
        }
      }
      if (scale_mismatch)
        ++counters.texcoord_scale_mismatch;
    }

    pipe.cull_mode = static_cast<std::uint8_t>(cull);
    const std::uint32_t zmode = bp_valid_[0x40] ? bp_regs_[0x40] : 0x17u;
    pipe.depth_test = static_cast<std::uint8_t>(bits(zmode, 1, 0));
    pipe.depth_func = static_cast<std::uint8_t>(bits(zmode, 3, 1));
    pipe.depth_update = static_cast<std::uint8_t>(bits(zmode, 1, 4));
    const std::uint32_t cmode0 = bp_valid_[0x41] ? bp_regs_[0x41] : 0u;
    pipe.blend_enable = static_cast<std::uint8_t>(bits(cmode0, 1, 0));
    pipe.blend_subtract = static_cast<std::uint8_t>(bits(cmode0, 1, 11));
    pipe.dst_factor = static_cast<std::uint8_t>(bits(cmode0, 3, 5));
    pipe.src_factor = static_cast<std::uint8_t>(bits(cmode0, 3, 8));
    pipe.color_update = static_cast<std::uint8_t>(bits(cmode0, 1, 3));
    const std::uint32_t dst_alpha = bp_valid_[0x42] ? bp_regs_[0x42] : 0u;
    const std::uint32_t pe_control = bp_valid_[0x43] ? bp_regs_[0x43] : 0u;
    const bool target_has_alpha = bits(pe_control, 3, 0) == 1u;
    pipe.alpha_update =
        static_cast<std::uint8_t>(bits(cmode0, 1, 4) != 0u && target_has_alpha);

    // Dolphin BlendingState keeps RGB and alpha factors separate. EFB formats
    // without alpha read destination alpha as one, and color factors collapse
    // to their alpha equivalents when operating on the alpha channel.
    if (!target_has_alpha) {
      if (pipe.src_factor == 6u)
        pipe.src_factor = 1u;
      else if (pipe.src_factor == 7u)
        pipe.src_factor = 0u;
      if (pipe.dst_factor == 6u)
        pipe.dst_factor = 1u;
      else if (pipe.dst_factor == 7u)
        pipe.dst_factor = 0u;
    }
    pipe.src_factor_alpha = pipe.src_factor;
    pipe.dst_factor_alpha = pipe.dst_factor;
    if (pipe.src_factor_alpha == 2u)
      pipe.src_factor_alpha = 6u;
    else if (pipe.src_factor_alpha == 3u)
      pipe.src_factor_alpha = 7u;
    if (pipe.dst_factor_alpha == 2u)
      pipe.dst_factor_alpha = 4u;
    else if (pipe.dst_factor_alpha == 3u)
      pipe.dst_factor_alpha = 5u;

    pipe.early_depth_test = static_cast<std::uint8_t>(bits(pe_control, 1, 6));
    if (pipe.depth_test != 0u && pipe.early_depth_test != 0u)
      ++counters.early_depth_active;

    // BP F4/F5 ZTexture. Dolphin writes the sampled depth only for late Z
    // tests; Wind Waker's JFWDisplay clear quad is REPLACE/U24 with depth
    // writes on. Keep early-Z/fog-only and malformed forms explicit gaps rather
    // than silently emitting the wrong fragment-depth contract.
    const std::uint32_t ztex2 = bp_valid_[0xF5] ? bp_regs_[0xF5] : 0u;
    const std::uint32_t ztex_type = bits(ztex2, 2, 0);
    const std::uint32_t ztex_op = bits(ztex2, 2, 2);
    if (ztex_op != 0u) {
      const bool supported = ztex_op <= 2u && ztex_type <= 2u &&
                             key.tev_valid != 0u && key.textured != 0u &&
                             pipe.depth_test != 0u && pipe.depth_update != 0u &&
                             pipe.early_depth_test == 0u;
      if (supported) {
        key.ztex_op = static_cast<std::uint8_t>(ztex_op);
        key.ztex_type = static_cast<std::uint8_t>(ztex_type);
        ++counters.ztexture_active;
      } else {
        ++counters.ztexture_ignored;
      }
    }
    const bool use_dst_alpha = bits(dst_alpha, 1, 8) != 0u &&
                               pipe.alpha_update != 0u && target_has_alpha;
    if (use_dst_alpha) {
      pipe.src_factor_alpha = 1u;
      pipe.dst_factor_alpha = 0u;
    }
    key.use_dst_alpha = use_dst_alpha ? 1u : 0u;
    key.dst_alpha = static_cast<std::uint8_t>(bits(dst_alpha, 8, 0));
    if (use_dst_alpha)
      ++counters.dst_alpha_active;
    if (bits(cmode0, 1, 1) != 0u)
      ++counters.logic_op_ignored;

    derived.version = version_;
    derived.vtx_fmt = draw.vtx_fmt;
    std::memcpy(derived.xf_regs, draw.xf_regs, sizeof(derived.xf_regs));
    derived.xf_reg_mask = draw.xf_reg_mask;
    std::memcpy(derived.chan_regs, draw.chan_regs, sizeof(derived.chan_regs));
    derived.chan_reg_mask = draw.chan_reg_mask;
    std::memcpy(derived.light_word_mask, draw.light_word_mask, sizeof(derived.light_word_mask));
    derived.textures = textures;
    derived.pipeline = plan.pipeline;
    derived.valid = true;
    for (auto member : counter_members)
      derived.delta.*member = counters.*member - before.*member;
    if (hit) {
      ++derived.verified;
      if (std::memcmp(&derived.pipeline, &plan.pipeline, sizeof(plan.pipeline)) != 0 &&
          ++derived.mismatches <= 8)
        std::fprintf(stderr, "[gx-derived] mismatch #%llu: vtx_fmt=%u version=%llx\n", derived.mismatches,
                     draw.vtx_fmt, static_cast<unsigned long long>(version_));
    } else {
      ++derived.misses;
    }
  }

  // Topology.
  const auto primitive =
      static_cast<ar::GxPrimitive>(draw.primitive & 0xF8u);
  if (primitive == ar::GxPrimitive::Lines ||
      primitive == ar::GxPrimitive::LineStrip) {
    pipe.primitive_topology = 1u;
  } else if (primitive == ar::GxPrimitive::Points) {
    pipe.primitive_topology = 2u;
  }
  std::uint32_t index_count = 0;
  if (draw.segments.empty()) {
    index_count = ar::build_topology_indices(
        primitive, 0u, static_cast<std::uint16_t>(draw.vertex_count),
        &plan.indices);
  } else {
    // Primitives fused onto the draw (GxCoreSink::submit_packet), each from
    // where the one before ends: the indices the draws apart had, rebased.
    std::uint32_t start = 0;
    for (const std::uint32_t segment : draw.segments) {
      const std::uint32_t count = segment & 0xFFFFu;
      index_count += ar::build_topology_indices(
          static_cast<ar::GxPrimitive>((segment >> 16) & 0xF8u),
          static_cast<std::uint16_t>(start), static_cast<std::uint16_t>(count),
          &plan.indices);
      start += count;
    }
  }
  if (index_count == 0u) {
    switch (primitive) {
    case ar::GxPrimitive::Quads:
      ++counters.topology_zero_quads;
      return noop("incomplete quad no-op");
    case ar::GxPrimitive::Triangles:
      ++counters.topology_zero_triangles;
      return noop("incomplete triangle no-op");
    case ar::GxPrimitive::TriangleStrip:
      ++counters.topology_zero_triangle_strip;
      return noop("incomplete triangle strip no-op");
    case ar::GxPrimitive::TriangleFan:
      ++counters.topology_zero_triangle_fan;
      return noop("incomplete triangle fan no-op");
    case ar::GxPrimitive::Lines:
      ++counters.topology_zero_lines;
      return noop("incomplete line no-op");
    case ar::GxPrimitive::LineStrip:
      ++counters.topology_zero_line_strip;
      return noop("incomplete line strip no-op");
    case ar::GxPrimitive::Points:
      ++counters.topology_zero_points;
      return noop("incomplete point no-op");
    default:
      ++counters.topology_zero_unknown;
      break;
    }
    ++counters.vertex_decode_failures;
    ++counters.vertex_topology_unsupported;
    return skip("unsupported or empty primitive");
  }

  // Vertex decode to the fixed layout.
  plan.vertex_count = draw.vertex_count;
  plan.vertices.assign(
      static_cast<std::size_t>(draw.vertex_count) * kVertexFloats, 0.f);
  const std::uint8_t* payload = draw.vertex_payload.data();
  const std::size_t payload_size = draw.vertex_payload.size();
  // Each indexed entry's array, found once for the draw (it was looked up
  // for every element of every vertex), and each entry's component scale.
  // Whether it is usable is still decided where the first vertex reaches
  // it, as before.
  const ar::ConsumedArrayInput* entry_arrays[24] = {};
  float entry_scales[24];
  for (std::uint32_t e = 0; e < walk.entry_count; ++e) {
    const WalkEntry& entry = walk.entries[e];
    if (entry.vcd_type > 1u)
      entry_arrays[e] = find_array(draw, entry.attr);
    entry_scales[e] = 1.f / static_cast<float>(1u << entry.frac);
  }
  for (std::uint32_t v = 0; v < draw.vertex_count; ++v) {
    float* out_vertex = plan.vertices.data() +
                        static_cast<std::size_t>(v) * kVertexFloats;
    // Defaults: color0/1 white, uv 0, posmtx = current matrix's first row.
    out_vertex[4] = out_vertex[5] = out_vertex[6] = out_vertex[7] = 1.f;
    out_vertex[8] = out_vertex[9] = out_vertex[10] = out_vertex[11] = 1.f;
    // NBT part ordinal for index3 (three separate normal/binormal/tangent
    // entries); a single 9-component entry decodes all three at once.
    std::uint32_t normal_part = 0;
    std::uint32_t posmtx_row = draw.current_pn_matrix * 3u;
    std::uint32_t texmtxidx_packed = 0; // texgens 0..3, one byte each
    std::uint32_t texmtxidx_packed_hi = 0; // texgens 4..7, one byte each
    std::size_t offset =
        static_cast<std::size_t>(v) * walk.vertex_size;
    for (std::uint32_t e = 0; e < walk.entry_count; ++e) {
      const WalkEntry& entry = walk.entries[e];
      if (offset >= payload_size) {
        ++counters.vertex_decode_failures;
        ++counters.vertex_payload_overrun;
        return skip("payload overrun");
      }
      const std::uint8_t* p = payload + offset;
      const std::uint8_t* element = nullptr;
      std::uint32_t advance = 0;
      if (entry.kind == WalkEntry::kPosMtxIdx) {
        posmtx_row = p[0];
        offset += 1;
        continue;
      }
      if (entry.kind == WalkEntry::kTexMtxIdx) {
        // GX per-vertex TEXMTXIDX: the byte is a matrix-memory row index (GX
        // pre-multiplies by 3, same convention as PNMTXIDX). Pack one byte per
        // texgen so the VS can select transformmatrices[row] per vertex.
        if (entry.out_slot < 4u) {
          texmtxidx_packed |= static_cast<std::uint32_t>(p[0])
                              << (8u * entry.out_slot);
        } else if (entry.out_slot < 8u) {
          texmtxidx_packed_hi |= static_cast<std::uint32_t>(p[0])
                                 << (8u * (entry.out_slot - 4u));
        }
        offset += 1;
        continue;
      }
      if (entry.vcd_type == 1u) {
        element = p;
        advance = entry.element_size;
      } else {
        const std::uint32_t idx_size = entry.vcd_type == 2u ? 1u : 2u;
        const std::uint32_t index = idx_size == 1u ? p[0] : load_be16(p);
        advance = idx_size;
        const ar::ConsumedArrayInput* array = entry_arrays[e];
        if (array == nullptr || !array->resolved ||
            array->host_data == nullptr) {
          ++counters.vertex_decode_failures;
          ++counters.vertex_array_unresolved;
          return skip("indexed attr has no resolved array");
        }
        const std::uint64_t element_offset =
            static_cast<std::uint64_t>(index) * array->stride;
        if (element_offset + entry.element_size > array->host_available) {
          ++counters.vertex_decode_failures;
          ++counters.vertex_array_out_of_bounds;
          return skip("indexed element outside resolved array");
        }
        element =
            static_cast<const std::uint8_t*>(array->host_data) +
            element_offset;
      }
      offset += advance;

      switch (entry.kind) {
      case WalkEntry::kPos: {
        std::uint32_t scalar = 0;
        component_scalar_size(entry.format, &scalar);
        const float scale = entry_scales[e];
        for (std::uint32_t c = 0; c < entry.count && c < 3u; ++c)
          out_vertex[c] = decode_scaled(element + c * scalar, entry.format, scale);
        if (entry.count == 2u)
          out_vertex[2] = 0.f;
        break;
      }
      case WalkEntry::kColor: {
        float rgba[4];
        decode_color(element, entry.format, rgba);
        float* dst = out_vertex + (entry.out_slot == 0u ? 4u : 8u);
        for (int c = 0; c < 4; ++c)
          dst[c] = rgba[c];
        break;
      }
      case WalkEntry::kTex: {
        if (entry.out_slot < kMaxTexGens) {
          std::uint32_t scalar = 0;
          component_scalar_size(entry.format, &scalar);
          float* dst = out_vertex + 12u + 2u * entry.out_slot;
          const float scale = entry_scales[e];
          dst[0] = decode_scaled(element, entry.format, scale);
          dst[1] = entry.count == 2u
                       ? decode_scaled(element + scalar, entry.format, scale)
                       : 0.f;
        }
        break;
      }
      case WalkEntry::kNormal: {
        // Normal (+ NBT binormal/tangent for emboss texgens). A single 9-part
        // entry (direct or non-index3 indexed) holds N,B,T contiguously; index3
        // arrives as three count==3 entries routed by normal_part.
        std::uint32_t scalar = 0;
        component_scalar_size(entry.format, &scalar);
        const float scale = entry_scales[e];
        auto decode3 = [&](const std::uint8_t* src, std::uint32_t dst_off) {
          float* dst = out_vertex + dst_off / 4u;
          for (std::uint32_t c = 0; c < 3u; ++c)
            dst[c] = decode_scaled(src + c * scalar, entry.format, scale);
        };
        if (entry.count >= 9u) {
          decode3(element, kVertexNormalOffset);
          decode3(element + 3u * scalar, kVertexBinormalOffset);
          decode3(element + 6u * scalar, kVertexTangentOffset);
          normal_part = 3;
        } else if (normal_part == 0u) {
          decode3(element, kVertexNormalOffset);
          normal_part = 1;
        } else if (normal_part == 1u) {
          decode3(element, kVertexBinormalOffset);
          normal_part = 2;
        } else if (normal_part == 2u) {
          decode3(element, kVertexTangentOffset);
          normal_part = 3;
        }
        break;
      }
      default:
        break;
      }
    }
    std::memcpy(out_vertex + 3, &posmtx_row, sizeof posmtx_row);
    std::memcpy(out_vertex + kVertexTexMtxIdxOffset / 4u, &texmtxidx_packed,
                sizeof texmtxidx_packed);
    std::memcpy(out_vertex + kVertexTexMtxIdxHiOffset / 4u,
                &texmtxidx_packed_hi, sizeof texmtxidx_packed_hi);
  }

  // Uniforms. Most draws are made with the transform state of the draw
  // before them (96 percent in the Forsaken Fortress, 17,500 draws a frame):
  // when the inputs below are the ones the plan's constants were made from,
  // they are kept, and the constants_id tells the submission layer that the
  // block is the same without comparing its 2.8 KB.
  // DOL_GXCORE_CONSTANTS_VERIFY=1 makes them again anyway and compares, and
  // DOL_GXCORE_CONSTANTS_KEEP=0 always makes them (debug).
  ConstantsInputs inputs{};
  inputs.xf_version = draw.xf_version;
  if (cached != nullptr) {
    inputs.has_cached = 1u;
    for (std::uint32_t k = 0; k < 3u; ++k) {
      inputs.cached[k] = cached->normal[k];
      inputs.cached[3u + k] = cached->tangent[k];
      inputs.cached[6u + k] = cached->binormal[k];
    }
  }
  inputs.lit_valid = key.lit_valid;
  inputs.chan_lit = channel_lit_path(key, 0u) || channel_lit_path(key, 1u) ? 1u : 0u;
  inputs.num_tex_gens = key.num_tex_gens;
  for (std::uint32_t i = 0; i < key.num_tex_gens && i < kMaxTexGens; ++i)
    if (key.tex_gens[i].projection != 0u)
      inputs.texgen_stq |= static_cast<std::uint8_t>(1u << i);
  static const bool keep_constants = [] {
    const char* env = std::getenv("DOL_GXCORE_CONSTANTS_KEEP");
    return env == nullptr || env[0] != '0';
  }();
  const bool reuse = keep_constants && inputs.xf_version != 0u && plan.constants_id != 0u &&
                     std::memcmp(&inputs, &plan.constants_inputs, sizeof inputs) == 0;
  static const bool verify_constants = [] {
    const char* env = std::getenv("DOL_GXCORE_CONSTANTS_VERIFY");
    return env != nullptr && env[0] == '1';
  }();
  static VertexShaderConstants verify_scratch;
  if (!reuse || verify_constants) {
    VertexShaderConstants& c = reuse ? verify_scratch : plan.constants;
    std::uint8_t texgen_row[kMaxTexGens];
    std::memset(texgen_row, 0, sizeof texgen_row);
    std::uint32_t unresolved = 0;
    c = VertexShaderConstants{};
    // Rows 0-29 are position_matrices laid end to end (3 rows of 4 per matrix)
    // and rows 30-62 are tex_matrices the same way, so the 63 per-row copies
    // load_matrix_row made (their validity result was discarded) are two block
    // copies. This ran for every draw, about 18,000 a frame in heavy scenes.
    static_assert(sizeof(draw.position_matrices) == 30u * 4u * sizeof(float));
    static_assert(sizeof(draw.tex_matrices) == 33u * 4u * sizeof(float));
    std::memcpy(&c.transformmatrices[0][0], draw.position_matrices,
                sizeof(draw.position_matrices));
    std::memcpy(&c.transformmatrices[30][0], draw.tex_matrices,
                sizeof(draw.tex_matrices));
    const std::uint32_t pn_row = draw.current_pn_matrix * 3u;
    for (std::uint32_t k = 0; k < 3u; ++k)
      (void)load_matrix_row(draw, pn_row + k, c.posnormalmatrix[k]);
    // Normal matrix (XF 0x400): matrix M = current_pn_matrix, 3 rows of
    // 3 (Dolphin VertexShaderManager normalMatrices[3*(PosNormalMtxIdx&31)] == our
    // per-matrix [M] slot). Fall back to the position rows when it was never
    // captured (keeps the A1 behavior for uncaptured draws).
    {
      const std::uint32_t m = draw.current_pn_matrix;
      const bool nm_valid =
          m < DOL_GX_RECOMP_NORMAL_MATRIX_COUNT &&
          draw.normal_matrix_word_mask[m] ==
              ((1u << DOL_GX_RECOMP_NORMAL_MATRIX_WORDS) - 1u);
      for (std::uint32_t k = 0; k < 3u; ++k) {
        if (nm_valid) {
          c.posnormalmatrix[3u + k][0] = draw.normal_matrices[m][3u * k + 0u];
          c.posnormalmatrix[3u + k][1] = draw.normal_matrices[m][3u * k + 1u];
          c.posnormalmatrix[3u + k][2] = draw.normal_matrices[m][3u * k + 2u];
          c.posnormalmatrix[3u + k][3] = 0.f;
        } else {
          (void)load_matrix_row(draw, pn_row + k, c.posnormalmatrix[3u + k]);
        }
      }
    }
    // Dolphin uploads the complete packed normal-matrix row bank for vertex
    // formats carrying PNMTXIDX. XF stores 96 floats as 32 rows of three; the
    // C/WGSL uniform pads each row to vec4 alignment.
    for (std::uint32_t row = 0; row < 32u; ++row) {
      const std::uint32_t matrix = row / 3u;
      const std::uint32_t matrix_row = row % 3u;
      if (matrix >= DOL_GX_RECOMP_NORMAL_MATRIX_COUNT)
        continue;
      const std::uint16_t row_mask =
          static_cast<std::uint16_t>(0x7u << (3u * matrix_row));
      if ((draw.normal_matrix_word_mask[matrix] & row_mask) != row_mask)
        continue;
      for (std::uint32_t col = 0; col < 3u; ++col)
        c.normalmatrices[row][col] =
            draw.normal_matrices[matrix][3u * matrix_row + col];
    }

    // Lighting uniforms (S15): 8 XF lights + 4 material/ambient registers. Light
    // words (gx_recomp.h): [3]=RGBA8 color, [4..6]=cosatt, [7..9]=distatt,
    // [10..12]=pos, [13..15]=dir (f32 bit patterns). Lights are read only when a
    // light is enabled (lit_valid); the material registers are read by any channel
    // taking the full path (register material/ambient or lighting). Both are left
    // zero otherwise — the WGSL for such a draw never references them.
    auto unpack_rgba8 = [](std::uint32_t col, std::int32_t out[4]) {
      out[0] = static_cast<std::int32_t>((col >> 24) & 0xFFu);
      out[1] = static_cast<std::int32_t>((col >> 16) & 0xFFu);
      out[2] = static_cast<std::int32_t>((col >> 8) & 0xFFu);
      out[3] = static_cast<std::int32_t>(col & 0xFFu);
    };
    if (key.lit_valid) {
      auto f32_of = [](std::uint32_t raw) {
        float f;
        std::memcpy(&f, &raw, sizeof f);
        return f;
      };
      for (std::uint32_t i = 0; i < 8u; ++i) {
        const std::uint32_t* w = draw.light_words[i];
        unpack_rgba8(w[3], c.lights[i].color);
        for (std::uint32_t k = 0; k < 3u; ++k) {
          c.lights[i].cosatt[k] = f32_of(w[4 + k]);
          c.lights[i].distatt[k] = f32_of(w[7 + k]);
          c.lights[i].pos[k] = f32_of(w[10 + k]);
          c.lights[i].dir[k] = f32_of(w[13 + k]);
        }
      }
    }
    if (channel_lit_path(key, 0u) || channel_lit_path(key, 1u)) {
      // I_MATERIALS: [0]=amb0(0x100A), [1]=amb1(0x100B), [2]=mat0(0x100C),
      // [3]=mat1(0x100D) — chan_regs slots 1..4.
      const std::uint32_t mat_slot[4] = {1u, 2u, 3u, 4u};
      for (std::uint32_t m = 0; m < 4u; ++m) {
        const std::uint32_t col = (draw.chan_reg_mask & (1u << mat_slot[m]))
                                      ? draw.chan_regs[mat_slot[m]]
                                      : 0u;
        unpack_rgba8(col, c.materials[m]);
      }
    }
    // Projection (GC 6-param form; Dolphin VertexShaderManager layout).
    const float* pr = draw.projection;
    if (draw.projection_type == 1u) { // orthographic
      c.projection[0][0] = pr[0];
      c.projection[0][3] = pr[1];
      c.projection[1][1] = pr[2];
      c.projection[1][3] = pr[3];
      c.projection[2][2] = pr[4];
      c.projection[2][3] = pr[5];
      c.projection[3][3] = 1.f;
    } else { // perspective
      c.projection[0][0] = pr[0];
      c.projection[0][2] = pr[1];
      c.projection[1][1] = pr[2];
      c.projection[1][2] = pr[3];
      c.projection[2][2] = pr[4];
      c.projection[2][3] = pr[5];
      c.projection[3][2] = -1.f;
    }
    // Per-texgen matrix rows via XF MatrixIndexA/B (CPMemory.h TMatrixIndexA/B).
    const bool mat_idx_a_valid = (draw.xf_reg_mask & 0x1ull) != 0ull;
    const bool mat_idx_b_valid = (draw.xf_reg_mask & 0x2ull) != 0ull;
    const std::uint32_t mat_idx_a = draw.xf_regs[0];
    const std::uint32_t mat_idx_b = draw.xf_regs[1];
    for (std::uint32_t i = 0; i < key.num_tex_gens; ++i) {
      std::uint32_t row = 60u; // GX_IDENTITY row when never configured
      bool row_known = false;
      if (i < 4u && mat_idx_a_valid) {
        row = bits(mat_idx_a, 6, 6 + 6 * i);
        row_known = true;
      } else if (i >= 4u && mat_idx_b_valid) {
        // CPMemory.h TMatrixIndexB begins Tex4MtxIdx at bit 0; unlike A it has
        // no leading PosNormalMtxIdx field.
        row = bits(mat_idx_b, 6, 6 * (i - 4u));
        row_known = true;
      }
      float rows[3][4];
      bool resolved = row_known;
      for (std::uint32_t k = 0; k < 3u; ++k)
        resolved = load_matrix_row(draw, row + k, rows[k]) && resolved;
      if (!resolved) {
        // Never-written matrix memory: pass raw ST through (identity) and
        // count it loudly instead of collapsing every UV to zero.
        identity_rows(rows);
        ++unresolved;
      }
      // Dual-texture post transform (XF 0x1012, on since GXInit): the texture
      // matrix's result goes through a second matrix from post-transform memory
      // (Dolphin VertexShaderGen WriteTexCoordTransforms). Without normalization
      // the two are one 3x4 matrix, folded here so the shader is unchanged:
      // coord.w is 1, so each post row's w joins the fourth column, and a
      // two-row (ST) texgen's third coordinate is the constant 1. The lava's
      // pattern and colour ramp are projected this way (d_magma.cpp).
      if ((draw.post_tex_mask & (1u << i)) != 0u &&
          (draw.post_tex_normalize & (1u << i)) == 0u) {
        const float* p = draw.post_tex_rows[i];
        const bool stq = key.tex_gens[i].projection != 0u;
        float folded[3][4];
        for (std::uint32_t r = 0; r < 3u; ++r) {
          const float* pr = p + 4u * r;
          for (std::uint32_t col = 0; col < 4u; ++col)
            folded[r][col] = pr[0] * rows[0][col] + pr[1] * rows[1][col] +
                             (stq ? pr[2] * rows[2][col] : 0.f);
          folded[r][3] += pr[3] + (stq ? 0.f : pr[2]);
        }
        std::memcpy(rows, folded, sizeof rows);
      }
      for (std::uint32_t k = 0; k < 3u; ++k)
        std::memcpy(c.texmatrices[3u * i + k], rows[k], sizeof rows[k]);
      if (i < kMaxTexGens)
        texgen_row[i] = static_cast<std::uint8_t>(resolved ? row : 0xFFu);
    }
    // Cached N/B/T fallback (Dolphin ConstantManager cached_normal): the
    // incoming cross-draw cache, for a draw whose format omits the attribute
    // (see the end of the plan, where this draw advances the cache).
    if (cached != nullptr) {
      for (std::uint32_t k = 0; k < 3u; ++k) {
        c.cached_normal[k] = cached->normal[k];
        c.cached_tangent[k] = cached->tangent[k];
        c.cached_binormal[k] = cached->binormal[k];
      }
    }
    if (!reuse) {
      std::memcpy(plan.texgen_row, texgen_row, sizeof texgen_row);
      plan.constants_unresolved = unresolved;
      plan.constants_inputs = inputs;
      static std::atomic<std::uint64_t> next_constants_id{0};
      plan.constants_id = next_constants_id.fetch_add(1u, std::memory_order_relaxed) + 1u;
    } else {
      static std::uint64_t mismatches = 0;
      if ((std::memcmp(&c, &plan.constants, sizeof c) != 0 ||
           std::memcmp(texgen_row, plan.texgen_row, sizeof texgen_row) != 0 ||
           unresolved != plan.constants_unresolved) &&
          ++mismatches <= 8u)
        std::fprintf(stderr, "[gx-constants] kept constants differ from remade ones (#%llu, xf version %llu)\n",
                     static_cast<unsigned long long>(mismatches), static_cast<unsigned long long>(inputs.xf_version));
    }
  }
  counters.unresolved_tex_matrix += plan.constants_unresolved;
  plan.matrix_index_a = (draw.xf_reg_mask & 0x1ull) != 0ull ? draw.xf_regs[0] : 0xFFFFFFFFu;

  // Viewport + texture.
  if ((draw.transform_flags & ar::kDrawTransformViewportValid) != 0u) {
    plan.viewport_valid = true;
    std::memcpy(plan.viewport, draw.viewport, sizeof plan.viewport);
  }
  if (bp_valid_[0x20u] && bp_valid_[0x21u]) {
    const std::uint32_t offset =
        bp_valid_[0x59u] ? bp_regs_[0x59u] : (171u | (171u << 10u));
    decode_scissor(plan, bp_regs_[0x20u], bp_regs_[0x21u], offset);
  }
  for (std::uint32_t t = 0; t < 8u; ++t) {
    const std::uint32_t mode0_reg =
        0x80u | ((t & 3u) | ((t & 4u) << 3u));
    const std::uint32_t mode1_reg = mode0_reg + 4u;
    PlanSampler& sampler = plan.samplers[t];
    if (bp_valid_[mode0_reg]) {
      const std::uint32_t mode0 = bp_regs_[mode0_reg];
      const std::uint32_t min_filter = bits(mode0, 3, 5);
      // TX_SETMODE0 bits 5-7: the mip mode in bits 5-6 (none, point, linear;
      // 3 samples as point) and the minification filter in bit 7 (Dolphin's
      // TexMode0). GX_LIN_MIP_NEAR is 5 and GX_NEAR_MIP_LIN 2 here.
      static constexpr std::uint8_t kMinFilter[8] = {
          0, 0, 0, 0, 1, 1, 1, 1,
      };
      static constexpr std::uint8_t kMipmapFilter[8] = {
          0, 1, 2, 1, 0, 1, 2, 1,
      };
      sampler.wrap_s = static_cast<std::uint8_t>(bits(mode0, 2, 0));
      sampler.wrap_t = static_cast<std::uint8_t>(bits(mode0, 2, 2));
      sampler.mag_filter = static_cast<std::uint8_t>(bits(mode0, 1, 4));
      sampler.min_filter = kMinFilter[min_filter];
      sampler.mipmap_filter = kMipmapFilter[min_filter];
      sampler.max_aniso = static_cast<std::uint8_t>(bits(mode0, 2, 19));
    }
    if (bp_valid_[mode1_reg]) {
      const std::uint32_t mode1 = bp_regs_[mode1_reg];
      sampler.min_lod = static_cast<std::uint8_t>(bits(mode1, 8, 0));
      sampler.max_lod = static_cast<std::uint8_t>(bits(mode1, 8, 8));
    }
  }
  if (key.textured != 0u) {
    const std::uint32_t used = used_texmap_mask(key);
    const ar::ConsumedTexture* primary = &draw.texture;
    if (texmap_popcount(used) == 1u) {
      for (std::uint32_t t = 0; t < 8u; ++t) {
        if ((used & (1u << t)) == 0u)
          continue;
        const ar::ConsumedTexture& candidate = draw.textures[t];
        if (candidate.valid && candidate.resolved && candidate.width > 0u &&
            candidate.height > 0u)
          primary = &candidate;
        break;
      }
    }
    plan.has_texture = true;
    plan.tex_slot = primary->slot & 7u;
    plan.tex_address = primary->address;
    plan.tex_size = primary->size;
    plan.tex_format = primary->format;
    plan.tex_width = primary->width;
    plan.tex_height = primary->height;
    plan.tex_data = primary->host_data;
    plan.tex_available = primary->host_available;
    // Carry the resolved TLUT palette for CI-format textures (A3 decode
    // consumer). Frontend leaves has_tlut false for non-CI/unresolved.
    plan.has_tlut = primary->has_tlut;
    plan.tlut_address = primary->tlut_address;
    plan.tlut_format = primary->tlut_format;
    plan.tlut_entries = primary->tlut_entries;
    plan.tlut_data = primary->tlut_host_data;
    plan.tlut_available = primary->tlut_host_available;
    // Multi-texmap (63/Mfin): when a TEV combines >1 texmap (THP YUV Y/U/V) the
    // single flat texture above is not enough. Populate the per-texmap set from
    // the per-slot bound textures. texmap_mask stays 0 for <=1 distinct texmap
    // so single-texmap draws retain the flat renderer path.
    if (texmap_popcount(used) > 1u) {
      plan.texmap_mask = used;
      for (std::uint32_t t = 0; t < 8u; ++t) {
        if ((used & (1u << t)) == 0u)
          continue;
        const ar::ConsumedTexture& src = draw.textures[t];
        PlanTexture& dst = plan.textures[t];
        dst.valid = src.valid && src.resolved && src.width > 0u &&
                    src.height > 0u;
        dst.address = src.address;
        dst.size = src.size;
        dst.format = src.format;
        dst.width = src.width;
        dst.height = src.height;
        dst.data = src.host_data;
        dst.available = src.host_available;
        dst.has_tlut = src.has_tlut;
        dst.tlut_address = src.tlut_address;
        dst.tlut_format = src.tlut_format;
        dst.tlut_entries = src.tlut_entries;
        dst.tlut_data = src.tlut_host_data;
        dst.tlut_available = src.tlut_host_available;
      }
    }
  }

  // Pixel-shader uniforms (S14): TEV color/konst registers + alpha refs.
  for (std::uint32_t r = 0; r < 4u; ++r)
    for (std::uint32_t ch = 0; ch < 4u; ++ch) {
      plan.pixel_constants.colors[r][ch] = tev_color_[r][ch];
      plan.pixel_constants.kcolors[r][ch] = konst_color_[r][ch];
    }
  if (bp_valid_[0xF3]) {
    plan.pixel_constants.alpha_ref[0] =
        static_cast<std::int32_t>(bits(bp_regs_[0xF3], 8, 0));
    plan.pixel_constants.alpha_ref[1] =
        static_cast<std::int32_t>(bits(bp_regs_[0xF3], 8, 8));
  }
  plan.pixel_constants.zbias[3] = static_cast<std::int32_t>(
      bp_valid_[0xF4] ? (bp_regs_[0xF4] & 0xFFFFFFu) : 0u);

  // BP SU_SSIZE/SU_TSIZE own rasterized texcoord scale. J3D normally writes
  // image width/height, but the hardware pairing is texcoord rather than
  // texmap and the two values may legitimately differ.
  for (std::uint32_t i = 0; i < 8u; ++i) {
    const std::uint32_t sreg = 0x30u + 2u * i;
    const std::uint32_t treg = sreg + 1u;
    plan.pixel_constants.texdims[i][2] = static_cast<std::int32_t>(
        (bp_valid_[sreg] ? bits(bp_regs_[sreg], 16, 0) : 0u) + 1u);
    plan.pixel_constants.texdims[i][3] = static_cast<std::int32_t>(
        (bp_valid_[treg] ? bits(bp_regs_[treg], 16, 0) : 0u) + 1u);
  }
  // texdims[texmap].xy: the size of each sampled texmap as the guest sees
  // it, which normalizes texcoords (Dolphin's texdim.xy). Slot 0 carries the
  // single-texmap path's texture; a multi-texmap draw fills each used slot.
  if (plan.has_texture) {
    if (plan.texmap_mask == 0u) {
      plan.pixel_constants.texdims[0][0] = static_cast<std::int32_t>(plan.tex_width);
      plan.pixel_constants.texdims[0][1] = static_cast<std::int32_t>(plan.tex_height);
    } else {
      for (std::uint32_t t = 0; t < 8u; ++t) {
        if ((plan.texmap_mask & (1u << t)) == 0u || !plan.textures[t].valid)
          continue;
        plan.pixel_constants.texdims[t][0] = static_cast<std::int32_t>(plan.textures[t].width);
        plan.pixel_constants.texdims[t][1] = static_cast<std::int32_t>(plan.textures[t].height);
      }
    }
  }

  for (std::uint32_t m = 0; m < 3u; ++m) {
    const std::uint32_t a = bp_valid_[0x06u + 3u * m]
                                ? bp_regs_[0x06u + 3u * m]
                                : 0u;
    const std::uint32_t b = bp_valid_[0x07u + 3u * m]
                                ? bp_regs_[0x07u + 3u * m]
                                : 0u;
    const std::uint32_t cword = bp_valid_[0x08u + 3u * m]
                                    ? bp_regs_[0x08u + 3u * m]
                                    : 0u;
    const std::int32_t matrix_shift = 17 - static_cast<std::int32_t>(
        bits(a, 2, 22) | (bits(b, 2, 22) << 2u) |
        (bits(cword, 1, 22) << 4u));
    plan.pixel_constants.indtexmtx[2u * m][0] = sx11(bits(a, 11, 0));
    plan.pixel_constants.indtexmtx[2u * m][1] = sx11(bits(b, 11, 0));
    plan.pixel_constants.indtexmtx[2u * m][2] = sx11(bits(cword, 11, 0));
    plan.pixel_constants.indtexmtx[2u * m][3] = matrix_shift;
    plan.pixel_constants.indtexmtx[2u * m + 1u][0] = sx11(bits(a, 11, 11));
    plan.pixel_constants.indtexmtx[2u * m + 1u][1] = sx11(bits(b, 11, 11));
    plan.pixel_constants.indtexmtx[2u * m + 1u][2] =
        sx11(bits(cword, 11, 11));
    plan.pixel_constants.indtexmtx[2u * m + 1u][3] = matrix_shift;
  }

  // Fog (S16, Dolphin PixelShaderManager fog constants). The fog uniform lives
  // in PixelShaderConstants, bound only on the TEV path, so fog emits on TEV
  // draws; a fog-enabled non-TEV draw is counted and renders fog-free.
  if (bp_valid_[0xF1]) {
    const std::uint32_t p3 = bp_regs_[0xF1];
    const std::uint32_t fsel = bits(p3, 3, 21);
    if (fsel != 0u) {
      if (key.tev_valid) {
        key.fog_fsel = static_cast<std::uint8_t>(fsel);
        key.fog_proj = static_cast<std::uint8_t>(bits(p3, 1, 20));
        // A (FogParam0 0xEE) and C (FogParam3 0xF1); the inf/inf NaN case
        // (Dolphin FogParams::GetA/GetC) maps to A=0, C=+-inf.
        const std::uint32_t p0 = bp_valid_[0xEE] ? bp_regs_[0xEE] : 0u;
        const bool nan_case =
            bits(p0, 8, 11) == 255u && bits(p3, 8, 11) == 255u;
        const float inf = std::numeric_limits<float>::infinity();
        plan.pixel_constants.fogf[0] = nan_case ? 0.f : fog_param_float(p0);
        plan.pixel_constants.fogf[1] =
            nan_case ? ((bits(p0, 1, 19) == 0u && bits(p3, 1, 19) == 0u) ? -inf
                                                                         : inf)
                     : fog_param_float(p3);
        plan.pixel_constants.fogi[1] = static_cast<std::int32_t>(
            bp_valid_[0xEF] ? (bp_regs_[0xEF] & 0xFFFFFFu) : 1u);
        plan.pixel_constants.fogi[3] = static_cast<std::int32_t>(
            bp_valid_[0xF0] ? (bp_regs_[0xF0] & 0xFFFFFFu) : 1u);
        const std::uint32_t col = bp_valid_[0xF2] ? bp_regs_[0xF2] : 0u;
        plan.pixel_constants.fogcolor[0] = static_cast<std::int32_t>(bits(col, 8, 16));
        plan.pixel_constants.fogcolor[1] = static_cast<std::int32_t>(bits(col, 8, 8));
        plan.pixel_constants.fogcolor[2] = static_cast<std::int32_t>(bits(col, 8, 0));
        // Range adjust (fogRange base 0xE8 + K[0..4] 0xE9..0xED). Center/width
        // need the viewport (Dolphin PixelShaderManager); fall back to Dolphin's
        // disabled default (center 0, width 1) when off or viewport-less.
        const std::uint32_t base = bp_valid_[0xE8] ? bp_regs_[0xE8] : 0u;
        if (bits(base, 1, 10) != 0u && plan.viewport_valid) {
          key.fog_range = 1;
          const float two_wd = 4.f * plan.viewport[0]; // viewport[0] = wd/2
          const int center = static_cast<int>(bits(base, 10, 0)) - 342;
          float ssc = two_wd != 0.f ? static_cast<float>(center) / two_wd : 0.f;
          plan.pixel_constants.fogf[2] = ssc * 2.f - 1.f;
          plan.pixel_constants.fogf[3] = two_wd;
          for (std::uint32_t i = 0, vi = 0; i < 5u; ++i) {
            const std::uint32_t kw =
                bp_valid_[0xE9u + i] ? bp_regs_[0xE9u + i] : 0u;
            // FogRangeKElement HI=bits(0,12) LO=bits(12,12); GetValue*scale(4).
            plan.pixel_constants.fogrange[vi / 4][vi % 4] =
                static_cast<float>(bits(kw, 12, 12)) / 256.f * 4.f; // LO
            ++vi;
            plan.pixel_constants.fogrange[vi / 4][vi % 4] =
                static_cast<float>(bits(kw, 12, 0)) / 256.f * 4.f; // HI
            ++vi;
          }
        } else {
          plan.pixel_constants.fogf[2] = 0.f;
          plan.pixel_constants.fogf[3] = 1.f;
        }
      } else {
        ++counters.fog_ignored;
      }
    }
  }

  // Cached N/B/T fallback (Dolphin VertexLoaderManager normal_cache /
  // ConstantManager cached_normal). The uniform has the incoming cross-draw
  // cache (above, with the constants) so a draw whose format omits an
  // attribute reads the last-decoded vertex's value instead of a zero input
  // (normalize(0) = NaN). Then, if THIS draw carried the attribute, advance the
  // cache to its LAST vertex's raw object-space value (Dolphin writes the cache
  // on the loader's m_remaining==0 vertex). N/B/T are appended in the fixed
  // vertex layout at these float slots.
  if (cached != nullptr) {
    if (plan.vertex_count > 0u) {
      const float* last =
          plan.vertices.data() +
          static_cast<std::size_t>(plan.vertex_count - 1u) * kVertexFloats;
      if (walk.has_normal)
        for (std::uint32_t k = 0; k < 3u; ++k)
          cached->normal[k] = last[kVertexNormalOffset / 4u + k];
      if (walk.has_nbt) {
        for (std::uint32_t k = 0; k < 3u; ++k) {
          cached->binormal[k] = last[kVertexBinormalOffset / 4u + k];
          cached->tangent[k] = last[kVertexTangentOffset / 4u + k];
        }
      }
    }
  }

  plan.ok = true;
  ++counters.draws_planned;
  return;
}

// --- Sink ---------------------------------------------------------------------

GxCoreSink::GxCoreSink() {
  consumer_.set_streaming(true);
  consumer_.set_assembly_totals(false);
  consumer_.set_draw_observer(&GxCoreSink::on_consumed_draw, this);
}

void GxCoreSink::on_consumed_draw(const ar::ConsumedDraw& draw,
                                  unsigned long long, void* user) {
  auto* self = static_cast<GxCoreSink*>(user);
  if (self->plan_observer_ == nullptr)
    return;
  DrawPlan& plan = self->scratch_plan_;
  self->pending_state_.build_draw_plan_into(draw, self->counters_,
                                            &self->cached_attrs_, plan);
  // A draw scope: the next draws with positions of their own, up to its
  // count, are its emitter's; any other draw ends it. A count with bit 23
  // set is a cloth's strips, which index their positions.
  const GxCoreState& state = self->pending_state_;
  if (state.bp_valid(GxCoreState::kDrawScopeRegister)) {
    const std::uint32_t count =
        state.bp_valid(GxCoreState::kDrawScopeCountRegister)
            ? state.bp(GxCoreState::kDrawScopeCountRegister)
            : 0u;
    self->scope_ = state.bp(GxCoreState::kDrawScopeRegister);
    self->scope_left_ = count & 0x7FFFFFu;
    self->scope_indexed_ = (count & 0x800000u) != 0u;
    self->scope_part_ = 1;
  }
  if (self->scope_left_ != 0u) {
    if (plan.ok && (plan.match_direct_position || self->scope_indexed_)) {
      plan.draw_scope = self->scope_;
      plan.draw_scope_part = self->scope_part_++;
      --self->scope_left_;
    } else {
      self->scope_left_ = 0;
    }
  }
  self->plan_observer_(plan, self->plan_observer_user_);
}

bool GxCoreSink::submit_packet(const ar::RenderPacket& packet) {
  if (packet.kind == ar::RenderPacketKind::State) {
    live_state_.apply(packet.state);
    if (!replay_overflow_) {
      if (since_draw_.size() < kMaxReplay)
        since_draw_.push_back(packet.state);
      else
        replay_overflow_ = true;
    }
    // A BP 0x52 with no copy observer (headless Mode A) is a stubbed copy;
    // when an observer is registered the copy is performed (counted below on
    // its resolved CopyDestination packet, which carries the real params).
    if (packet.state.kind == ar::RenderStateKind::BpReg &&
        packet.state.index == 0x52u && copy_observer_ == nullptr)
      ++counters_.efb_copy_ignored;
  }
  // EFB copy: the resolved CopyDestination carries the copy params. Flush the
  // pending draw FIRST (its geometry must be in the pass the copy resolves),
  // then perform the copy at its true stream position.
  if (packet.kind == ar::RenderPacketKind::Resource &&
      packet.resource.kind == ar::RenderResourceKind::CopyDestination &&
      copy_observer_ != nullptr) {
    consumer_.flush_assembly();
    const ar::RenderResourcePacket& r = packet.resource;
    // BP 0x4F clear R/A, 0x50 clear B/G, 0x51 clear Z (GXSetCopyClear encoding).
    const std::uint32_t c0 = live_state_.bp(0x4Fu);
    const std::uint32_t c1 = live_state_.bp(0x50u);
    const std::uint32_t cz = live_state_.bp(0x51u);
    // Clear write masks (Dolphin BPStructs ClearScreen): cmode0 bits 3/4,
    // zmode bit 4. Default enabled when the trace never wrote the reg
    // (GXInit leaves color/alpha/z update on).
    const bool cmode0_seen = live_state_.bp_valid(0x41u);
    const bool zmode_seen = live_state_.bp_valid(0x40u);
    const std::uint32_t cmode0 = live_state_.bp(0x41u);
    const std::uint32_t zmode = live_state_.bp(0x40u);
    const bool pe_control_seen = live_state_.bp_valid(0x43u);
    const std::uint32_t pe_control = live_state_.bp(0x43u);
    const EfbCopyCommand cmd{
        .dest_address = r.address,
        .byte_size = r.size,
        .format = r.format,
        .src_x = r.copy_src_x,
        .src_y = r.copy_src_y,
        .width = r.copy_src_width != 0u ? r.copy_src_width : r.width,
        .height = r.copy_src_height != 0u ? r.copy_src_height : r.height,
        .destination_width = r.width,
        .destination_height = r.height,
        .clear = r.copy_clear != 0u,
        .clear_r = (c0 >> 0u) & 0xFFu,
        .clear_g = (c1 >> 8u) & 0xFFu,
        .clear_b = (c1 >> 0u) & 0xFFu,
        .clear_a = (c0 >> 8u) & 0xFFu,
        .clear_z = cz & 0xFFFFFFu,
        .color_update = !cmode0_seen || ((cmode0 >> 3u) & 1u) != 0u,
        .alpha_update = !cmode0_seen || ((cmode0 >> 4u) & 1u) != 0u,
        .depth_update = !zmode_seen || ((zmode >> 4u) & 1u) != 0u,
        .efb_has_alpha = !pe_control_seen || (pe_control & 7u) == 1u,
    };
    copy_observer_(cmd, copy_observer_user_);
    if (r.format == 0xFu) {
      ++counters_.efb_display_copies;
    } else {
      ++counters_.efb_copies;
      // The frontend folds PE_CONTROL Z24 into the format's GXTexFmt Z bit.
      if ((r.format & 0x10u) != 0u)
        ++counters_.efb_copy_depth;
    }
  }
  // Draw fusion: a display list's primitives mostly follow one another with
  // no state between them (nine draws in ten at Forest Haven), and each was
  // planned, submitted and matched for Smooth Motion apart, to be batched
  // together at the end. Such a primitive joins the draw before: triangles
  // after triangles, positions from an array (not a particle's, a sprite's or
  // a wake's own), outside a draw scope and a tagged draw, and up to
  // kFuseMaxVertices. The consumer checks the rest (no texture or copy
  // between, the same vertex format and transform). DOL_GX_FUSE=0 turns it off.
  if (packet.kind == ar::RenderPacketKind::Draw) {
    static const bool s_fuse = [] {
      const char* env = std::getenv("DOL_GX_FUSE");
      return env == nullptr || env[0] != '0';
    }();
    const std::uint32_t primitive = packet.draw.primitive & 0xF8u;
    const bool triangles = primitive == 0x80u || primitive == 0x90u || primitive == 0x98u || primitive == 0xA0u;
    if (s_fuse && triangles && last_draw_triangles_ && since_draw_.empty() && !replay_overflow_ &&
        scope_left_ == 0u && !pending_state_.bp_valid(GxCoreState::kDrawScopeRegister) &&
        !pending_state_.bp_valid(GxCoreState::kDrawTagRegister) && pending_state_.position_indexed())
      consumer_.fuse_next_draw(kFuseMaxVertices);
    last_draw_triangles_ = triangles;
  }
  // Forward first: a Draw packet makes the PREVIOUS draw span-complete and
  // fires the observer, which must see the state that was current at that
  // previous draw (pending_state_), not this packet's.
  const bool ok = consumer_.submit_packet(packet);
  if (packet.kind == ar::RenderPacketKind::Draw) {
    // pending_state_ becomes live_state_: by replaying what changed since the
    // last draw, not by copying the whole state (1.4 KB) at every draw - that
    // copy was a large part of the GX worker's memmove (simulator sample).
    if (replay_overflow_) {
      pending_state_ = live_state_;
      replay_overflow_ = false;
    } else {
      pending_state_.forget_draw_tag(); // unless written again since
      for (const auto& state : since_draw_)
        pending_state_.apply(state);
    }
    live_state_.forget_draw_tag();
    since_draw_.clear();
  }
  return ok;
}

void GxCoreSink::flush_frame() { consumer_.flush_assembly(); }

namespace {
constexpr std::uint32_t kSinkStateMagic = 0x47585343u; // "GXSC"
static_assert(std::is_trivially_copyable_v<GxCoreState>);
static_assert(std::is_trivially_copyable_v<CachedVertexAttrs>);
} // namespace

std::vector<std::uint8_t> GxCoreSink::save_state() const {
  const std::uint32_t header[3] = {kSinkStateMagic,
                                   static_cast<std::uint32_t>(sizeof(GxCoreState)),
                                   static_cast<std::uint32_t>(sizeof(CachedVertexAttrs))};
  std::vector<std::uint8_t> out(sizeof header + sizeof(GxCoreState) +
                                sizeof(CachedVertexAttrs));
  std::uint8_t* at = out.data();
  std::memcpy(at, header, sizeof header);
  at += sizeof header;
  std::memcpy(at, &live_state_, sizeof(GxCoreState));
  at += sizeof(GxCoreState);
  std::memcpy(at, &cached_attrs_, sizeof(CachedVertexAttrs));
  return out;
}

bool GxCoreSink::load_state(const std::uint8_t* data, std::size_t size) {
  std::uint32_t header[3];
  if (data == nullptr ||
      size != sizeof header + sizeof(GxCoreState) + sizeof(CachedVertexAttrs))
    return false;
  std::memcpy(header, data, sizeof header);
  if (header[0] != kSinkStateMagic || header[1] != sizeof(GxCoreState) ||
      header[2] != sizeof(CachedVertexAttrs))
    return false;
  consumer_.flush_assembly();
  std::memcpy(&live_state_, data + sizeof header, sizeof(GxCoreState));
  std::memcpy(&cached_attrs_, data + sizeof header + sizeof(GxCoreState),
              sizeof(CachedVertexAttrs));
  // The saved version belongs to the run that saved it: in another run it may
  // be one this run's derived-state cache holds for other registers.
  live_state_.renew_version();
  pending_state_ = live_state_;
  since_draw_.clear();
  replay_overflow_ = false;
  scope_ = scope_left_ = scope_part_ = 0u;
  return true;
}

} // namespace gxruntime::gxcore
