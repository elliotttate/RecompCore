// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdint>
namespace gxruntime::gxcore {
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
  std::uint8_t attr = 0;     // CP array index for indexed fetch
  std::uint8_t vcd_type = 0; // 1 direct, 2 idx8, 3 idx16
  std::uint8_t format = 0;   // ComponentFormat / ColorFormat
  std::uint8_t count = 0;    // components (pos 2/3, tex 1/2, normal 3/9)
  std::uint8_t frac = 0;
  std::uint8_t out_slot = 0;      // color: 0/1; tex: 0-7
  std::uint32_t element_size = 0; // bytes of one element in the source array
};

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

// Shared format description for the CPU decoder and packed GPU vertex fetch.
bool derive_vertex_layout(std::uint32_t lo, std::uint32_t hi,
                          const std::uint32_t vat[3], WalkLayout &out);
} // namespace gxruntime::gxcore
