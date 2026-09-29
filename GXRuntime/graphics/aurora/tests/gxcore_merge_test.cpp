// SPDX-License-Identifier: GPL-3.0-or-later
#include <gtest/gtest.h>
#include "gfx/gxcore_draw.hpp"

namespace {
using namespace aurora::gfx::gxcore;
DrawData first() {
  DrawData a{};
  a.pipeline = 7;
  a.textureBindGroup = 9;
  a.vertRange = {0, 3 * a.vertexStride};
  a.idxRange = {0, 6};
  a.indexCount = 3;
  a.uniformRange = {256, 128};
  a.pixelUniformRange = {512, 128};
  return a;
}
DrawData next(const DrawData& a) {
  auto b = a;
  b.vertRange.offset += a.vertRange.size;
  b.idxRange.offset += a.idxRange.size;
  return b;
}
TEST(GxCoreMerge, AdjacentListsKeepStateAndRebasedIndexBounds) {
  auto a = first();
  auto b = next(a);
  EXPECT_TRUE(can_merge_draws(a, b));
  b.pipeline++;
  EXPECT_FALSE(can_merge_draws(a, b));
  b = next(a);
  b.textureBindGroup++;
  EXPECT_FALSE(can_merge_draws(a, b));
  b = next(a);
  b.uniformRange.offset += 256;
  EXPECT_FALSE(can_merge_draws(a, b));
  b = next(a);
  b.pixelUniformRange.offset += 256;
  EXPECT_FALSE(can_merge_draws(a, b));
  b = next(a);
  b.interpUniformRange = {256, 128};
  EXPECT_FALSE(can_merge_draws(a, b));
  b = next(a);
  b.vertRange.offset += 4;
  EXPECT_FALSE(can_merge_draws(a, b));
  b = next(a);
  b.idxRange.offset += 2;
  EXPECT_FALSE(can_merge_draws(a, b));
  a.vertRange.size = 65533 * a.vertexStride;
  b = next(a);
  b.vertRange.size = 3 * a.vertexStride;
  EXPECT_FALSE(can_merge_draws(a, b));
  b.vertRange.size = 2 * a.vertexStride;
  EXPECT_TRUE(can_merge_draws(a, b));
}
TEST(GxCoreMerge, DepthPrepassesRemainInterleaved) {
  auto a = first();
  a.depthPipeline = 11;
  auto b = next(a);
  EXPECT_FALSE(can_merge_draws(a, b));
  a.depthPipeline = 0;
  EXPECT_FALSE(can_merge_draws(a, b));
  b.depthPipeline = 0;
  a.depthPipeline = 11;
  EXPECT_FALSE(can_merge_draws(a, b));
}
TEST(GxCoreMerge, PackedAndDecodedGeometryNeverMix) {
  auto a = first();
  a.packedVertices = true;
  a.vertexStride = 32;
  a.vertRange.size = 96;
  auto b = next(a);
  EXPECT_TRUE(can_merge_draws(a, b));
  b.packedVertices = false;
  EXPECT_FALSE(can_merge_draws(a, b));
  b = next(a);
  b.vertexStride = 64;
  EXPECT_FALSE(can_merge_draws(a, b));
}
} // namespace
