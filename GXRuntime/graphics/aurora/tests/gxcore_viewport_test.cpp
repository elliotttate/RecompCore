// SPDX-License-Identifier: GPL-3.0-or-later
#include <gtest/gtest.h>
#include "gfx/gxcore_draw.hpp"

TEST(GxCoreViewport, RetailFullscreenHasNoExtraOffset) {
  // Retail GXSetViewport(0, 0, 640, 480, 0, 1), encoded by GXTransform.c.
  const float raw[6] = {320.f, -240.f, 16777215.f, 662.f, 582.f, 16777215.f};
  const auto viewport = aurora::gfx::gxcore::retail_viewport(raw);
  EXPECT_FLOAT_EQ(viewport.left, 0.f);
  EXPECT_FLOAT_EQ(viewport.top, 0.f);
  EXPECT_FLOAT_EQ(viewport.width, 640.f);
  EXPECT_FLOAT_EQ(viewport.height, 480.f);
  EXPECT_FLOAT_EQ(viewport.znear, 0.f);
  EXPECT_FLOAT_EQ(viewport.zfar, 1.f);
  // At 3x the previous 340 decoder displaced every draw six target pixels,
  // including the postprocess quad which sampled the already-shifted scene.
  EXPECT_FLOAT_EQ(3.f * viewport.left, 0.f);
  EXPECT_FLOAT_EQ(3.f * viewport.top, 0.f);
}

TEST(GxCoreViewport, PreservesFractionalOriginAndDepth) {
  const float raw[6] = {160.f, -120.f, 8388607.5f, 519.25f, 470.5f, 12582911.25f};
  const auto viewport = aurora::gfx::gxcore::retail_viewport(raw);
  EXPECT_FLOAT_EQ(viewport.left, 17.25f);
  EXPECT_FLOAT_EQ(viewport.top, 8.5f);
  EXPECT_FLOAT_EQ(viewport.width, 320.f);
  EXPECT_FLOAT_EQ(viewport.height, 240.f);
  EXPECT_FLOAT_EQ(viewport.znear, 0.25f);
  EXPECT_FLOAT_EQ(viewport.zfar, 0.75f);
}
