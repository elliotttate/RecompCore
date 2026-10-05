// Standalone checks for lib/gfx/frame_interp.cpp: draw matching across
// frames, plausibility rejection, the halfway blend, the cut verdict and the
// pacing of in-between frames.
#include "../lib/gfx/frame_interp.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace fi = aurora::gfx::frame_interp;
namespace gxc = gxruntime::gxcore;

static int g_failures = 0;
#define CHECK(cond)                                                                                                    \
  do {                                                                                                                 \
    if (!(cond)) {                                                                                                     \
      std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #cond);                                             \
      ++g_failures;                                                                                                    \
    }                                                                                                                  \
  } while (0)

static bool near(float a, float b) { return std::fabs(a - b) < 1e-4f; }

// A perspective draw at view-space (x, y, z), rotated by `yaw` degrees.
static gxc::VertexShaderConstants draw_at(float x, float y, float z, float yaw = 0.f) {
  gxc::VertexShaderConstants c{};
  const float r = yaw * 3.14159265f / 180.f;
  const float cs = std::cos(r), sn = std::sin(r);
  const float m[3][4] = {{cs, 0, sn, x}, {0, 1, 0, y}, {-sn, 0, cs, z}};
  std::memcpy(c.posnormalmatrix, m, sizeof(m));
  const float p[4][4] = {{1.5f, 0, 0, 0}, {0, 2.f, 0, 0}, {0, 0, -1.f, -10.f}, {0, 0, -1.f, 0}};
  std::memcpy(c.projection, p, sizeof(p));
  for (int i = 0; i + 2 < 64; i += 3) {
    c.transformmatrices[i][0] = c.transformmatrices[i + 1][1] = c.transformmatrices[i + 2][2] = 1.f;
  }
  return c;
}

// A world-space model at (x, y, z) seen by a camera at (cx, 0, cz) turned
// `yaw` degrees: the draw's matrix is the view matrix times the model's.
static gxc::VertexShaderConstants seen(float yaw, float cx, float cz, float x, float y, float z) {
  const float r = yaw * 3.14159265f / 180.f;
  const float cs = std::cos(r), sn = std::sin(r);
  const float dx = x - cx, dz = z - cz;
  return draw_at(cs * dx + sn * dz, y, -sn * dx + cs * dz, yaw);
}

struct Camera {
  float yaw, x, z;
};

// Rooms (a key each) that vote for the camera's motion, then a list of grass
// clumps (one key) at the given world x positions, the one at `liftedX` raised
// by `lift`. Each clump's blended constants, if it was blended, go to `out`
// (blend_draw's result is only valid until its next call).
struct Result {
  bool blended = false;
  gxc::VertexShaderConstants constants;
};
static void draw_scene(const Camera& c, const float* clumps, int count, Result* out, float liftedX = 0.f,
                       float lift = 0.f) {
  for (int room = 0; room < 6; ++room)
    fi::blend_draw(200 + room, 0, seen(c.yaw, c.x, c.z, room * 300.f - 750.f, -50.f, -900.f - room * 40.f));
  for (int i = 0; i < count; ++i) {
    const float y = clumps[i] == liftedX ? lift : 0.f;
    const auto* result = fi::blend_draw(500, 0, seen(c.yaw, c.x, c.z, clumps[i], y, -800.f));
    out[i].blended = result != nullptr;
    if (result != nullptr)
      out[i].constants = *result;
  }
}

// Whether a clump at world x was drawn halfway between its own two places
// (raised by yBefore and y).
static bool halfway(const Result& result, const Camera& before, const Camera& now, float x, float yBefore = 0.f,
                    float y = 0.f) {
  if (!result.blended)
    return false;
  const auto a = seen(before.yaw, before.x, before.z, x, yBefore, -800.f);
  const auto b = seen(now.yaw, now.x, now.z, x, y, -800.f);
  // Halfway along the camera's turn: the straight midpoint of the two places,
  // within the arc's bulge (about a unit at these distances and turns); a clump
  // paired with its neighbour is off by tens of units.
  float length = 0.f;
  for (int r = 0; r < 3; ++r) {
    const float expected = (a.posnormalmatrix[r][3] + b.posnormalmatrix[r][3]) / 2.f;
    length += expected * expected;
  }
  const float tolerance = 0.003f * std::sqrt(length) + 0.05f;
  for (int r = 0; r < 3; ++r) {
    const float expected = (a.posnormalmatrix[r][3] + b.posnormalmatrix[r][3]) / 2.f;
    if (std::fabs(result.constants.posnormalmatrix[r][3] - expected) > tolerance)
      return false;
  }
  return true;
}

// A draw submitted as gxcore_draw.cpp does: its inputs captured from the plan.
static const gxc::VertexShaderConstants* blend_plan(const gxc::DrawPlan& plan, const gxc::VertexShaderConstants& c) {
  static fi::DrawInput input;
  // Production captures the same constants it then blends. Most synthetic
  // plans above describe only geometry, so supply their transform here too.
  auto submitted = plan;
  submitted.constants = c;
  fi::capture_draw(submitted, input);
  return fi::blend_draw(input, c);
}

int main() {
  // Pacing at 120 Hz as well, for its checks (unset, 120 Hz is never lowered).
  setenv("DOL_AURORA_FRAME_INTERP_PACING", "1", 1);
  fi::set_enabled(true);
  const uint64_t kShip = 11, kTree = 22, kHud = 33;

  // Reused glyphs stay paired with their screen slot despite changing draw
  // order/counts. Nearby UI movement and heart pulses still interpolate.
  {
    gxc::DrawPlan hud;
    const uint8_t payload[96]{};
    hud.match_payload = payload;
    hud.match_payload_size = sizeof(payload);
    hud.match_primitive = 0x80;
    hud.match_direct_position = true;
    hud.vertex_count = 4;
    hud.tex_address = 0x00EA78A0;
    hud.pipeline.shader.num_tex_gens = 1;
    hud.vertices.resize(4 * gxc::kCompactVertexFloats);
    const auto quad = [&](float width, float height) {
      for (unsigned v = 0; v < 4; ++v) {
        auto* vertex = hud.vertices.data() + v * gxc::kCompactVertexFloats;
        vertex[gxc::kVertexPosOffset / sizeof(float)] = (v == 1 || v == 2) ? width : 0.f;
        vertex[gxc::kVertexPosOffset / sizeof(float) + 1] = v >= 2 ? height : 0.f;
        vertex[gxc::kVertexUv0Offset / sizeof(float)] = (v == 1 || v == 2) ? 1.f : 0.f;
        vertex[gxc::kVertexUv0Offset / sizeof(float) + 1] = v >= 2 ? 1.f : 0.f;
        for (unsigned c = 0; c < 4; ++c)
          vertex[gxc::kVertexColor0Offset / sizeof(float) + c] = 1.f;
      }
    };
    quad(20.f, 30.f);
    const auto at = [](float x, float y) {
      auto c = draw_at(x, y, 0.f);
      std::memset(c.projection, 0, sizeof(c.projection));
      c.projection[0][0] = 2.f / 640.f;
      c.projection[1][1] = -2.f / 480.f;
      c.projection[3][3] = 1.f;
      return c;
    };
    hud.constants = at(412.f, 412.f);
    // A stale particle tag must not distinguish otherwise identical UI.
    hud.draw_tag = 123;
    const auto key = fi::draw_key(hud);
    CHECK(key != 0);
    hud.draw_tag = 456;
    CHECK(fi::draw_key(hud) == key);
    hud.vertices[gxc::kVertexColor0Offset / sizeof(float) + 3] = .5f;
    CHECK(fi::draw_key(hud) == key); // fade alpha may animate
    hud.vertices[gxc::kVertexUv0Offset / sizeof(float)] = .25f;
    CHECK(fi::draw_key(hud) != key); // another atlas region is another glyph
    quad(20.f, 30.f);
    hud.vertices[gxc::kVertexColor0Offset / sizeof(float)] = 0.f;
    CHECK(fi::draw_key(hud) != key); // a shadow is distinct from its foreground
    quad(20.f, 30.f);
    fi::set_steps(3);

    blend_plan(hud, at(643.f, 443.f));
    blend_plan(hud, at(412.f, 412.f));
    fi::end_game_frame();
    for (const auto c : {at(413.f, 412.f), at(642.f, 443.f)}) {
      const auto* blended = blend_plan(hud, c);
      CHECK(blended != nullptr);
      if (blended != nullptr) {
        const float beforeX = c.posnormalmatrix[0][3] == 413.f ? 412.f : 643.f;
        for (int step = 0; step < 3; ++step)
          CHECK(near(fi::blended_step(step)->posnormalmatrix[0][3],
                     beforeX + (c.posnormalmatrix[0][3] - beforeX) * float(step + 1) / 4.f));
      }
    }
    CHECK(fi::frame_verdict());
    fi::end_game_frame();
    // The timer changes to another digit; next frame the shared glyph returns
    // without borrowing the only remaining copy, at the distant rupee counter.
    blend_plan(hud, at(642.f, 443.f));
    fi::end_game_frame();
    CHECK(blend_plan(hud, at(412.f, 412.f)) == nullptr);
    CHECK(fi::blended_positions() == nullptr);
    CHECK(blend_plan(hud, at(641.f, 443.f)) != nullptr);
    // A new copy cannot reuse the same counterpart already consumed above.
    CHECK(blend_plan(hud, at(640.f, 443.f)) == nullptr);
    fi::end_game_frame();

    // The last heart grows about its own centre, not a neighbouring heart.
    blend_plan(hud, at(170.f, 40.f));
    blend_plan(hud, at(195.f, 40.f));
    fi::end_game_frame();
    auto pulse = at(168.f, 37.f);
    pulse.posnormalmatrix[0][0] = pulse.posnormalmatrix[1][1] = 1.2f;
    CHECK(blend_plan(hud, pulse) != nullptr);
    for (int step = 0; step < 3; ++step) {
      const auto* b = fi::blended_step(step);
      CHECK(b != nullptr);
      if (b != nullptr) {
        CHECK(near(b->posnormalmatrix[0][0], 1.f + .2f * float(step + 1) / 4.f));
        CHECK(near(b->posnormalmatrix[0][3] + 10.f * b->posnormalmatrix[0][0], 180.f));
        CHECK(near(b->posnormalmatrix[1][3] + 15.f * b->posnormalmatrix[1][1], 55.f));
      }
    }
    CHECK(blend_plan(hud, at(195.f, 40.f)) == nullptr); // the neighbour stays still
    fi::end_game_frame();

    // A UI pulse can also resize its direct vertices with a fixed matrix.
    quad(20.f, 30.f);
    blend_plan(hud, at(170.f, 40.f));
    fi::end_game_frame();
    quad(22.f, 32.f);
    blend_plan(hud, at(170.f, 40.f));
    for (int step = 0; step < 3; ++step) {
      const auto* b = fi::blended_positions(step);
      CHECK(b != nullptr);
      if (b != nullptr) {
        CHECK(near(b[3], 20.f + 2.f * float(step + 1) / 4.f));
        CHECK(near(b[7], 30.f + 2.f * float(step + 1) / 4.f));
      }
    }
    CHECK(fi::frame_verdict());
    fi::end_game_frame();

    // Invalid bounds do not leave a stale interpolated vertex buffer behind.
    hud.constants = at(170.f, 40.f);
    hud.vertices[0] = INFINITY;
    fi::DrawInput invalid;
    fi::capture_draw(hud, invalid);
    CHECK(invalid.key == 0);
    CHECK(fi::blend_draw(invalid, hud.constants) == nullptr);
    CHECK(fi::blended_positions() == nullptr);
    quad(20.f, 30.f);
    // Orthographic depth-buffer geometry and perspective particles retain
    // their existing identities and interpolation paths.
    hud.pipeline.depth_test = 1;
    CHECK(fi::draw_key(hud) != key);
    hud.pipeline.depth_test = 0;
    hud.pipeline.depth_update = 1;
    CHECK(fi::draw_key(hud) != key);
    hud.pipeline.depth_update = 0;
    hud.constants.projection[3][2] = -1.f;
    CHECK(fi::draw_key(hud) != key);
    fi::set_steps(1);
    fi::end_game_frame();
    fi::end_game_frame();
  }

  // Frame 1: the ship, two trees, the HUD.
  CHECK(fi::blend_draw(kShip, 0, draw_at(0, 0, -500)) == nullptr); // no previous frame yet
  fi::blend_draw(kTree, 0, draw_at(-200, 0, -800));
  fi::blend_draw(kTree, 0, draw_at(300, 0, -900));
  fi::blend_draw(kHud, 0, draw_at(10, 10, -1));
  CHECK(!fi::frame_verdict());
  fi::end_game_frame();

  // Frame 2: the ship moved 20 units and turned 10 degrees; the trees are in
  // the other order; the HUD did not move.
  const auto* ship = fi::blend_draw(kShip, 0, draw_at(20, 0, -500, 10.f));
  CHECK(ship != nullptr);
  if (ship != nullptr) {
    CHECK(near(ship->posnormalmatrix[0][3], 10.f));
    CHECK(near(ship->posnormalmatrix[2][3], -500.f));
    CHECK(near(ship->posnormalmatrix[0][0], (1.f + std::cos(10.f * 3.14159265f / 180.f)) / 2.f));
  }
  const auto* treeA = fi::blend_draw(kTree, 0, draw_at(305, 0, -900));
  CHECK(treeA != nullptr);
  if (treeA != nullptr) // occurrence 0 was the tree at -200, 505 away: implausible, so the nearest
    CHECK(near(treeA->posnormalmatrix[0][3], 302.5f));
  const auto* treeB = fi::blend_draw(kTree, 0, draw_at(-195, 0, -800));
  CHECK(treeB != nullptr);
  if (treeB != nullptr)
    CHECK(near(treeB->posnormalmatrix[0][3], -197.5f));
  CHECK(fi::blend_draw(kHud, 0, draw_at(10, 10, -1)) == nullptr); // identical: its own constants
  CHECK(fi::frame_verdict());
  fi::end_game_frame();

  // Frame 3: a camera cut. Everything turns 120 degrees and jumps.
  const float cutX[] = {900, -700, 400, -300, 800, -600, 500, -400, 700, -500, 600, -800, 300, -900, 200, -100};
  fi::blend_draw(kShip, 0, draw_at(cutX[0], 0, 200, 130.f));
  for (int i = 0; i < 20; ++i)
    fi::blend_draw(kTree, 0, draw_at(cutX[i % 16], 50, 300, 120.f));
  CHECK(!fi::frame_verdict());
  fi::end_game_frame();

  // Frame 4: a scrolling texture wraps (0.98 -> 0.02) and keeps its value;
  // one that moves a little is blended.
  {
    auto a = draw_at(0, 0, -300);
    a.texmatrices[0][0] = a.texmatrices[1][1] = 1.f;
    a.texmatrices[0][3] = 0.98f;
    a.texmatrices[3][0] = a.texmatrices[4][1] = 1.f;
    a.texmatrices[3][3] = 0.10f;
    fi::blend_draw(99, 0, a);
    fi::end_game_frame();
    auto b = a;
    b.texmatrices[0][3] = 0.02f;
    b.texmatrices[3][3] = 0.20f;
    b.posnormalmatrix[0][3] = 4.f;
    const auto* blended = fi::blend_draw(99, 0, b);
    CHECK(blended != nullptr);
    if (blended != nullptr) {
      CHECK(near(blended->texmatrices[0][3], 0.02f));
      CHECK(near(blended->texmatrices[3][3], 0.15f));
      CHECK(near(blended->posnormalmatrix[0][3], 2.f));
    }
    fi::end_game_frame();
  }

  // A collapsed (hidden) object is not grown halfway.
  {
    auto shown = draw_at(0, 0, -300);
    fi::blend_draw(77, 0, shown);
    fi::end_game_frame();
    auto hidden = shown;
    std::memset(hidden.posnormalmatrix, 0, sizeof(float) * 12);
    CHECK(fi::blend_draw(77, 0, hidden) == nullptr);
    fi::end_game_frame();
  }

  // Grass clumps (copies of one draw) under a turning camera. The camera's
  // motion, voted for by the rooms, carries each clump to its own place in
  // the frame before, whatever the list order: a clump culled at the front
  // shifts every later one onto its neighbour, 40 units away (well inside the
  // plausibility bound), and the turn moves distant clumps farther than that.
  {
    const Camera a{0.f, 0.f, 0.f}, b{6.f, 10.f, -20.f}, c{12.f, 15.f, -45.f}, d{18.f, 20.f, -70.f},
        e{24.f, 25.f, -95.f}, f{30.f, 30.f, -120.f};
    Result r[6];
    const float first[] = {-100, -60, -20, 20, 60, 100};
    draw_scene(a, first, 6, r);
    fi::end_game_frame();

    // The first clump left the view; the others still pair with themselves.
    const float second[] = {-60, -20, 20, 60, 100};
    draw_scene(b, second, 5, r);
    for (int i = 0; i < 5; ++i)
      CHECK(halfway(r[i], a, b, second[i]));
    CHECK(fi::frame_verdict());
    fi::end_game_frame();

    // A clump comes into view at the end of the list: nothing stood where it
    // is, so it is drawn where the in-between camera sees it (blended from
    // where the camera's motion carries it from), not slid from a neighbour.
    const float third[] = {-60, -20, 20, 60, 100, 140};
    draw_scene(c, third, 6, r);
    CHECK(halfway(r[5], b, c, 140));
    fi::end_game_frame();
    draw_scene(d, third, 6, r);
    CHECK(halfway(r[5], c, d, 140));
    fi::end_game_frame();

    // A clump that starts to move at an actor's pace (a bush Link lifts) is
    // taken for one that stood at its new place for a frame, then pairs with
    // itself.
    draw_scene(e, third, 6, r, 60.f, 15.f);
    CHECK(halfway(r[3], d, e, 60, 15.f, 15.f) && halfway(r[4], d, e, 100));
    fi::end_game_frame();
    draw_scene(f, third, 6, r, 60.f, 30.f);
    CHECK(halfway(r[3], e, f, 60, 15.f, 30.f));
    fi::end_game_frame();
  }

  // A fast turn (50 degrees in one game frame, a flick of the mouse camera):
  // distant scenery moves far past the plausibility bounds, but the rooms'
  // shared motion is the camera's, so the frame is still interpolated, each
  // room is drawn exactly where the halfway camera sees it (no shrink from
  // blending two views far apart), and a character walking meanwhile is
  // halfway along its own path in that view.
  {
    fi::end_game_frame();
    fi::end_game_frame(); // no previous frame, no prediction
    const Camera a{0.f, 0.f, 0.f}, b{50.f, 0.f, 0.f}, mid{25.f, 0.f, 0.f};
    const auto frame = [](const Camera& c, float walker, Result* rooms, Result& link) {
      for (int room = 0; room < 12; ++room) {
        const auto* r = fi::blend_draw(700 + room, 0,
                                       seen(c.yaw, c.x, c.z, room * 400.f - 2200.f, -80.f, -2500.f - room * 90.f));
        rooms[room].blended = r != nullptr;
        if (r != nullptr)
          rooms[room].constants = *r;
      }
      const auto* l = fi::blend_draw(799, 0, seen(c.yaw, c.x, c.z, walker, 0.f, -300.f));
      link.blended = l != nullptr;
      if (l != nullptr)
        link.constants = *l;
    };
    Result rooms[12], link;
    frame(a, 0.f, rooms, link);
    fi::end_game_frame();
    frame(b, 20.f, rooms, link);
    CHECK(fi::frame_verdict());
    bool exact = true;
    for (int room = 0; room < 12; ++room) {
      const auto want = seen(mid.yaw, 0.f, 0.f, room * 400.f - 2200.f, -80.f, -2500.f - room * 90.f);
      exact = exact && rooms[room].blended;
      for (int r = 0; r < 3 && exact; ++r)
        for (int c = 0; c < 4; ++c)
          exact = exact && std::fabs(rooms[room].constants.posnormalmatrix[r][c] - want.posnormalmatrix[r][c]) <
                               (c == 3 ? 0.05f : 1e-4f);
    }
    CHECK(exact);
    const auto walking = seen(mid.yaw, 0.f, 0.f, 10.f, 0.f, -300.f);
    CHECK(link.blended && std::fabs(link.constants.posnormalmatrix[0][3] - walking.posnormalmatrix[0][3]) < 0.05f &&
          std::fabs(link.constants.posnormalmatrix[2][3] - walking.posnormalmatrix[2][3]) < 0.05f);
    fi::end_game_frame();
    // Most of the view new at once (a field of grass swung into view): still
    // a turn, not a cut.
    {
      frame(b, 20.f, rooms, link);
      for (int clump = 0; clump < 4; ++clump) // the model's copies elsewhere, far behind
        fi::blend_draw(900, 0, seen(b.yaw, b.x, b.z, 5000.f + clump * 30.f, 0.f, 4000.f));
      fi::end_game_frame();
      const Camera c{70.f, 0.f, 0.f};
      frame(c, 20.f, rooms, link);
      for (int clump = 0; clump < 40; ++clump)
        fi::blend_draw(900, 0, seen(c.yaw, c.x, c.z, -600.f + clump * 30.f, 0.f, -700.f));
      CHECK(fi::frame_verdict());
      fi::end_game_frame();
    }
    // A cut (the camera across the island) is still not interpolated.
    frame(Camera{140.f, 3000.f, -4000.f}, 20.f, rooms, link);
    CHECK(!fi::frame_verdict());
    fi::end_game_frame();
  }

  // A model's parts repeat the constants of the part before them, and the
  // caller says so: the in-between block is reused, and it is the one blending
  // them again gives. A draw with per-vertex matrix indices has the matrices it
  // reads blended and the others left as they are now.
  {
    fi::end_game_frame();
    const auto part = draw_at(0, 0, -400);
    auto indexed = draw_at(0, 0, -400);
    fi::blend_draw(7001, 0, part, false);
    fi::blend_draw(7002, 0, part, true);
    fi::blend_draw(7003, 0, part, true);
    fi::blend_draw(7004, 1ull << 3, indexed, false);
    fi::end_game_frame();

    const auto moved = draw_at(10, 0, -400, 5.f);
    const auto* first = fi::blend_draw(7001, 0, moved, false);
    CHECK(first != nullptr && !fi::last_blend_repeated());
    const gxc::VertexShaderConstants fresh = first != nullptr ? *first : gxc::VertexShaderConstants{};
    const auto* second = fi::blend_draw(7002, 0, moved, true);
    CHECK(second != nullptr && fi::last_blend_repeated());
    CHECK(second != nullptr && std::memcmp(second, &fresh, sizeof(fresh)) == 0);
    const auto* third = fi::blend_draw(7003, 0, moved, false); // not told: blended again, the same
    CHECK(third != nullptr && !fi::last_blend_repeated());
    CHECK(third != nullptr && std::memcmp(third, &fresh, sizeof(fresh)) == 0);
    CHECK(near(fresh.posnormalmatrix[0][3], 5.f));

    indexed.transformmatrices[3][3] = 20.f; // the matrix at row 3, which it reads
    indexed.transformmatrices[6][3] = 40.f; // the one at row 6, which it does not
    const auto* blendedIndexed = fi::blend_draw(7004, 1ull << 3, indexed, false);
    CHECK(blendedIndexed != nullptr);
    if (blendedIndexed != nullptr) {
      CHECK(near(blendedIndexed->transformmatrices[3][3], 10.f));
      CHECK(near(blendedIndexed->transformmatrices[6][3], 40.f));
    }
    fi::end_game_frame();
  }

  // Sailing: the camera follows the boat. The sea is strips drawn with the
  // view matrix whose texture coordinates follow the player, so each strip's
  // key is new every frame (no counterpart). In the in-between frame the sea
  // must still be seen from the in-between camera, like the islands and the
  // boat, or the water steps at 30 FPS under a boat that moves at 60.
  {
    fi::end_game_frame();
    fi::end_game_frame();
    const auto near_translation = [](const gxc::VertexShaderConstants& got, const gxc::VertexShaderConstants& a,
                                     const gxc::VertexShaderConstants& b) {
      float length = 0.f;
      for (int r = 0; r < 3; ++r) {
        const float expected = (a.posnormalmatrix[r][3] + b.posnormalmatrix[r][3]) / 2.f;
        length += expected * expected;
      }
      const float tolerance = 0.003f * std::sqrt(length) + 0.05f;
      for (int r = 0; r < 3; ++r) {
        const float expected = (a.posnormalmatrix[r][3] + b.posnormalmatrix[r][3]) / 2.f;
        if (std::fabs(got.posnormalmatrix[r][3] - expected) > tolerance)
          return false;
      }
      return true;
    };
    // The boat sails 30 units a frame along -z; the camera 600 units behind it
    // turns a degree a frame.
    const auto camera_at = [](int frame) { return Camera{float(frame), 0.f, 600.f - 30.f * frame}; };
    const auto boat_z = [](int frame) { return -30.f * frame; };
    uint64_t seaKey = 5000;
    const auto sail = [&](int frame, Result* sea, Result& boat, Result* rooms) {
      const Camera c = camera_at(frame);
      for (int room = 0; room < 8; ++room) {
        const auto* r = fi::blend_draw(8100 + room, 0,
                                       seen(c.yaw, c.x, c.z, room * 500.f - 2000.f, -60.f, -6000.f - room * 70.f));
        rooms[room].blended = r != nullptr;
        if (r != nullptr)
          rooms[room].constants = *r;
      }
      const auto* b = fi::blend_draw(8200, 0, seen(c.yaw, c.x, c.z, 0.f, 20.f, boat_z(frame)));
      boat.blended = b != nullptr;
      if (b != nullptr)
        boat.constants = *b;
      for (int strip = 0; strip < 8; ++strip) {
        const auto* s = fi::blend_draw(++seaKey, 0, seen(c.yaw, c.x, c.z, 0.f, 0.f, 0.f), strip > 0);
        sea[strip].blended = s != nullptr;
        if (s != nullptr)
          sea[strip].constants = *s;
      }
    };
    Result sea[8], boat, rooms[8];
    sail(0, sea, boat, rooms);
    fi::end_game_frame();
    sail(1, sea, boat, rooms);
    fi::end_game_frame();
    sail(2, sea, boat, rooms);
    CHECK(fi::frame_verdict());
    const auto viewBefore = seen(camera_at(1).yaw, camera_at(1).x, camera_at(1).z, 0.f, 0.f, 0.f);
    const auto viewNow = seen(camera_at(2).yaw, camera_at(2).x, camera_at(2).z, 0.f, 0.f, 0.f);
    bool seaHalfway = true;
    for (int strip = 0; strip < 8; ++strip)
      seaHalfway = seaHalfway && sea[strip].blended && near_translation(sea[strip].constants, viewBefore, viewNow);
    CHECK(seaHalfway);
    // Every strip the same: the in-between block of the first is reused.
    CHECK(sea[7].blended && std::memcmp(&sea[7].constants, &sea[0].constants, sizeof(sea[0].constants)) == 0);
    const auto boatBefore = seen(camera_at(1).yaw, camera_at(1).x, camera_at(1).z, 0.f, 20.f, boat_z(1));
    const auto boatNow = seen(camera_at(2).yaw, camera_at(2).x, camera_at(2).z, 0.f, 20.f, boat_z(2));
    CHECK(boat.blended && near_translation(boat.constants, boatBefore, boatNow));
    fi::end_game_frame();

    // Turning at open sea, in the game's order: a few draws with the view
    // matrix (effects, some of them copies), the sky around the camera, then
    // the boat's many parts, then distant copies (rocks, clouds), the wave
    // crests (copies drawn with the view matrix) and the far islands last.
    // The boat's parts, which hardly move in the camera's view, outnumber
    // everything drawn before the scenery, but the camera's motion must still
    // be the one that carries the distant copies and the crests.
    fi::end_game_frame();
    fi::end_game_frame();
    const auto turning = [](int frame) { return Camera{2.f * frame, 0.f, 600.f - 30.f * frame}; };
    // The camera follows the boat as it turns: the boat stays in front of it,
    // bobbing a little.
    const auto boat_part = [](int part, int frame) {
      return draw_at(part * 2.f, 20.f + part + 0.5f * (frame % 2), -600.f - part);
    };
    Result far[5], crests[10], boatParts[20];
    const auto open_sea = [&](int frame, bool check) {
      const Camera c = turning(frame);
      fi::blend_draw(9001, 0, seen(c.yaw, c.x, c.z, 0.f, 0.f, 0.f));
      fi::blend_draw(9002, 0, seen(c.yaw, c.x, c.z, 0.f, 0.f, 0.f), true);
      for (int effect = 0; effect < 3; ++effect)
        for (int copy = 0; copy < 12; ++copy)
          fi::blend_draw(9010 + effect, 0, seen(c.yaw, c.x, c.z, 0.f, 0.f, 0.f), true);
      for (int sky = 0; sky < 8; ++sky)
        fi::blend_draw(9020 + sky, 0, seen(c.yaw, c.x, c.z, c.x + sky * 3.f, 40.f * sky, c.z - 5.f * sky));
      for (int part = 0; part < 20; ++part) {
        const auto* p = fi::blend_draw(9100 + part, 0, boat_part(part, frame));
        boatParts[part].blended = p != nullptr;
        if (p != nullptr)
          boatParts[part].constants = *p;
      }
      for (int copy = 0; copy < 5; ++copy) {
        const auto* p = fi::blend_draw(9200, 0, seen(c.yaw, c.x, c.z, -4000.f + copy * 2000.f, 300.f, -30000.f));
        far[copy].blended = p != nullptr;
        if (p != nullptr)
          far[copy].constants = *p;
      }
      for (int copy = 0; copy < 10; ++copy) {
        const auto* p = fi::blend_draw(9300, 0, seen(c.yaw, c.x, c.z, 0.f, 0.f, 0.f), copy > 0);
        crests[copy].blended = p != nullptr;
        if (p != nullptr)
          crests[copy].constants = *p;
      }
      for (int island = 0; island < 6; ++island)
        fi::blend_draw(9400 + island, 0,
                       seen(c.yaw, c.x, c.z, island * 20000.f - 50000.f, -100.f, -80000.f - island * 5000.f));
      if (!check)
        return;
      const Camera b = turning(frame - 1);
      bool farHalfway = true;
      for (int copy = 0; copy < 5; ++copy) {
        const float x = -4000.f + copy * 2000.f;
        farHalfway = farHalfway && far[copy].blended &&
                     near_translation(far[copy].constants, seen(b.yaw, b.x, b.z, x, 300.f, -30000.f),
                                      seen(c.yaw, c.x, c.z, x, 300.f, -30000.f));
      }
      CHECK(farHalfway);
      bool crestsHalfway = true;
      for (int copy = 0; copy < 10; ++copy)
        crestsHalfway = crestsHalfway && crests[copy].blended &&
                        near_translation(crests[copy].constants, seen(b.yaw, b.x, b.z, 0.f, 0.f, 0.f),
                                         seen(c.yaw, c.x, c.z, 0.f, 0.f, 0.f));
      CHECK(crestsHalfway);
      bool boatHalfway = true;
      for (int part = 0; part < 20; ++part)
        boatHalfway = boatHalfway && boatParts[part].blended &&
                      near_translation(boatParts[part].constants, boat_part(part, frame - 1), boat_part(part, frame));
      CHECK(boatHalfway);
    };
    open_sea(0, false);
    fi::end_game_frame();
    open_sea(1, false);
    fi::end_game_frame();
    open_sea(2, true);
    CHECK(fi::frame_verdict());
    fi::end_game_frame();
    open_sea(3, true);
    fi::end_game_frame();
  }

  // Under full sail (100 units a frame, the camera following): the hull is
  // skinned and each of its parts is drawn twice, into the shadow map (an
  // orthographic projection) and into the scene, and some parts twice into
  // the scene. None of these draws is a copy come into view: each must be
  // drawn halfway, or the in-between frame shows a second hull ahead of it.
  {
    fi::end_game_frame();
    fi::end_game_frame();
    const auto camera_at = [](int frame) { return Camera{1.f * frame, 0.f, 700.f - 100.f * frame}; };
    // A skinned draw of a model at world (x, y, z): its matrix in XF row 0.
    const auto skinned = [](const Camera& c, float x, float y, float z, bool shadowMap) {
      auto constants = seen(c.yaw, c.x, c.z, 0.f, 0.f, 0.f);
      const auto model = seen(c.yaw, c.x, c.z, x, y, z);
      std::memcpy(constants.transformmatrices[0], model.posnormalmatrix, sizeof(float) * 12);
      if (shadowMap) {
        std::memset(constants.projection, 0, sizeof(constants.projection));
        constants.projection[0][0] = constants.projection[1][1] = 0.002f;
        constants.projection[2][2] = -0.001f;
        constants.projection[3][3] = 1.f;
      }
      return constants;
    };
    const auto hull_z = [](int frame) { return 100.f - 100.f * frame; };
    const auto halfway_row0 = [](const gxc::VertexShaderConstants* got, const gxc::VertexShaderConstants& a,
                                 const gxc::VertexShaderConstants& b) {
      if (got == nullptr)
        return false;
      for (int r = 0; r < 3; ++r) {
        const float expected = (a.transformmatrices[r][3] + b.transformmatrices[r][3]) / 2.f;
        if (std::fabs(got->transformmatrices[r][3] - expected) > 0.5f)
          return false;
      }
      return true;
    };
    bool hullHalfway = true, twiceHalfway = true;
    for (int frame = 0; frame < 4; ++frame) {
      const Camera c = camera_at(frame), b = camera_at(frame - 1);
      for (int room = 0; room < 8; ++room)
        fi::blend_draw(9500 + room, 0, seen(c.yaw, c.x, c.z, room * 700.f - 2500.f, -60.f, -5000.f - room * 90.f));
      for (int part = 0; part < 3; ++part)
        fi::blend_draw(9600 + part, 1, skinned(c, part * 30.f, 0.f, hull_z(frame), true));
      for (int part = 0; part < 3; ++part) {
        const auto* got = fi::blend_draw(9600 + part, 1, skinned(c, part * 30.f, 0.f, hull_z(frame), false));
        if (frame >= 2)
          hullHalfway = hullHalfway && halfway_row0(got, skinned(b, part * 30.f, 0.f, hull_z(frame - 1), false),
                                                    skinned(c, part * 30.f, 0.f, hull_z(frame), false));
      }
      for (int copy = 0; copy < 2; ++copy) {
        const auto* got = fi::blend_draw(9700, 1, skinned(c, 0.f, 40.f, hull_z(frame), false), copy == 1);
        if (frame >= 2)
          twiceHalfway = twiceHalfway && halfway_row0(got, skinned(b, 0.f, 40.f, hull_z(frame - 1), false),
                                                      skinned(c, 0.f, 40.f, hull_z(frame), false));
      }
      if (frame >= 2)
        CHECK(fi::frame_verdict());
      fi::end_game_frame();
    }
    CHECK(hullHalfway);
    CHECK(twiceHalfway);
  }

  // The boat's hull is skinned on the CPU: the game rewrites its vertices in
  // world space every frame and draws them with the view matrix, so its key
  // and matrices match the frame before while its vertices moved 100 units.
  // The in-between frame draws this frame's vertices: with the view blended
  // alone, the hull would be where it is now, half a frame ahead of the rest
  // of the boat. Its vertices must land halfway between the two frames.
  {
    fi::end_game_frame();
    fi::end_game_frame();
    const uint8_t payload[] = {0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5};
    const float local[6][3] = {{-40, 0, -90}, {40, 0, -90}, {-45, 30, 0}, {45, 30, 0}, {-30, 10, 80}, {30, 10, 80}};
    const auto camera_at = [](int frame) { return Camera{0.f, 0.f, 700.f - 100.f * frame}; };
    // The boat turns 3 degrees a frame about its middle, 100 units ahead.
    const auto world = [&](int frame, int v, float out[3]) {
      const float a = 3.f * frame * 3.14159265f / 180.f, cs = std::cos(a), sn = std::sin(a);
      out[0] = cs * local[v][0] + sn * local[v][2];
      out[1] = local[v][1];
      out[2] = -sn * local[v][0] + cs * local[v][2] - 100.f * frame;
    };
    const auto seen_at = [](const float m[][4], const float p[3], float out[3]) {
      for (int r = 0; r < 3; ++r)
        out[r] = m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2] + m[r][3];
    };
    float drawn[3][6][3]; // each real frame's view-space vertices
    bool hullHalfway = true;
    for (int frame = 0; frame < 3; ++frame) {
      const Camera c = camera_at(frame);
      for (int room = 0; room < 8; ++room)
        fi::blend_draw(9800 + room, 0, seen(c.yaw, c.x, c.z, room * 700.f - 2500.f, -60.f, -5000.f - room * 90.f));
      gxc::DrawPlan plan;
      plan.match_payload = payload;
      plan.match_payload_size = sizeof(payload);
      plan.match_primitive = 0x98;
      plan.match_vertex_stride = 2;
      plan.vertex_count = 6;
      plan.vertices.assign(6 * gxc::kCompactVertexFloats, 0.f);
      for (int v = 0; v < 6; ++v)
        world(frame, v, &plan.vertices[v * gxc::kCompactVertexFloats]);
      const auto view = seen(c.yaw, c.x, c.z, 0.f, 0.f, 0.f);
      for (int v = 0; v < 6; ++v)
        seen_at(view.posnormalmatrix, &plan.vertices[v * gxc::kCompactVertexFloats], drawn[frame][v]);
      const auto* got = blend_plan(plan, view);
      if (frame == 2) {
        const float* vertices = fi::blended_positions();
        hullHalfway = vertices != nullptr;
        for (int v = 0; hullHalfway && v < 6; ++v) {
          float at[3];
          seen_at(got != nullptr ? got->posnormalmatrix : view.posnormalmatrix, vertices + v * 3, at);
          for (int r = 0; r < 3; ++r)
            hullHalfway = hullHalfway && std::fabs(at[r] - (drawn[1][v][r] + drawn[2][v][r]) / 2.f) < 1.f;
        }
        CHECK(fi::frame_verdict());
      }
      fi::end_game_frame();
    }
    CHECK(hullHalfway);
  }

  // Adjacent CPU-deformed ocean strips share a vertex, but have different
  // wave heights elsewhere. Every 120-Hz step must keep that shared edge
  // together, including at large world coordinates and with camera motion.
  for (bool movingCamera : {false, true}) {
    fi::end_game_frame();
    fi::end_game_frame();
    fi::set_steps(3);
    uint8_t payload[2][40]{};
    float shared[3][2][3]{};
    for (int frame = 0; frame < 3; ++frame) {
      const Camera c{0.f, -200000.f + (movingCamera ? 20.f * frame : 0.f), 2000.f};
      const auto view = seen(c.yaw, c.x, c.z, 0.f, 0.f, 0.f);
      for (int room = 0; room < 8; ++room)
        fi::blend_draw(18000 + room, 0, seen(c.yaw, c.x, c.z, -200000.f + room * 500.f, 0.f, -5000.f));
      for (int strip = 0; strip < 2; ++strip) {
        gxc::DrawPlan plan;
        for (int v = 0; v < 4; ++v) {
          payload[strip][v * 10 + 1] = static_cast<uint8_t>(strip * 2 + v);
          const float uv[2] = {float(frame), float(v)};
          std::memcpy(payload[strip] + v * 10 + 2, uv, sizeof(uv));
        }
        plan.match_payload = payload[strip];
        plan.match_payload_size = sizeof(payload[strip]);
        plan.match_primitive = 0x98;
        plan.match_vertex_stride = 10;
        plan.match_position_size = 2;
        plan.match_direct_texcoord_mask = (1u << 0) | (1u << 2);
        plan.vertex_count = 4;
        // A format with tex2 decodes to the full layout.
        plan.vertex_floats = gxc::kFullVertexFloats;
        plan.vertices.assign(4 * gxc::kFullVertexFloats, 0.f);
        for (int v = 0; v < 4; ++v) {
          const int index = strip * 2 + v;
          float* p = plan.vertices.data() + v * gxc::kFullVertexFloats;
          p[0] = -200000.f + 800.f * (index / 2) + (movingCamera ? 20.f * frame : 0.f);
          p[1] = float((index * index + 3) * frame); // non-rigid deformation
          p[2] = -800.f * (index % 2);
          float* uv = p + gxc::kVertexUv0Offset / sizeof(float);
          uv[0] = p[0] * .0005f;
          uv[1] = p[2] * .0005f;
          float* uv2 = p + gxc::vertex_uv_offset(2) / sizeof(float);
          uv2[0] = .02f * frame; // another direct channel, also animated
          uv2[1] = .1f * index;
        }
        const auto* got = blend_plan(plan, view);
        if (frame == 2) {
          for (int step = 0; step < 3; ++step) {
            const float* vertices = fi::blended_positions(step);
            const float* texcoords = fi::blended_texcoords(step);
            CHECK(vertices != nullptr);
            CHECK(texcoords != nullptr);
            CHECK(fi::blended_texcoord_mask() == ((1u << 0) | (1u << 2)));
            if (vertices == nullptr)
              continue;
            const float t = float(step + 1) / 4.f;
            for (int v = 0; v < 4; ++v) {
              const int index = strip * 2 + v;
              CHECK(std::fabs(vertices[v * 3 + 1] - (index * index + 3) * (1.f + t)) < .01f);
              CHECK(std::fabs(vertices[v * 3] - (-200000.f + 800.f * (index / 2) +
                    (movingCamera ? 20.f * (1.f + t) : 0.f))) < .05f);
              if (texcoords != nullptr) {
                // Both attributes describe the same intermediate world point.
                CHECK(std::fabs(texcoords[v * 4] - vertices[v * 3] * .0005f) < 2e-5f);
                CHECK(std::fabs(texcoords[v * 4 + 1] - vertices[v * 3 + 2] * .0005f) < 2e-5f);
                CHECK(std::fabs(texcoords[v * 4 + 2] - .02f * (1.f + t)) < 1e-6f);
                CHECK(std::fabs(texcoords[v * 4 + 3] - .1f * index) < 1e-6f);
              }
            }
            const float* p = vertices + (strip == 0 ? 2 : 0) * 3;
            const float (*m)[4] = got != nullptr ? fi::blended_step(step)->posnormalmatrix : view.posnormalmatrix;
            for (int r = 0; r < 3; ++r)
              shared[step][strip][r] = m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2] + m[r][3];
          }
        }
      }
      if (frame == 2)
        for (int step = 0; step < 3; ++step)
          for (int r = 0; r < 3; ++r)
            CHECK(std::fabs(shared[step][0][r] - shared[step][1][r]) < .01f);
      fi::end_game_frame();
    }
    fi::set_steps(1);
  }

  // Direct UV animation also blends when the mesh itself stays still. An
  // atlas switch/wrap keeps its current channel instead of sweeping across it.
  for (bool wraps : {false, true}) {
    fi::end_game_frame();
    fi::end_game_frame();
    fi::set_steps(3);
    uint8_t payload[8] = {0, 0, 0, 1, 0, 2, 0, 3};
    for (int frame = 0; frame < 2; ++frame) {
      gxc::DrawPlan plan;
      plan.match_payload = payload;
      plan.match_payload_size = sizeof(payload);
      plan.match_primitive = 0x98;
      plan.match_vertex_stride = 2;
      plan.match_position_size = 2;
      plan.match_direct_texcoord_mask = (1u << 0) | (1u << 2);
      plan.vertex_count = 4;
      // A format with tex2 decodes to the full layout.
      plan.vertex_floats = gxc::kFullVertexFloats;
      plan.vertices.assign(4 * gxc::kFullVertexFloats, 0.f);
      for (int v = 0; v < 4; ++v) {
        float* p = plan.vertices.data() + v * gxc::kFullVertexFloats;
        p[0] = v * 20.f;
        p[2] = -400.f;
        float* uv = p + gxc::kVertexUv0Offset / sizeof(float);
        uv[0] = wraps ? (frame == 0 ? .98f : .02f) : .1f + .04f * frame;
        uv[1] = .2f;
        float* uv2 = p + gxc::vertex_uv_offset(2) / sizeof(float);
        uv2[0] = .3f + .02f * frame;
        uv2[1] = .4f;
      }
      blend_plan(plan, draw_at(0, 0, 0));
      if (frame == 1) {
        CHECK(fi::blended_positions() == nullptr);
        CHECK(fi::blended_texcoord_mask() == ((1u << 0) | (1u << 2)));
        CHECK(fi::frame_verdict());
        for (int step = 0; step < 3; ++step) {
          const float* uv = fi::blended_texcoords(step);
          CHECK(uv != nullptr);
          if (uv != nullptr)
            for (int v = 0; v < 4; ++v) {
              const float t = float(step + 1) / 4.f;
              CHECK(std::fabs(uv[v * 4] - (wraps ? .02f : .1f + .04f * t)) < 1e-6f);
              CHECK(std::fabs(uv[v * 4 + 2] - (.3f + .02f * t)) < 1e-6f);
            }
        }
      }
      fi::end_game_frame();
    }
    fi::set_steps(1);
  }

  // A bone of Link's sword arm turns 100 degrees in one game frame: a draw
  // with a key of its own is still blended, halfway along the turn (50
  // degrees, its size kept, not a straight blend shrunk to 64 percent), with
  // its scale blended straight.
  {
    fi::end_game_frame();
    fi::end_game_frame();
    const auto scaled = [](gxc::VertexShaderConstants c, float k) {
      for (int r = 0; r < 3; ++r)
        for (int col = 0; col < 3; ++col)
          c.posnormalmatrix[r][col] *= k;
      return c;
    };
    fi::blend_draw(8001, 0, draw_at(10, 0, -400, 0.f));
    fi::blend_draw(8002, 0, scaled(draw_at(-10, 0, -400, 0.f), 1.f));
    fi::blend_draw(8003, 0, draw_at(30, 0, -400, 0.f));
    fi::blend_draw(8003, 0, draw_at(60, 0, -400, 0.f));
    fi::end_game_frame();
    const auto* bone = fi::blend_draw(8001, 0, draw_at(10, 0, -400, 100.f));
    CHECK(bone != nullptr);
    if (bone != nullptr) {
      const float c50 = std::cos(50.f * 3.14159265f / 180.f), s50 = std::sin(50.f * 3.14159265f / 180.f);
      CHECK(std::fabs(bone->posnormalmatrix[0][0] - c50) < 1e-3f && std::fabs(bone->posnormalmatrix[0][2] - s50) < 1e-3f);
      CHECK(std::fabs(bone->posnormalmatrix[2][0] + s50) < 1e-3f && std::fabs(bone->posnormalmatrix[1][1] - 1.f) < 1e-3f);
      CHECK(near(bone->posnormalmatrix[0][3], 10.f) && near(bone->posnormalmatrix[2][3], -400.f));
    }
    const auto* grown = fi::blend_draw(8002, 0, scaled(draw_at(-10, 0, -400, 90.f), 1.3f));
    CHECK(grown != nullptr);
    if (grown != nullptr) {
      const float c45 = std::cos(45.f * 3.14159265f / 180.f);
      CHECK(std::fabs(grown->posnormalmatrix[0][0] - 1.15f * c45) < 1e-3f);
      CHECK(std::fabs(grown->posnormalmatrix[1][1] - 1.15f) < 1e-3f);
    }
    // Two copies that changed places in the list, each turning 100 degrees
    // where it stands: each pairs with the one nearest it, itself, not with
    // the one at its place in the list.
    const auto* swappedA = fi::blend_draw(8003, 0, draw_at(60, 0, -400, 100.f));
    CHECK(swappedA != nullptr && near(swappedA->posnormalmatrix[0][3], 60.f));
    const auto* swappedB = fi::blend_draw(8003, 0, draw_at(30, 0, -400, 100.f));
    CHECK(swappedB != nullptr && near(swappedB->posnormalmatrix[0][3], 30.f));
    // A small turn is still a straight blend (the same to within float).
    fi::end_game_frame();
    fi::blend_draw(8001, 0, draw_at(10, 0, -400, 110.f));
    fi::end_game_frame();
    const auto* small = fi::blend_draw(8001, 0, draw_at(10, 0, -400, 106.f));
    CHECK(small != nullptr && std::fabs(small->posnormalmatrix[0][0] - (std::cos(1.91986f) + std::cos(1.85005f)) / 2.f) < 1e-5f);
  }

  // Particles: the game sends each one's corners as positions, and the host
  // tags each draw with its particle and age. Each is drawn halfway between
  // its own two places, even when the list's order changes; a new particle at
  // a dead one's address, or a jump no particle makes, keeps its own.
  {
    static const uint8_t payload[48] = {1};
    const auto particle = [](uint32_t tag, uint32_t age, float x, float y) {
      gxc::DrawPlan plan;
      plan.match_payload = payload;
      plan.match_payload_size = sizeof(payload);
      plan.match_primitive = 0x80;
      plan.match_direct_position = true;
      plan.vertex_count = 4;
      plan.draw_tag = tag;
      plan.draw_tag_age = age;
      plan.tex_address = 0x1234;
      plan.vertices.assign(4 * gxc::kCompactVertexFloats, 0.f);
      const float corners[4][2] = {{-5, 5}, {5, 5}, {5, -5}, {-5, -5}};
      for (int i = 0; i < 4; ++i) {
        float* v = plan.vertices.data() + i * gxc::kCompactVertexFloats + gxc::kVertexPosOffset / sizeof(float);
        v[0] = x + corners[i][0];
        v[1] = y + corners[i][1];
        v[2] = -400.f;
      }
      return plan;
    };
    // A billboard: view-space corners and the identity matrix.
    const auto identity = draw_at(0, 0, 0);
    const auto submit = [&](const gxc::DrawPlan& plan) {
      blend_plan(plan, identity);
      return fi::blended_positions();
    };
    fi::end_game_frame();
    fi::end_game_frame();
    submit(particle(100, 3, 0, 0));
    submit(particle(101, 7, 50, 0));
    submit(particle(102, 9, -50, 0));
    submit(particle(103, 2, 0, 80));
    fi::end_game_frame();
    // The list's order changed (a new particle first); each still pairs with
    // itself.
    CHECK(submit(particle(104, 0, 200, 0)) == nullptr); // new
    const float* second = submit(particle(101, 8, 70, 10));
    CHECK(second != nullptr && near(second[0], 60.f - 5.f) && near(second[1], 5.f + 5.f) && near(second[2], -400.f));
    const float* first = submit(particle(100, 4, 10, 0));
    CHECK(first != nullptr && near(first[0], 5.f - 5.f) && near(first[1], 5.f));
    CHECK(submit(particle(102, 0, -50, 20)) == nullptr);   // a new particle at a dead one's address
    CHECK(submit(particle(103, 3, 900, 80)) == nullptr);   // too far for a frame
    // Untagged, one of its shape (the boat's shadow on the sea's triangles
    // under it): not the same points, so drawn as it is.
    fi::end_game_frame();
    auto trail = particle(0, 0, 0, 0);
    trail.match_primitive = 0x98;
    blend_plan(trail, identity);
    fi::end_game_frame();
    auto moved = particle(0, 0, 8, 0);
    moved.match_primitive = 0x98;
    blend_plan(moved, identity);
    const float* halfway = fi::blended_positions();
    CHECK(halfway == nullptr);

    // The camera turns 12 degrees past a spark that hangs still: its corners,
    // sent in the camera's view, change every frame. It is drawn where the
    // halfway camera sees it, as the rooms around it are.
    const auto frame = [&](float yaw) {
      for (int room = 0; room < 8; ++room)
        fi::blend_draw(900 + room, 0, seen(yaw, 0, 0, room * 300.f - 1000.f, -50.f, -900.f - room * 40.f));
      const auto at = seen(yaw, 0, 0, 100.f, 20.f, -600.f);
      auto spark = particle(300, 0, at.posnormalmatrix[0][3], at.posnormalmatrix[1][3]);
      for (int i = 0; i < 4; ++i)
        spark.vertices[i * gxc::kCompactVertexFloats + 2] = at.posnormalmatrix[2][3];
      return spark;
    };
    fi::end_game_frame();
    fi::end_game_frame();
    submit(frame(0.f));
    fi::end_game_frame();
    auto spark = frame(12.f);
    spark.draw_tag_age = 1;
    const float* hung = submit(spark);
    const auto mid = seen(6.f, 0, 0, 100.f, 20.f, -600.f);
    CHECK(hung != nullptr && std::fabs(hung[0] - (mid.posnormalmatrix[0][3] - 5.f)) < 0.5f &&
          std::fabs(hung[2] - mid.posnormalmatrix[2][3]) < 0.5f);
  }

  // The boat's real shadow at full sail. Its volume is one box for every real
  // shadow (copies of a key), carried along by the boat 60 units a frame, and
  // the camera follows it: the copies keep their place in the view and pair
  // with themselves. Its receiving triangles, in the shadow's own texture,
  // pair with the frame before's though their count changed; their positions
  // are the next frame's (not the same points).
  {
    fi::end_game_frame();
    fi::end_game_frame();
    static const uint8_t payload[48] = {2};
    const auto receiver = [](uint32_t triangles, const gxc::VertexShaderConstants& view) {
      gxc::DrawPlan plan;
      plan.constants = view;
      plan.match_payload = payload;
      plan.match_payload_size = triangles * 36u;
      plan.match_primitive = 0x90;
      plan.match_direct_position = true;
      plan.vertex_count = triangles * 3u;
      plan.tex_address = 0x5678;
      plan.vertices.assign(plan.vertex_count * gxc::kCompactVertexFloats, 0.f);
      return plan;
    };
    const char* outcomes[8] = {};
    const auto frame = [&](float cameraX, uint32_t triangles, Result* volume, bool& receiverBlended) {
      for (int room = 0; room < 8; ++room)
        fi::blend_draw(960 + room, 0, seen(0.f, cameraX, 0, room * 300.f - 1000.f, -50.f, -900.f - room * 40.f));
      for (int copy = 0; copy < 8; ++copy) {
        // Bobbing a little with the boat.
        const auto* r = fi::blend_draw(950, 0, seen(0.f, cameraX, 0, cameraX + copy * 3.f, cameraX * 0.01f - 20.f, -700.f));
        outcomes[copy] = fi::last_outcome();
        volume[copy].blended = r != nullptr;
        if (r != nullptr)
          volume[copy].constants = *r;
      }
      const auto plan = receiver(triangles, seen(0.f, cameraX, 0, 0.f, 0.f, 0.f));
      receiverBlended = blend_plan(plan, plan.constants) != nullptr;
      CHECK(fi::blended_positions() == nullptr);
    };
    Result volume[8];
    bool receiverBlended = false;
    frame(0.f, 18, volume, receiverBlended);
    fi::end_game_frame();
    frame(60.f, 18, volume, receiverBlended);
    fi::end_game_frame();
    frame(120.f, 16, volume, receiverBlended);
    CHECK(fi::frame_verdict());
    CHECK(receiverBlended);
    for (int copy = 0; copy < 8; ++copy) {
      CHECK(volume[copy].blended && std::fabs(volume[copy].constants.posnormalmatrix[0][3] - copy * 3.f) < 0.5f &&
            std::fabs(volume[copy].constants.posnormalmatrix[1][3] - (0.9f - 20.f)) < 0.05f);
      if (!volume[copy].blended)
        std::fprintf(stderr, "copy %d: %s\n", copy, outcomes[copy]);
    }
  }

  // A broken pot: its shards are one model drawn five times at random sizes,
  // each flying off and tumbling 70 degrees a frame. Each pairs with itself
  // (the same place in the list, the same size) and is blended halfway along
  // its turn.
  {
    fi::end_game_frame();
    fi::end_game_frame();
    const float sizes[5] = {1.f, 0.575f, 1.389f, 1.298f, 0.775f};
    const auto shard = [&](int i, int frame) {
      auto c = draw_at(20.f * i + 3.f * frame, 100.f + 5.f * frame, -480.f, 70.f * frame + 13.f * i);
      for (int r = 0; r < 3; ++r)
        for (int col = 0; col < 3; ++col)
          c.posnormalmatrix[r][col] *= sizes[i];
      return c;
    };
    const auto rooms = [](int) {
      for (int room = 0; room < 6; ++room)
        fi::blend_draw(980 + room, 0, draw_at(room * 300.f - 750.f, -50.f, -900.f - room * 40.f));
    };
    rooms(0);
    for (int i = 0; i < 5; ++i)
      fi::blend_draw(990, 0, shard(i, 0));
    fi::end_game_frame();
    rooms(1);
    for (int i = 0; i < 5; ++i) {
      const auto* r = fi::blend_draw(990, 0, shard(i, 1));
      CHECK(r != nullptr);
      if (r != nullptr) {
        const float turn = (35.f + 13.f * i) * 3.14159265f / 180.f;
        CHECK(std::fabs(r->posnormalmatrix[0][0] - sizes[i] * std::cos(turn)) < 2e-3f);
        CHECK(std::fabs(r->posnormalmatrix[0][3] - (20.f * i + 1.5f)) < 1e-3f);
      }
    }
  }

  // The boat's trail, drawn by its emitter as strips through its particles:
  // each strip pairs with the one at its place the frame before, and its
  // vertices blend in order. A steady trail's particles move one place a
  // frame, so that is the trail half a frame later, and its front stays at
  // the boat (halfway) instead of half a frame ahead of it.
  {
    fi::end_game_frame();
    fi::end_game_frame();
    static const uint8_t payload[48] = {3};
    const auto strip = [](uint32_t part, float boatZ) {
      gxc::DrawPlan plan;
      plan.match_payload = payload;
      plan.match_payload_size = sizeof(payload);
      plan.match_primitive = 0x98;
      plan.match_direct_position = true;
      plan.vertex_count = 6;
      plan.draw_scope = 0x4321;
      plan.draw_scope_part = part;
      plan.vertices.assign(6 * gxc::kCompactVertexFloats, 0.f);
      for (int i = 0; i < 6; ++i) {
        float* v = plan.vertices.data() + i * gxc::kCompactVertexFloats;
        v[0] = (i % 3) * 10.f - 10.f;
        v[2] = boatZ + 60.f * (part - 1 + i / 3); // a row every 60 units behind the boat
      }
      return plan;
    };
    const auto view = draw_at(0, -50, -400);
    const auto s1 = strip(1, 0.f);
    blend_plan(s1, view);
    const auto s2 = strip(2, 0.f);
    blend_plan(s2, view);
    fi::end_game_frame();
    const auto n1 = strip(1, -60.f); // the boat sailed 60 units on
    blend_plan(n1, view);
    const float* front = fi::blended_positions();
    CHECK(front != nullptr && near(front[2], -30.f) && near(front[3 * 3 + 2], 30.f));
  }

  // A cloth's strips (a scope over draws that index their positions): the
  // game moves the vertices itself, under a matrix that stays put, so each
  // vertex is blended halfway, as a particle's are.
  {
    fi::end_game_frame();
    fi::end_game_frame();
    static const uint8_t payload[24] = {7};
    const auto strip = [](uint32_t part, float sway) {
      gxc::DrawPlan plan;
      plan.match_payload = payload;
      plan.match_payload_size = sizeof(payload);
      plan.match_primitive = 0x98;
      plan.match_direct_position = false; // indexed positions
      plan.vertex_count = 4;
      plan.draw_scope = 0x5151;
      plan.draw_scope_part = part;
      plan.vertices.assign(4 * gxc::kCompactVertexFloats, 0.f);
      for (int i = 0; i < 4; ++i) {
        float* v = plan.vertices.data() + i * gxc::kCompactVertexFloats;
        v[0] = (i & 1) * 20.f + sway * (i >> 1); // the free edge sways
        v[1] = (i >> 1) * 30.f;
      }
      return plan;
    };
    const auto pole = draw_at(0, 0, -300);
    blend_plan(strip(1, 0.f), pole);
    blend_plan(strip(2, 0.f), pole);
    fi::end_game_frame();
    CHECK(blend_plan(strip(1, 8.f), pole) == nullptr); // the matrix did not move
    const float* swayed = fi::blended_positions();
    CHECK(swayed != nullptr && near(swayed[2 * 3], 4.f) && near(swayed[3 * 3], 24.f) && near(swayed[0], 0.f));
    // The second strip did not move: drawn as it is.
    blend_plan(strip(2, 0.f), pole);
    CHECK(fi::blended_positions() == nullptr);
  }

  // Colours: a TEV draw's colour and konst registers and its fog colour, and
  // the material and light colours, halfway between the two frames' (a fade,
  // a particle's colour over its life). The rest of the pixel constants and
  // the alpha-test references stay this frame's.
  {
    fi::end_game_frame();
    fi::end_game_frame();
    gxc::PixelShaderConstants before{};
    before.kcolors[0][0] = 0;
    before.kcolors[0][3] = 255;
    before.colors[1][0] = -100;
    before.fogcolor[2] = 10;
    before.alpha_ref[0] = 4;
    gxc::PixelShaderConstants after = before;
    after.kcolors[0][0] = 200;
    after.kcolors[0][3] = 0;
    after.colors[1][0] = 101;
    after.fogcolor[2] = 30;
    after.alpha_ref[0] = 90;
    auto lit = draw_at(0, 0, -500);
    lit.materials[2][3] = 255;
    lit.lights[0].color[1] = 40;
    fi::DrawInput input;
    input.key = 4242;
    fi::blend_draw(input, lit, false, &before);
    CHECK(fi::blended_pixel(0) == nullptr); // no previous frame for it
    fi::end_game_frame();
    auto faded = lit;
    faded.materials[2][3] = 0;
    faded.lights[0].color[1] = 80;
    const auto* block = fi::blend_draw(input, faded, false, &after);
    const auto* pixel = fi::blended_pixel(0);
    CHECK(pixel != nullptr);
    if (pixel != nullptr) {
      CHECK(pixel->kcolors[0][0] == 100 && pixel->kcolors[0][3] == 128);
      CHECK(pixel->colors[1][0] == 1); // round(-100 + 201 / 2)
      CHECK(pixel->fogcolor[2] == 20);
      CHECK(pixel->alpha_ref[0] == 90);
    }
    CHECK(block != nullptr && block->materials[2][3] == 128 && block->lights[0].color[1] == 60);
    // The same colours the next frame: nothing to blend.
    fi::end_game_frame();
    fi::blend_draw(input, faded, false, &after);
    CHECK(fi::blended_pixel(0) == nullptr);
  }

  // 120 Hz: three in-between frames at a quarter, half and three quarters of
  // the way. A draw moving at 40 units a frame is at 10, 20 and 30; a turn of
  // the camera by 24 degrees puts what stands still at 6, 12 and 18 degrees
  // of it; a particle's corners move in quarters too.
  {
    fi::set_steps(3);
    fi::end_game_frame();
    fi::end_game_frame();
    CHECK(fi::frame_steps() == 3);
    const auto rooms = [](float yaw, int skip) {
      for (int room = 0; room < 8; ++room)
        if (room != skip)
          fi::blend_draw(1100 + room, 0, seen(yaw, 0, 0, room * 300.f - 1000.f, -50.f, -900.f - room * 40.f));
    };
    static const uint8_t payload[48] = {4};
    const auto spark = [](float x) {
      gxc::DrawPlan plan;
      plan.match_payload = payload;
      plan.match_payload_size = sizeof(payload);
      plan.match_primitive = 0x80;
      plan.match_direct_position = true;
      plan.vertex_count = 4;
      plan.draw_tag = 0x777;
      plan.vertices.assign(4 * gxc::kCompactVertexFloats, 0.f);
      for (int i = 0; i < 4; ++i) {
        plan.vertices[i * gxc::kCompactVertexFloats] = x + (i & 1) * 5.f;
        plan.vertices[i * gxc::kCompactVertexFloats + 2] = -400.f;
      }
      return plan;
    };
    // A moving draw and a spark, the camera still.
    fi::blend_draw(1150, 0, draw_at(0, 0, -400));
    auto p0 = spark(0.f);
    blend_plan(p0, draw_at(0, 0, 0));
    fi::end_game_frame();
    const auto* moving = fi::blend_draw(1150, 0, draw_at(40, 0, -400));
    CHECK(moving != nullptr);
    if (moving != nullptr) {
      for (int step = 0; step < 3; ++step)
        CHECK(near(fi::blended_step(step)->posnormalmatrix[0][3], 10.f * (step + 1)));
    }
    auto p1 = spark(40.f);
    p1.draw_tag_age = 1;
    blend_plan(p1, draw_at(0, 0, 0));
    for (int step = 0; step < 3; ++step) {
      const float* corners = fi::blended_positions(step);
      CHECK(corners != nullptr && near(corners[0], 10.f * (step + 1)));
    }
    // The camera turns 24 degrees; the rooms stand still.
    fi::end_game_frame();
    fi::end_game_frame();
    rooms(0.f, -1);
    fi::end_game_frame();
    rooms(24.f, 3);
    const auto* room = fi::blend_draw(1100 + 3, 0, seen(24.f, 0, 0, 3 * 300.f - 1000.f, -50.f, -900.f - 3 * 40.f));
    CHECK(room != nullptr);
    if (room != nullptr) {
      for (int step = 0; step < 3; ++step) {
        const float yaw = 6.f * (step + 1);
        const auto expected = seen(yaw, 0, 0, 3 * 300.f - 1000.f, -50.f, -900.f - 3 * 40.f);
        const auto* got = fi::blended_step(step);
        CHECK(std::fabs(got->posnormalmatrix[0][0] - expected.posnormalmatrix[0][0]) < 1e-3f);
        CHECK(std::fabs(got->posnormalmatrix[0][3] - expected.posnormalmatrix[0][3]) < 0.5f &&
              std::fabs(got->posnormalmatrix[2][3] - expected.posnormalmatrix[2][3]) < 0.5f);
      }
    }
    // The camera turns 24 degrees and moves: each quarter step is the same
    // motion (the screw motion's power), so the frames advance evenly.
    fi::end_game_frame();
    fi::end_game_frame();
    const auto moved_rooms = [](float yaw, float cx, float cz, int skip) {
      for (int room = 0; room < 8; ++room)
        if (room != skip)
          fi::blend_draw(1200 + room, 0, seen(yaw, cx, cz, room * 300.f - 1000.f, -50.f, -900.f - room * 40.f));
    };
    moved_rooms(0.f, 0.f, 0.f, -1);
    fi::end_game_frame();
    moved_rooms(24.f, 60.f, -40.f, 5);
    const auto* far = fi::blend_draw(1205, 0, seen(24.f, 60.f, -40.f, 5 * 300.f - 1000.f, -50.f, -900.f - 5 * 40.f));
    CHECK(far != nullptr);
    if (far != nullptr) {
      // The step-to-step motion S = M_{k+1} M_k^-1, for rows of 3x4.
      const auto inverse = [](const float m[][4], double out[3][4]) {
        // Rigid with unit scale here: the inverse is the transpose.
        for (int r = 0; r < 3; ++r) {
          for (int c = 0; c < 3; ++c)
            out[r][c] = m[c][r];
          out[r][3] = -(m[0][r] * m[0][3] + m[1][r] * m[1][3] + m[2][r] * m[2][3]);
        }
      };
      const auto step_motion = [&](int k, double out[3][4]) {
        double inv[3][4];
        inverse(fi::blended_step(k)->posnormalmatrix, inv);
        const auto& next = fi::blended_step(k + 1)->posnormalmatrix;
        for (int r = 0; r < 3; ++r) {
          for (int c = 0; c < 4; ++c)
            out[r][c] = next[r][0] * inv[0][c] + next[r][1] * inv[1][c] + next[r][2] * inv[2][c];
          out[r][3] += next[r][3];
        }
      };
      double first[3][4], second[3][4];
      step_motion(0, first);
      step_motion(1, second);
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
          CHECK(std::fabs(first[r][c] - second[r][c]) < (c == 3 ? 0.05 : 1e-4));
    }
    fi::set_steps(1);
    fi::end_game_frame();
    CHECK(fi::frame_steps() == 1);
  }

  // Draw keys: the same payload gives the same key; another primitive does not.
  {
    const uint8_t payload[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    gxc::DrawPlan plan;
    plan.match_payload = payload;
    plan.match_payload_size = sizeof(payload);
    plan.match_primitive = 0x90;
    plan.vertex_count = 3;
    const uint64_t k1 = fi::draw_key(plan);
    CHECK(k1 != 0 && k1 == fi::draw_key(plan));
    plan.match_primitive = 0x98;
    CHECK(fi::draw_key(plan) != k1);
    plan.match_payload = nullptr;
    CHECK(fi::draw_key(plan) == 0);
  }

  // Pacing: overloads in 3 of 8 game frames take 120 Hz to 60, and more after
  // the frames in flight to none; one now and then does nothing. 3 s of calm
  // brings a level back, and an overload soon after coming back makes the
  // next wait twice as long.
  {
    const auto overload_frames = [](int n) {
      for (int i = 0; i < n; ++i) {
        fi::note_overload("test");
        fi::end_game_frame();
      }
    };
    fi::set_steps(3);
    fi::end_game_frame();
    CHECK(fi::frame_steps() == 3 && !fi::frame_skipped());
    // Now and then: never a drop.
    for (int i = 0; i < 40; ++i) {
      overload_frames(1);
      for (int j = 0; j < 8; ++j)
        fi::end_game_frame();
    }
    CHECK(fi::frame_steps() == 3);
    overload_frames(2);
    CHECK(fi::frame_steps() == 3);
    overload_frames(1);
    CHECK(fi::frame_steps() == 1 && !fi::frame_skipped());
    overload_frames(3); // still the frames in flight: no second drop
    CHECK(fi::frame_steps() == 1 && !fi::frame_skipped());
    overload_frames(1); // with the three before: the next drop
    CHECK(fi::frame_skipped());
    for (int i = 0; i < 89; ++i)
      fi::end_game_frame();
    CHECK(fi::frame_skipped());
    fi::end_game_frame();
    CHECK(!fi::frame_skipped() && fi::frame_steps() == 1);
    for (int i = 0; i < 90; ++i)
      fi::end_game_frame();
    CHECK(fi::frame_steps() == 3);
    overload_frames(3);
    CHECK(fi::frame_steps() == 1);
    for (int i = 0; i < 90; ++i)
      fi::end_game_frame();
    CHECK(fi::frame_steps() == 1);
    for (int i = 0; i < 90; ++i)
      fi::end_game_frame();
    CHECK(fi::frame_steps() == 3);
    // A new setting starts over at it.
    overload_frames(3);
    CHECK(fi::frame_steps() == 1);
    fi::set_steps(1);
    fi::end_game_frame();
    CHECK(fi::frame_steps() == 1 && !fi::frame_skipped());
  }

  if (g_failures == 0)
    std::puts("frame_interp_test: all checks passed");
  return g_failures == 0 ? 0 : 1;
}
