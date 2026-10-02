#include "frame_interp.hpp"

#include <algorithm>
#include <bit>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>

#include <aurora/aurora.h>

namespace aurora::gfx::frame_interp {
namespace gxc = gxruntime::gxcore;
namespace {

bool env_flag(const char* name, bool fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0')
    return fallback;
  return value[0] != '0';
}

uint64_t env_u64(const char* name, uint64_t fallback) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' ? std::strtoull(value, nullptr, 10) : fallback;
}

std::atomic_bool g_enabled{env_flag("DOL_AURORA_FRAME_INTERP", false)};
// In-between frames per game frame (1: 60 Hz, 3: 120 Hz; DOL_AURORA_FRAME_INTERP_STEPS),
// asked for at any time and taken at a game frame's start (g_frameSteps).
std::atomic<int> g_steps{static_cast<int>(std::clamp<uint64_t>(env_u64("DOL_AURORA_FRAME_INTERP_STEPS", 1), 1, kMaxSteps))};
int g_frameSteps = g_steps.load(std::memory_order_relaxed);
// In-between frames need not make the game wait (pace_steps): when the GPU
// or the render worker keeps falling behind (note_overload), 60 Hz drops to
// none, and comes back after a calm spell (120 Hz to 60 only when asked).
std::atomic_bool g_overloaded{false};
std::atomic<const char*> g_overloadWhy{nullptr}; // the latest overload's cause, for the log
bool g_frameSkipped = false; // the recording frame has none (recording thread)
struct Pacing {
  uint64_t frame = 0;
  uint64_t lastDrop = 0;
  uint64_t lastRaise = 0;
  int wanted = 0;  // g_steps when the budget was last set
  int budget = 0;  // in-between frames allowed now, 0 to wanted
  int calm = 0;    // game frames since the last overload
  int calmNeeded = 0;
  uint32_t recent = 0; // one bit a game frame, newest lowest: an overload reported
  // The game's own speed (slow_game): when the last game frame ended here,
  // the gaps between the last 60 (a ring), and the last two checks' medians.
  std::chrono::steady_clock::time_point lastEnd{};
  std::array<float, 60> gapRing{};
  int gaps = 0;      // gaps in the ring, up to its size
  int gapNext = 0;   // where the next gap goes
  int sinceCheck = 0;
  int slowChecks = 0; // consecutive checks whose median was slow
  double gapMs = 0.0; // the latest check's median
} g_pacing;
std::atomic_bool g_encodingInterpolated{false};
std::atomic<uint64_t> g_gameFrame{1};

// One game frame's draws: their constants (a consecutive run of identical
// blocks is stored once) and, per draw, its key and block, a few of its
// vertices (see draw_key), where its position matrix put it, and whether it
// stood still (the camera's motion carried a copy from the frame before onto
// it; see blend_draw).
constexpr uint32_t kNoSamples = UINT32_MAX;
using VertexSamples = std::array<float, 9>;
struct Record {
  uint64_t key;
  uint32_t constants;
  uint32_t samples; // index in FrameRecords::samples, or kNoSamples
  float position[3];
  bool still;
  // A draw whose positions came in its payload (see blend_positions()): its
  // first position in FrameRecords::positions (kNoSamples: not kept), how
  // many, and a tagged draw's age.
  uint32_t positions;
  uint32_t positionCount;
  uint32_t age;
};
struct FrameRecords {
  std::vector<gxc::VertexShaderConstants> pool;
  std::vector<Record> records;
  std::vector<VertexSamples> samples;
  std::vector<float> positions; // x, y, z per vertex
  void clear() {
    pool.clear();
    records.clear();
    samples.clear();
    positions.clear();
  }
};

// At most this many vertices per draw are kept and blended.
constexpr uint32_t kMaxBlendedVertices = 1024;
// blend_draw()'s in-between positions for its draw (x, y, z per vertex), per step.
std::vector<float> g_blendedPositions[kMaxSteps];
bool g_haveBlendedPositions = false;

FrameRecords g_frames[2];
unsigned g_current = 0;
bool g_havePrevious = false;
// The previous frame's records by key, in draw order within a key.
std::vector<Record> g_previousIndex;
// Per key in this frame: how many draws had it, and the previous frame's copy
// after the last one a motion carried here.
struct KeyState {
  uint32_t occurrence = 0;
  uint32_t next = 0;
};
absl::flat_hash_map<uint64_t, KeyState> g_keys;
// The in-between blocks blend_draw() made, one per step.
gxc::VertexShaderConstants g_blended[kMaxSteps];

// Most draws repeat the constants of the draw before them (a model's parts,
// copies drawn with one matrix): 96 percent in the Forsaken Fortress, where a
// frame has 17,500 draws. For those the caller's comparison is enough: the
// pool entry is not compared again, and an in-between block made from the same
// match the same way is the one g_blended already holds.
uint64_t g_drawSerial = 0;
uint64_t g_pooledSerial = 0; // the draw whose constants are the current pool's last
uint64_t g_motionGeneration = 0; // bumped when any motion's values change
enum class BlendPath { Main, OwnMotion, StoodBefore, Unmatched };
struct LastBlend {
  uint64_t serial = 0; // 0: none
  BlendPath path = BlendPath::Main;
  const void* previous = nullptr;
  const void* motion = nullptr;
  uint64_t motionGeneration = 0;
  uint64_t rows = 0;
  bool identical = false;
};
LastBlend g_lastBlend;
bool g_blendRepeated = false;

// Motions seen this frame (see vote_motion()): M_now * M_before^-1 of the
// draws matched without ambiguity, how many draws agree with each, and how
// much their votes weigh.
struct Motion {
  double m[3][4];
  uint32_t votes;
  double weight;
  // For the camera's motion (see camera_motion()): its inverse, and the part
  // of it each in-between frame is at (D^t: the rotation turned by t of its
  // angle about the same screw axis, so D^0.5 * D^0.5 = D).
  bool derived;
  bool bounded;
  double inverse[3][4];
  double part[kMaxSteps][3][4];
};
constexpr int kMaxMotions = 6;
constexpr uint32_t kMinMotionVotes = 4;
// A vote weighs one per this many units of its draw's distance, and at least
// one (see vote_motion()).
constexpr double kVoteReach = 1000.0;
// At most this many copies of a model, none standing still, are one moving
// model drawn more than once (see blend_draw).
constexpr size_t kFewCopies = 4;
Motion g_motions[kMaxMotions];
int g_motionCount = 0;
int g_leadingMotion = -1;
// The camera's motion the frame before, for the draws of this frame that come
// before enough of its own have voted (the sky, the first rooms): a turning
// camera turns about as far from one frame to the next.
Motion g_predicted;
bool g_havePredicted = false;
// The largest camera motion taken for a turn rather than a cut: a mouse can
// swing the camera tens of degrees in a frame, but a cut usually moves it far.
constexpr double kMaxCameraTurn = 70.0 * M_PI / 180.0;
constexpr double kMaxCameraShift = 1000.0;
// Where the camera's motion carries each of the previous frame's copies, in a
// grid per key, built the first time a key needs it in a frame: a fast turn
// brings dozens of copies into view at once, and finding each one's copy (or
// that it has none) must not cost a pass over every copy of its model.
constexpr double kCellSize = 8.0;
absl::flat_hash_map<uint64_t, uint32_t> g_cellHead; // (key, cell) -> first copy + 1
std::vector<uint32_t> g_cellNext;                   // per previous record: next in its cell + 1
std::vector<std::array<float, 3>> g_carried;        // per previous record: where the camera carries it
absl::flat_hash_set<uint64_t> g_griddedKeys;
// The constants of the last draw found new in view: the other parts of that
// copy (grass is nine draws with one matrix) are new too.
uint32_t g_newInView = UINT32_MAX;

struct Counts {
  uint64_t draws = 0;
  uint64_t matched = 0;   // a plausible counterpart in the previous frame
  uint64_t identical = 0; // matched, and nothing changed
  uint64_t blended = 0;   // matched and blended
  uint64_t rejected = 0;  // counterparts exist, none plausible (with the new in view)
  uint64_t fresh = 0;     // of those, copies just come into view (blended from where they stood)
  uint64_t unmatched = 0; // no counterpart
  uint64_t moved = 0;     // matched draws whose vertices the game moved (see vertex_motion())
  uint64_t positions = 0; // particles whose positions were blended (see blend_positions())
};
Counts g_frameCounts;
Counts g_totalCounts;
uint64_t g_framesSeen = 0;
uint64_t g_framesInterpolated = 0;
bool g_lastVerdict = false;

const bool g_log = env_flag("DOL_AURORA_FRAME_INTERP_LOG", false);
// DOL_AURORA_FRAME_INTERP_LOG_FRAMES: a line per game frame (debug).
const bool g_logFrames = env_flag("DOL_AURORA_FRAME_INTERP_LOG_FRAMES", false);
// DOL_AURORA_FRAME_INTERP_TRACE=frame or first-last: a line per draw (debug).
struct TraceRange {
  uint64_t first = 0, last = 0;
};
const TraceRange g_trace = [] {
  TraceRange range;
  const char* value = std::getenv("DOL_AURORA_FRAME_INTERP_TRACE");
  if (value == nullptr || value[0] == '\0')
    return range;
  char* end = nullptr;
  range.first = range.last = std::strtoull(value, &end, 10);
  if (end != nullptr && *end == '-')
    range.last = std::strtoull(end + 1, nullptr, 10);
  return range;
}();
const char* g_outcome = "";
const char* g_rejectReason = "";

// --- Matching -----------------------------------------------------------------

double translation_length(const float rows[][4]) {
  return std::sqrt(double(rows[0][3]) * rows[0][3] + double(rows[1][3]) * rows[1][3] +
                   double(rows[2][3]) * rows[2][3]);
}

// How far plausible_matrix() lets a linear part turn from one frame to the
// next, as |A - B|^2 / k^2 = 4 (1 - cos t) for a rotation by t at scale k.
// Copies of one model: 41 degrees, since a turning camera must not pair a
// copy with the next one along. A draw with a key of its own is the same
// object as its counterpart whatever it did: 150 degrees. The bones of
// Link's sword arm turn 90 degrees and more in one game frame of a swing; at
// 41 degrees his arm and sword were drawn where the next frame has them
// while the rest of him was halfway, and stepped at 30 FPS.
constexpr double kTurnLimit = 1.0;
constexpr double kOwnTurnLimit = 7.46;

// Whether two 3x4 matrices can be one object's transform a frame apart. A
// camera cut or a reused draw for another object fails at least one bound:
// scale within 1.5x, the linear part within the turn limit, and the
// translation within a fifth of its distance from the camera plus 100 units
// (a camera turning 10 degrees a frame moves a point about 0.17 of its
// distance; the boat at full sail covers about 60 units a frame).
bool plausible_matrix(const float current[][4], const float previous[][4], double turnLimit) {
  if (std::memcmp(current, previous, sizeof(float) * 12) == 0)
    return true;
  double currentScale = 0.0, previousScale = 0.0, difference = 0.0;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      const double a = current[r][c];
      const double b = previous[r][c];
      currentScale += a * a;
      previousScale += b * b;
      difference += (a - b) * (a - b);
    }
  }
  if (currentScale < 1e-12 || previousScale < 1e-12) {
    g_rejectReason = "collapsed";
    return false; // a collapsed matrix hides the object; do not grow it halfway
  }
  const double ratio = currentScale / previousScale;
  if (ratio > 2.25 || ratio < 1.0 / 2.25) {
    g_rejectReason = "scale";
    return false;
  }
  // For a rotation by t at scale k, |A-B|^2 = 4 k^2 (1 - cos t): 1.0 is 41 deg.
  const double axisScale = std::max(currentScale, previousScale) / 3.0;
  if (difference / axisScale > turnLimit) {
    g_rejectReason = "rotation";
    return false;
  }
  double moved = 0.0;
  for (int r = 0; r < 3; ++r) {
    const double d = double(current[r][3]) - previous[r][3];
    moved += d * d;
  }
  moved = std::sqrt(moved);
  const double distance = std::max(translation_length(current), translation_length(previous));
  if (moved > 0.2 * distance + 100.0) {
    g_rejectReason = "translation";
    return false;
  }
  return true;
}

// Projection: the same kind (perspective or orthographic), and the scale terms
// within 25 percent (a zoom, not a different camera).
bool plausible_projection(const float current[4][4], const float previous[4][4]) {
  if (std::memcmp(current, previous, sizeof(float) * 16) == 0)
    return true;
  const bool currentPerspective = current[3][2] != 0.f;
  const bool previousPerspective = previous[3][2] != 0.f;
  if (currentPerspective != previousPerspective) {
    g_rejectReason = "projection kind";
    return false;
  }
  for (int i = 0; i < 2; ++i) {
    const float a = current[i][i];
    const float b = previous[i][i];
    if (a == 0.f || b == 0.f) {
      g_rejectReason = "projection zero";
      return a == b;
    }
    const float ratio = a / b;
    if (ratio > 1.25f || ratio < 0.8f) {
      g_rejectReason = "projection scale";
      return false;
    }
  }
  return true;
}

bool plausible(const gxc::VertexShaderConstants& current, const gxc::VertexShaderConstants& previous,
               uint64_t usedMatrixRows, double turnLimit) {
  if (!plausible_projection(current.projection, previous.projection))
    return false;
  if (usedMatrixRows == 0)
    return plausible_matrix(current.posnormalmatrix, previous.posnormalmatrix, turnLimit);
  // Per-vertex matrices (skinned models): the bank slots this draw's vertices
  // use. The other slots hold whatever other models loaded last.
  for (uint64_t rows = usedMatrixRows; rows != 0; rows &= rows - 1) {
    const int row = __builtin_ctzll(rows);
    if (!plausible_matrix(&current.transformmatrices[row], &previous.transformmatrices[row], turnLimit))
      return false;
  }
  return true;
}

// The matrix that places a draw: the current position matrix, or for
// per-vertex matrices (skinned models) the first bank slot.
const float (*position_signature(const gxc::VertexShaderConstants& constants, uint64_t usedMatrixRows))[4] {
  return usedMatrixRows != 0 ? &constants.transformmatrices[__builtin_ctzll(usedMatrixRows)]
                             : constants.posnormalmatrix;
}

double translation_distance2(const float a[][4], const float b[3]) {
  double d = 0.0;
  for (int r = 0; r < 3; ++r) {
    const double delta = double(a[r][3]) - b[r];
    d += delta * delta;
  }
  return d;
}

// --- Camera motion ------------------------------------------------------------
//
// A draw's matrix is the view matrix times the object's own, so everything
// that stands still (the rooms, grass, bushes, trees, ropes) moves from one
// frame to the next by one transform, the camera's: D = V_now * V_before^-1 =
// M_now * M_before^-1 for any of them. Each draw whose key is unique is a vote
// for its M_now * M_before^-1. The rooms cast hundreds of votes for the
// camera's motion; the sky, drawn around the camera, and each moving actor
// cast a few for their own.
//
// Copies of one model are then matched by where such a motion carries each
// previous copy. Neither the draw order nor the nearest copy is enough for a
// list of copies: the game culls grass, bushes and trees one by one, so a copy
// leaving the view at the start of the list shifts every copy after it onto
// its neighbour, and a turning camera moves a distant copy farther than the
// spacing between copies. Either pairs a clump with the next clump, which is
// close enough to look plausible, and the in-between frame draws it halfway
// there: the foliage flickers at 60 FPS.

// out = a * b for 3x4 affine matrices.
template <typename T>
void affine_multiply(const double a[3][4], const T b[][4], double out[3][4]) {
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 4; ++c)
      out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
    out[r][3] += a[r][3];
  }
}

template <typename T>
bool affine_inverse(const T m[][4], double out[3][4]) {
  const double a = m[0][0], b = m[0][1], c = m[0][2];
  const double d = m[1][0], e = m[1][1], f = m[1][2];
  const double g = m[2][0], h = m[2][1], i = m[2][2];
  const double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (std::fabs(det) < 1e-12)
    return false;
  const double linear[3][3] = {
      {(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det},
      {(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det},
      {(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det},
  };
  for (int r = 0; r < 3; ++r) {
    for (int k = 0; k < 3; ++k)
      out[r][k] = linear[r][k];
    out[r][3] = -(linear[r][0] * m[0][3] + linear[r][1] * m[1][3] + linear[r][2] * m[2][3]);
  }
  return true;
}

// Whether `motion` carries `previous` onto `current`: the linear part within
// 0.1 percent, the translation within float rounding at that distance.
bool motion_agrees(const double motion[3][4], const float current[][4], const float previous[][4]) {
  double carried[3][4];
  affine_multiply(motion, previous, carried);
  double linear = 0.0, scale = 0.0, moved = 0.0;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      const double delta = carried[r][c] - current[r][c];
      linear += delta * delta;
      scale += double(current[r][c]) * current[r][c];
    }
    const double delta = carried[r][3] - current[r][3];
    moved += delta * delta;
  }
  const double tolerance = 0.5 + 1e-5 * translation_length(current);
  return linear <= 1e-6 * scale && moved <= tolerance * tolerance;
}

// The camera's motion is the one the scenery shares, and what stands farthest
// pins it down: a vote weighs as much as its draw is far from the camera, in
// thousands of units (one at least, so near the camera every draw counts the
// same). At sea the camera follows the boat, and the boat and Link are drawn
// before the scenery: their hundred parts, all moving with the boat, agreed
// on its motion before the islands and the sea voted, so the boat's motion
// was taken for the camera's through most of the frame. The sky, drawn around
// the camera, turns with it but does not move with it, and weighs little too.
void vote_motion(const float current[][4], const float previous[][4]) {
  const double weight = std::max(1.0, translation_length(current) / kVoteReach);
  int agreed = -1;
  if (g_leadingMotion >= 0 && motion_agrees(g_motions[g_leadingMotion].m, current, previous))
    agreed = g_leadingMotion;
  for (int i = 0; agreed < 0 && i < g_motionCount; ++i) {
    if (i != g_leadingMotion && motion_agrees(g_motions[i].m, current, previous))
      agreed = i;
  }
  if (agreed >= 0) {
    ++g_motions[agreed].votes;
    g_motions[agreed].weight += weight;
  } else {
    double inverse[3][4];
    if (!affine_inverse(previous, inverse))
      return;
    // A new motion, in place of the one with the least weight.
    agreed = 0;
    if (g_motionCount < kMaxMotions) {
      agreed = g_motionCount++;
    } else {
      for (int i = 1; i < kMaxMotions; ++i) {
        if (g_motions[i].weight < g_motions[agreed].weight)
          agreed = i;
      }
      if (agreed == g_leadingMotion)
        g_leadingMotion = -1;
    }
    double(*m)[4] = g_motions[agreed].m;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 4; ++c)
        m[r][c] = double(current[r][0]) * inverse[0][c] + double(current[r][1]) * inverse[1][c] +
                  double(current[r][2]) * inverse[2][c];
      m[r][3] += current[r][3];
    }
    g_motions[agreed].votes = 1;
    g_motions[agreed].weight = weight;
    g_motions[agreed].derived = false;
    ++g_motionGeneration;
  }
  // The motion with the most weight is the camera's.
  if (g_leadingMotion < 0 || g_motions[agreed].weight > g_motions[g_leadingMotion].weight)
    g_leadingMotion = agreed;
}

// How far from `current` the motion carries a previous position.
double motion_error(const Motion& motion, const float current[][4], const float previous[3]) {
  double error = 0.0;
  for (int r = 0; r < 3; ++r) {
    const double carried = motion.m[r][0] * previous[0] + motion.m[r][1] * previous[1] +
                           motion.m[r][2] * previous[2] + motion.m[r][3];
    error += (carried - current[r][3]) * (carried - current[r][3]);
  }
  return std::sqrt(error);
}

uint64_t cell_key(uint64_t key, int64_t x, int64_t y, int64_t z) {
  uint64_t h = key * 0x9E3779B97F4A7C15ull;
  h ^= static_cast<uint64_t>(x) * 0xC2B2AE3D27D4EB4Full;
  h ^= static_cast<uint64_t>(y) * 0x165667B19E3779F9ull;
  h ^= static_cast<uint64_t>(z) * 0x27D4EB2F165667C5ull;
  return h ^ (h >> 29);
}

int64_t cell_of(double value) { return static_cast<int64_t>(std::floor(value / kCellSize)); }

// The copies of `key` (g_previousIndex[first, first + count)) where `motion`
// carries them, into the grid.
void grid_copies(uint64_t key, const Motion& motion, size_t first, size_t count) {
  if (!g_griddedKeys.insert(key).second)
    return;
  if (g_carried.size() < g_previousIndex.size()) {
    g_carried.resize(g_previousIndex.size());
    g_cellNext.resize(g_previousIndex.size());
  }
  for (size_t i = first; i < first + count; ++i) {
    const float* p = g_previousIndex[i].position;
    double c[3];
    for (int r = 0; r < 3; ++r)
      c[r] = motion.m[r][0] * p[0] + motion.m[r][1] * p[1] + motion.m[r][2] * p[2] + motion.m[r][3];
    g_carried[i] = {float(c[0]), float(c[1]), float(c[2])};
    uint32_t& head = g_cellHead[cell_key(key, cell_of(c[0]), cell_of(c[1]), cell_of(c[2]))];
    g_cellNext[i] = head;
    head = static_cast<uint32_t>(i + 1);
  }
}

// The copy of `key` the camera carries onto `here` (within `tolerance`), as
// an index from `first`, or `count` if none; and how far off the nearest
// copy that stood still is (HUGE_VAL if none is within a cell).
size_t find_carried(uint64_t key, const float here[][4], size_t first, size_t count, double tolerance,
                    double& nearestStill) {
  size_t pick = count;
  double best = tolerance;
  const int64_t cx = cell_of(here[0][3]), cy = cell_of(here[1][3]), cz = cell_of(here[2][3]);
  for (int64_t dx = -1; dx <= 1; ++dx) {
    for (int64_t dy = -1; dy <= 1; ++dy) {
      for (int64_t dz = -1; dz <= 1; ++dz) {
        const auto it = g_cellHead.find(cell_key(key, cx + dx, cy + dy, cz + dz));
        for (uint32_t link = it == g_cellHead.end() ? 0u : it->second; link != 0u; link = g_cellNext[link - 1]) {
          const size_t i = link - 1;
          const auto& c = g_carried[i];
          const double ex = c[0] - here[0][3], ey = c[1] - here[1][3], ez = c[2] - here[2][3];
          const double e = std::sqrt(ex * ex + ey * ey + ez * ez);
          if (e <= best) {
            best = e;
            pick = i - first;
          }
          if (g_previousIndex[i].still)
            nearestStill = std::min(nearestStill, e);
        }
      }
    }
  }
  return pick;
}

double step_weight(int step);

// The inverse of a camera motion and its part at the blend weight, once; a
// motion that turns or shifts too far to be a turn is not used (a cut).
bool derive(Motion& motion) {
  if (motion.derived)
    return motion.bounded;
  ++g_motionGeneration;
  motion.derived = true;
  motion.bounded = false;
  const double(*m)[4] = motion.m;
  if (!affine_inverse(m, motion.inverse))
    return false;
  // A camera's motion is rigid: a rotation and a shift, no scale or skew.
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) {
      const double dot = m[0][i] * m[0][k] + m[1][i] * m[1][k] + m[2][i] * m[2][k];
      if (std::fabs(dot - (i == k ? 1.0 : 0.0)) > 2e-3)
        return false;
    }
  }
  // The rotation as a quaternion (the camera's is rigid).
  double w, x, y, z;
  const double trace = m[0][0] + m[1][1] + m[2][2];
  if (trace > 0.0) {
    const double s = std::sqrt(trace + 1.0) * 2.0;
    w = 0.25 * s;
    x = (m[2][1] - m[1][2]) / s;
    y = (m[0][2] - m[2][0]) / s;
    z = (m[1][0] - m[0][1]) / s;
  } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
    const double s = std::sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0;
    w = (m[2][1] - m[1][2]) / s;
    x = 0.25 * s;
    y = (m[0][1] + m[1][0]) / s;
    z = (m[0][2] + m[2][0]) / s;
  } else if (m[1][1] > m[2][2]) {
    const double s = std::sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0;
    w = (m[0][2] - m[2][0]) / s;
    x = (m[0][1] + m[1][0]) / s;
    y = 0.25 * s;
    z = (m[1][2] + m[2][1]) / s;
  } else {
    const double s = std::sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0;
    w = (m[1][0] - m[0][1]) / s;
    x = (m[0][2] + m[2][0]) / s;
    y = (m[1][2] + m[2][1]) / s;
    z = 0.25 * s;
  }
  const double norm = std::sqrt(w * w + x * x + y * y + z * z);
  if (!(norm > 1e-9))
    return false;
  w /= norm, x /= norm, y /= norm, z /= norm;
  if (w < 0.0)
    w = -w, x = -x, y = -y, z = -z;
  const double angle = 2.0 * std::acos(std::min(1.0, w));
  const double shift = std::sqrt(m[0][3] * m[0][3] + m[1][3] * m[1][3] + m[2][3] * m[2][3]);
  if (angle > kMaxCameraTurn || shift > kMaxCameraShift)
    return false;
  const double axis = std::sqrt(x * x + y * y + z * z);
  for (int step = 0; step < g_frameSteps; ++step) {
    // The rotation by t of the angle about the same axis.
    const double t = step_weight(step);
    double pw = 1.0, px = 0.0, py = 0.0, pz = 0.0;
    if (axis > 1e-12) {
      const double half = 0.5 * angle * t;
      pw = std::cos(half);
      const double k = std::sin(half) / axis;
      px = x * k, py = y * k, pz = z * k;
    }
    double r[3][3] = {
        {1 - 2 * (py * py + pz * pz), 2 * (px * py - pz * pw), 2 * (px * pz + py * pw)},
        {2 * (px * py + pz * pw), 1 - 2 * (px * px + pz * pz), 2 * (py * pz - px * pw)},
        {2 * (px * pz - py * pw), 2 * (py * pz + px * pw), 1 - 2 * (px * px + py * py)},
    };
    // The translation: at t = 0.5 exactly half of D, (R_h + I) t_h = t, so
    // that the half applied twice is D. At another t, the screw motion's: D
    // turns about an axis through c (with c = (v + cot(angle/2) u x v) / 2
    // for the part v of its shift across the axis u) and slides along it, so
    // D^t = R^t (x - c) + c + t (the slide).
    double shiftPart[3] = {m[0][3] * t, m[1][3] * t, m[2][3] * t};
    if (t == 0.5) {
      const float sum[3][4] = {
          {float(r[0][0] + 1.0), float(r[0][1]), float(r[0][2]), 0.f},
          {float(r[1][0]), float(r[1][1] + 1.0), float(r[1][2]), 0.f},
          {float(r[2][0]), float(r[2][1]), float(r[2][2] + 1.0), 0.f},
      };
      double solve[3][4];
      if (affine_inverse(sum, solve)) {
        for (int i = 0; i < 3; ++i)
          shiftPart[i] = solve[i][0] * m[0][3] + solve[i][1] * m[1][3] + solve[i][2] * m[2][3];
      }
    } else if (axis > 1e-12 && angle > 1e-6) {
      const double u[3] = {x / axis, y / axis, z / axis};
      const double v[3] = {m[0][3], m[1][3], m[2][3]};
      const double along = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
      const double across[3] = {v[0] - along * u[0], v[1] - along * u[1], v[2] - along * u[2]};
      const double cross[3] = {u[1] * across[2] - u[2] * across[1], u[2] * across[0] - u[0] * across[2],
                               u[0] * across[1] - u[1] * across[0]};
      const double cot = 1.0 / std::tan(0.5 * angle);
      const double c[3] = {0.5 * (across[0] + cot * cross[0]), 0.5 * (across[1] + cot * cross[1]),
                           0.5 * (across[2] + cot * cross[2])};
      for (int i = 0; i < 3; ++i)
        shiftPart[i] = c[i] - (r[i][0] * c[0] + r[i][1] * c[1] + r[i][2] * c[2]) + t * along * u[i];
    }
    for (int i = 0; i < 3; ++i) {
      for (int k = 0; k < 3; ++k)
        motion.part[step][i][k] = r[i][k];
      motion.part[step][i][3] = shiftPart[i];
    }
  }
  motion.bounded = true;
  return true;
}

// A draw's own motion from one frame to the next, M_now * M_before^-1.
bool own_motion(const float previous[][4], const float current[][4], Motion& out) {
  double inverse[3][4];
  if (!affine_inverse(previous, inverse))
    return false;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 4; ++c)
      out.m[r][c] = double(current[r][0]) * inverse[0][c] + double(current[r][1]) * inverse[1][c] +
                    double(current[r][2]) * inverse[2][c];
    out.m[r][3] += current[r][3];
  }
  out.votes = 1;
  out.weight = 1.0;
  out.derived = false;
  return true;
}

// The camera's motion this frame, once enough draws agree on it, or the frame
// before's until then; null when neither is a turn.
Motion* camera_motion() {
  if (g_leadingMotion >= 0 && g_motions[g_leadingMotion].votes >= kMinMotionVotes)
    return derive(g_motions[g_leadingMotion]) ? &g_motions[g_leadingMotion] : nullptr;
  return g_havePredicted && derive(g_predicted) ? &g_predicted : nullptr;
}

// A draw's matrices as the previous frame's camera saw them: D^-1 * M. What
// is left between that and the previous frame's matrices is the object's own
// motion, which is what plausibility judges once the camera's is known (a fast
// turn moves distant scenery far on screen, not in the world).
void as_previous_camera(const Motion& camera, const float rows[][4], float out[][4]) {
  double carried[3][4];
  affine_multiply(camera.inverse, rows, carried);
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 4; ++c)
      out[r][c] = static_cast<float>(carried[r][c]);
}

bool plausible_relative(const Motion* camera, const gxc::VertexShaderConstants& current,
                        const gxc::VertexShaderConstants& previous, uint64_t usedMatrixRows, double turnLimit) {
  if (camera == nullptr || previous.projection[3][2] == 0.f || current.projection[3][2] == 0.f)
    return plausible(current, previous, usedMatrixRows, turnLimit);
  if (!plausible_projection(current.projection, previous.projection))
    return false;
  float relative[3][4];
  if (usedMatrixRows == 0) {
    as_previous_camera(*camera, current.posnormalmatrix, relative);
    return plausible_matrix(relative, previous.posnormalmatrix, turnLimit);
  }
  for (uint64_t rows = usedMatrixRows; rows != 0; rows &= rows - 1) {
    const int row = __builtin_ctzll(rows);
    if (row > 61)
      continue;
    as_previous_camera(*camera, &current.transformmatrices[row], relative);
    if (!plausible_matrix(relative, &previous.transformmatrices[row], turnLimit))
      return false;
  }
  return true;
}

// Where a draw that stands still was the frame before: M_before = D^-1 * M_now
// for the camera's motion D, and its normal matrix turned back by D's
// rotation (D is rigid, so that is its inverse transpose too). The rest
// (lights, texture matrices) stays as now.
bool stood_before(const Motion& camera, const gxc::VertexShaderConstants& current,
                  gxc::VertexShaderConstants& out) {
  double inverse[3][4];
  if (!affine_inverse(camera.m, inverse))
    return false;
  std::memcpy(&out, &current, sizeof(out));
  double placed[3][4];
  affine_multiply(inverse, current.posnormalmatrix, placed);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 4; ++c)
      out.posnormalmatrix[r][c] = static_cast<float>(placed[r][c]);
    for (int c = 0; c < 3; ++c)
      out.posnormalmatrix[3 + r][c] = static_cast<float>(inverse[r][0] * current.posnormalmatrix[3][c] +
                                                         inverse[r][1] * current.posnormalmatrix[4][c] +
                                                         inverse[r][2] * current.posnormalmatrix[5][c]);
  }
  return true;
}
gxc::VertexShaderConstants g_stoodBefore;

// --- Blending -----------------------------------------------------------------

// A straight blend of two matrices turned far apart shrinks what they draw
// halfway (to 71 percent of its size at 90 degrees) and pulls it inward. A
// linear part that turned more than about 15 degrees is blended as what it
// is, a rotation times a scale per axis (J3D's joint matrices are R * S): the
// rotation by slerp, the scales straight. A normal matrix, R * S^-1, is the
// same kind.
constexpr double kTurnedFar = 0.136; // 4 (1 - cos 15 degrees)

template <typename A, typename B>
bool turned_far(const A from[][4], const B to[][4]) {
  double difference = 0.0, size = 0.0;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      const double a = from[r][c], b = to[r][c];
      difference += (b - a) * (b - a);
      size += a * a + b * b;
    }
  }
  // size / 6 is the mean of the two k^2.
  return difference * 6.0 > kTurnedFar * size && size > 1e-12;
}

// A linear part's rotation, as a unit quaternion (w, x, y, z), and its scale
// per axis; false when it is not a rotation times a scale (sheared or
// collapsed). A mirror has its last scale negative.
template <typename A>
bool rotation_and_scale(const A m[][4], double q[4], double scale[3]) {
  double r[3][3];
  for (int c = 0; c < 3; ++c) {
    const double s = std::sqrt(double(m[0][c]) * m[0][c] + double(m[1][c]) * m[1][c] + double(m[2][c]) * m[2][c]);
    if (s < 1e-9)
      return false;
    for (int i = 0; i < 3; ++i)
      r[i][c] = m[i][c] / s;
    scale[c] = s;
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = a + 1; b < 3; ++b) {
      if (std::fabs(r[0][a] * r[0][b] + r[1][a] * r[1][b] + r[2][a] * r[2][b]) > 0.02)
        return false;
    }
  }
  const double det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) -
                     r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]) +
                     r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
  if (det < 0.0) {
    scale[2] = -scale[2];
    for (int i = 0; i < 3; ++i)
      r[i][2] = -r[i][2];
  }
  double w, x, y, z;
  const double trace = r[0][0] + r[1][1] + r[2][2];
  if (trace > 0.0) {
    const double s = std::sqrt(trace + 1.0) * 2.0;
    w = 0.25 * s;
    x = (r[2][1] - r[1][2]) / s;
    y = (r[0][2] - r[2][0]) / s;
    z = (r[1][0] - r[0][1]) / s;
  } else if (r[0][0] > r[1][1] && r[0][0] > r[2][2]) {
    const double s = std::sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0;
    w = (r[2][1] - r[1][2]) / s;
    x = 0.25 * s;
    y = (r[0][1] + r[1][0]) / s;
    z = (r[0][2] + r[2][0]) / s;
  } else if (r[1][1] > r[2][2]) {
    const double s = std::sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0;
    w = (r[0][2] - r[2][0]) / s;
    x = (r[0][1] + r[1][0]) / s;
    y = 0.25 * s;
    z = (r[1][2] + r[2][1]) / s;
  } else {
    const double s = std::sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0;
    w = (r[1][0] - r[0][1]) / s;
    x = (r[0][2] + r[2][0]) / s;
    y = (r[1][2] + r[2][1]) / s;
    z = 0.25 * s;
  }
  const double norm = std::sqrt(w * w + x * x + y * y + z * z);
  if (!(norm > 1e-9))
    return false;
  q[0] = w / norm, q[1] = x / norm, q[2] = y / norm, q[3] = z / norm;
  return true;
}

// out's linear part (the first three columns of three rows): from's and to's
// blended by t as a rotation and scales, or straight when either is not a
// rotation times a scale.
template <typename A, typename B, typename O>
void blend_turn(const A from[][4], const B to[][4], double t, O out[][4]) {
  double q0[4], q1[4], s0[3], s1[3];
  if (!rotation_and_scale(from, q0, s0) || !rotation_and_scale(to, q1, s1) || (s0[2] < 0.0) != (s1[2] < 0.0)) {
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        out[r][c] = static_cast<O>(from[r][c] + (double(to[r][c]) - from[r][c]) * t);
    return;
  }
  double dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
  if (dot < 0.0) {
    dot = -dot;
    for (double& v : q1)
      v = -v;
  }
  double w0 = 1.0 - t, w1 = t;
  if (dot < 0.9995) {
    const double angle = std::acos(dot);
    const double sine = std::sin(angle);
    w0 = std::sin((1.0 - t) * angle) / sine;
    w1 = std::sin(t * angle) / sine;
  }
  double q[4];
  double norm = 0.0;
  for (int k = 0; k < 4; ++k) {
    q[k] = w0 * q0[k] + w1 * q1[k];
    norm += q[k] * q[k];
  }
  norm = std::sqrt(norm);
  const double w = q[0] / norm, x = q[1] / norm, y = q[2] / norm, z = q[3] / norm;
  const double r[3][3] = {
      {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
      {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
      {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)},
  };
  for (int c = 0; c < 3; ++c) {
    const double scale = s0[c] + (s1[c] - s0[c]) * t;
    for (int i = 0; i < 3; ++i)
      out[i][c] = static_cast<O>(r[i][c] * scale);
  }
}

// After a straight blend of the rows into out: its linear part blended as a
// rotation when it turned far.
template <typename A, typename B, typename O>
inline void blend_linear(const A from[][4], const B to[][4], double t, O out[][4]) {
  if (turned_far(from, to))
    blend_turn(from, to, t, out);
}

inline void lerp_rows(float out[][4], const float from[][4], const float to[][4], int rows, float t) {
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < 4; ++c)
      out[r][c] = from[r][c] + (to[r][c] - from[r][c]) * t;
}

// A texture matrix whose offset jumped by more than half the texture wrapped
// (a scrolling texture going from 0.98 to 0.02); blending would run it
// backwards across the whole texture, so it keeps the current value.
inline void lerp_texture_rows(float out[][4], const float from[][4], const float to[][4], float t) {
  for (int r = 0; r < 3; ++r) {
    if (std::fabs(to[r][3] - from[r][3]) > 0.5f)
      return; // `out` already holds the current matrix
  }
  lerp_rows(out, from, to, 3, t);
}

// Each in-between frame's blend weight toward this frame: step / (steps + 1)
// from the frame before (0.5 for one; 0.25, 0.5 and 0.75 for three).
// DOL_AURORA_FRAME_INTERP_T sets the one step's (debug).
double step_weight(int step) {
  static const double weight = [] {
    const char* env = std::getenv("DOL_AURORA_FRAME_INTERP_T");
    return env != nullptr && env[0] != '\0' ? std::strtod(env, nullptr) : 0.5;
  }();
  return g_frameSteps == 1 ? weight : double(step + 1) / double(g_frameSteps + 1);
}

// Which of the ten indexed position matrices (XF rows 0-29, and their normal
// matrices) a draw reads, from used_matrix_rows(): none for a draw without
// per-vertex matrix indices, which reads only the current matrix; all of them
// for one that indexes rows past them. The rest of the out block keeps the
// current frame's values, which the draw does not read.
inline unsigned used_matrices(uint64_t rows) {
  if ((rows >> 30) != 0)
    return 0x3FFu;
  unsigned matrices = 0;
  for (int m = 0; m < 10; ++m) {
    if (((rows >> (m * 3)) & 7u) != 0)
      matrices |= 1u << m;
  }
  return matrices;
}

// Vertices the game moves itself. A model skinned on the CPU (J3D's
// J3DSkinDeform: the boat's hull) has its vertices rewritten every frame in
// world space and is drawn with the bare view matrix. Its key and matrices
// match the frame before, but the in-between frame draws this frame's
// vertices, so blending the matrices alone leaves it where it is now: under
// full sail the hull and its shadow were drawn half a frame (50 units) ahead
// of the boat's head, Link and the sail, a second hull flickering in front.
//
// The rigid motion E that carries three of the frame before's vertices onto
// this frame's (the same vertices: the key is the same index stream) makes
// the frame before's matrices M * E^-1 for these vertices, and those are
// blended. A mesh that did not move rigidly (cloth) is left as it was.
bool vertex_motion(const VertexSamples& before, const VertexSamples& now, const gxc::VertexShaderConstants& previous,
                   uint64_t rows, gxc::VertexShaderConstants& out) {
  if (std::memcmp(before.data(), now.data(), sizeof(float) * 9) == 0)
    return false;
  double p[3][3], q[3][3], moved = 0.0, extent = 0.0;
  for (int i = 0; i < 3; ++i) {
    double d = 0.0;
    for (int k = 0; k < 3; ++k) {
      p[i][k] = before[i * 3 + k];
      q[i][k] = now[i * 3 + k];
      d += (q[i][k] - p[i][k]) * (q[i][k] - p[i][k]);
    }
    moved = std::max(moved, std::sqrt(d));
  }
  if (moved < 0.01)
    return false;
  // An orthonormal frame on each triangle of samples; the rotation carries
  // one onto the other (none when the samples are in a line).
  const auto frame = [&](const double v[3][3], double f[3][3]) {
    double a[3], b[3];
    for (int k = 0; k < 3; ++k) {
      a[k] = v[1][k] - v[0][k];
      b[k] = v[2][k] - v[0][k];
    }
    const double la = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    extent = std::max(extent, la);
    if (la < 1e-3)
      return false;
    for (int k = 0; k < 3; ++k)
      a[k] /= la;
    const double along = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    for (int k = 0; k < 3; ++k)
      b[k] -= along * a[k];
    const double lb = std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
    if (lb < 1e-3 * la)
      return false;
    for (int k = 0; k < 3; ++k)
      b[k] /= lb;
    const double c[3] = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
    for (int k = 0; k < 3; ++k) { // columns a, b, c
      f[k][0] = a[k];
      f[k][1] = b[k];
      f[k][2] = c[k];
    }
    return true;
  };
  double fp[3][3], fq[3][3], r[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  if (frame(p, fp) && frame(q, fq)) {
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 3; ++k)
        r[i][k] = fq[i][0] * fp[k][0] + fq[i][1] * fp[k][1] + fq[i][2] * fp[k][2];
  }
  double cp[3], cq[3], t[3];
  for (int k = 0; k < 3; ++k) {
    cp[k] = (p[0][k] + p[1][k] + p[2][k]) / 3.0;
    cq[k] = (q[0][k] + q[1][k] + q[2][k]) / 3.0;
  }
  for (int i = 0; i < 3; ++i)
    t[i] = cq[i] - (r[i][0] * cp[0] + r[i][1] * cp[1] + r[i][2] * cp[2]);
  for (int i = 0; i < 3; ++i) {
    double d = 0.0;
    for (int k = 0; k < 3; ++k) {
      const double carried = r[k][0] * p[i][0] + r[k][1] * p[i][1] + r[k][2] * p[i][2] + t[k];
      d += (carried - q[i][k]) * (carried - q[i][k]);
    }
    if (std::sqrt(d) > 0.05 * moved + 0.01 * extent + 0.1)
      return false; // not rigid
  }
  // E^-1 = [R^T | -R^T t].
  double inverse[3][4];
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k)
      inverse[i][k] = r[k][i];
    inverse[i][3] = -(r[0][i] * t[0] + r[1][i] * t[1] + r[2][i] * t[2]);
  }
  std::memcpy(&out, &previous, sizeof(out));
  const auto place = [&](const float from[][4], float to[][4]) {
    for (int i = 0; i < 3; ++i) {
      double row[4];
      for (int c = 0; c < 4; ++c)
        row[c] = from[i][0] * inverse[0][c] + from[i][1] * inverse[1][c] + from[i][2] * inverse[2][c];
      row[3] += from[i][3];
      for (int c = 0; c < 4; ++c)
        to[i][c] = static_cast<float>(row[c]);
    }
  };
  const auto turn = [&](const float from[][4], float to[][4]) {
    for (int i = 0; i < 3; ++i) {
      double row[3];
      for (int c = 0; c < 3; ++c)
        row[c] = from[i][0] * inverse[0][c] + from[i][1] * inverse[1][c] + from[i][2] * inverse[2][c];
      for (int c = 0; c < 3; ++c)
        to[i][c] = static_cast<float>(row[c]);
    }
  };
  if (rows == 0) {
    place(previous.posnormalmatrix, out.posnormalmatrix);
    turn(&previous.posnormalmatrix[3], &out.posnormalmatrix[3]);
  } else {
    const unsigned matrices = used_matrices(rows);
    for (int m = 0; m < 10; ++m) {
      if ((matrices >> m) & 1u) {
        place(&previous.transformmatrices[m * 3], &out.transformmatrices[m * 3]);
        turn(&previous.normalmatrices[m * 3], &out.normalmatrices[m * 3]);
      }
    }
  }
  return true;
}
gxc::VertexShaderConstants g_movedPrevious;

void blend(const gxc::VertexShaderConstants& previous, const gxc::VertexShaderConstants& current, float t,
           gxc::VertexShaderConstants& out, uint64_t rows) {
  std::memcpy(&out, &current, sizeof(out));
  lerp_rows(out.posnormalmatrix, previous.posnormalmatrix, current.posnormalmatrix, 6, t);
  blend_linear(previous.posnormalmatrix, current.posnormalmatrix, t, out.posnormalmatrix);
  blend_linear(&previous.posnormalmatrix[3], &current.posnormalmatrix[3], t, &out.posnormalmatrix[3]);
  lerp_rows(out.projection, previous.projection, current.projection, 4, t);
  for (int m = 0; m < 8; ++m)
    lerp_texture_rows(&out.texmatrices[m * 3], &previous.texmatrices[m * 3], &current.texmatrices[m * 3], t);
  // XF matrix memory: position matrices in rows 0-29, texture matrices in
  // 30-59, and the identity rows after them.
  const unsigned matrices = used_matrices(rows);
  for (int m = 0; m < 10; ++m) {
    if ((matrices >> m) & 1u) {
      lerp_rows(&out.transformmatrices[m * 3], &previous.transformmatrices[m * 3],
                &current.transformmatrices[m * 3], 3, t);
      blend_linear(&previous.transformmatrices[m * 3], &current.transformmatrices[m * 3], t,
                   &out.transformmatrices[m * 3]);
      lerp_rows(&out.normalmatrices[m * 3], &previous.normalmatrices[m * 3], &current.normalmatrices[m * 3], 3, t);
      blend_linear(&previous.normalmatrices[m * 3], &current.normalmatrices[m * 3], t, &out.normalmatrices[m * 3]);
    }
  }
  for (int m = 0; m < 10; ++m) {
    const int row = 30 + m * 3;
    lerp_texture_rows(&out.transformmatrices[row], &previous.transformmatrices[row],
                      &current.transformmatrices[row], t);
  }
  lerp_rows(&out.transformmatrices[60], &previous.transformmatrices[60], &current.transformmatrices[60], 4, t);
  if ((rows >> 30) != 0)
    lerp_rows(&out.normalmatrices[30], &previous.normalmatrices[30], &current.normalmatrices[30], 2, t);
  // Lights are in view space and move with the camera.
  for (int l = 0; l < 8; ++l) {
    for (int c = 0; c < 4; ++c) {
      out.lights[l].pos[c] = previous.lights[l].pos[c] + (current.lights[l].pos[c] - previous.lights[l].pos[c]) * t;
      out.lights[l].dir[c] = previous.lights[l].dir[c] + (current.lights[l].dir[c] - previous.lights[l].dir[c]) * t;
    }
  }
}

// The in-between frame when the camera's motion is known: each matrix is the
// object's own motion blended in the previous camera's view, then carried by
// the camera's part motion. Scenery that stood still lands exactly where the
// halfway camera sees it however far the camera turned (a straight blend of
// two views turned far apart shrinks them), and what moves keeps its place
// against it.
void blend_relative(const Motion& camera, const gxc::VertexShaderConstants& previous,
                    const gxc::VertexShaderConstants& current, int step, gxc::VertexShaderConstants& out,
                    uint64_t rows) {
  const double t = step_weight(step);
  const double(*part)[4] = camera.part[step];
  blend(previous, current, float(t), out, rows);
  const auto place = [&](const float from[][4], const float to[][4], float result[][4], bool affine) {
    double seen[3][4];
    affine_multiply(camera.inverse, to, seen);
    double mid[3][4];
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 4; ++c) {
        const double value = affine || c < 3 ? seen[r][c] : double(to[r][c]);
        mid[r][c] = from[r][c] + (value - from[r][c]) * t;
      }
    }
    blend_linear(from, seen, t, mid);
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c)
        result[r][c] = static_cast<float>(part[r][0] * mid[0][c] + part[r][1] * mid[1][c] +
                                          part[r][2] * mid[2][c]);
      result[r][3] = static_cast<float>(
          affine ? part[r][0] * mid[0][3] + part[r][1] * mid[1][3] + part[r][2] * mid[2][3] +
                       part[r][3]
                 : mid[r][3]);
    }
  };
  place(previous.posnormalmatrix, current.posnormalmatrix, out.posnormalmatrix, true);
  // Normal matrices turn with the camera's rotation only (it is rigid).
  place(&previous.posnormalmatrix[3], &current.posnormalmatrix[3], &out.posnormalmatrix[3], false);
  const unsigned matrices = used_matrices(rows);
  for (int m = 0; m < 10; ++m) {
    if ((matrices >> m) & 1u) {
      place(&previous.transformmatrices[m * 3], &current.transformmatrices[m * 3], &out.transformmatrices[m * 3],
            true);
      place(&previous.normalmatrices[m * 3], &current.normalmatrices[m * 3], &out.normalmatrices[m * 3], false);
    }
  }
  // Lights in view space: positions carried, directions turned.
  for (int l = 0; l < 8; ++l) {
    double pos[3], dir[3];
    for (int r = 0; r < 3; ++r) {
      const auto& now = current.lights[l];
      pos[r] = camera.inverse[r][0] * now.pos[0] + camera.inverse[r][1] * now.pos[1] +
               camera.inverse[r][2] * now.pos[2] + camera.inverse[r][3];
      dir[r] = camera.inverse[r][0] * now.dir[0] + camera.inverse[r][1] * now.dir[1] + camera.inverse[r][2] * now.dir[2];
    }
    double midPos[3], midDir[3];
    for (int r = 0; r < 3; ++r) {
      midPos[r] = previous.lights[l].pos[r] + (pos[r] - previous.lights[l].pos[r]) * t;
      midDir[r] = previous.lights[l].dir[r] + (dir[r] - previous.lights[l].dir[r]) * t;
    }
    for (int r = 0; r < 3; ++r) {
      out.lights[l].pos[r] = static_cast<float>(part[r][0] * midPos[0] + part[r][1] * midPos[1] +
                                                part[r][2] * midPos[2] + part[r][3]);
      out.lights[l].dir[r] = static_cast<float>(part[r][0] * midDir[0] + part[r][1] * midDir[1] +
                                                part[r][2] * midDir[2]);
    }
  }
}

inline uint64_t mix64(uint64_t h) {
  h ^= h >> 33;
  h *= 0xFF51AFD7ED558CCDull;
  h ^= h >> 33;
  h *= 0xC4CEB9FE1A85EC53ull;
  h ^= h >> 33;
  return h;
}

// What blend_draw() matched its draw to: the counterpart's record and block,
// and the motion the matrices were blended in (see blend_relative()), or none
// for a straight blend. No record: nothing matched.
struct Match {
  const Record* record = nullptr;
  const gxc::VertexShaderConstants* block = nullptr;
  const Motion* motion = nullptr;
};
Match g_match;
Motion g_ownMotion; // the "blended by its own motion" path's
// The draw being matched is a particle the host tagged, and its age.
bool g_currentTagged = false;
uint32_t g_currentAge = 0;

// A draw whose positions the game sends itself (particles, the sword's
// trail): its matrices say nothing of how it moved, so the in-between frame
// drew it where the next frame has it, at 30 FPS. Each vertex is taken
// halfway from where it was drawn in the frame before to where it is now, in
// the previous camera's view and carried by the camera's part motion as the
// matrices are (blend_relative()), then back through the in-between frame's
// matrix into the space the draw's matrix takes its positions from:
//   q = Mh^-1 P ((1 - t) M0 p0 + t D^-1 M1 p1)
// for the previous and current matrices M0 and M1, the motion D and its part
// P (none: D = P = I) and the in-between matrix Mh. False when no vertex
// moves from where it is drawn now, or one moved implausibly far for a frame
// (a pair that is not the same thing).
bool blend_positions(const float* before, const float* now, uint32_t count, const float previousMatrix[][4],
                     const float currentMatrix[][4], const Motion* motion, int step, const float inBetween[][4],
                     double t, bool tagged, std::vector<float>& out) {
  double back[3][4];
  if (!affine_inverse(inBetween, back))
    return false;
  double a0[3][4], a1[3][4];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 4; ++c)
      a0[r][c] = previousMatrix[r][c];
  if (motion != nullptr) {
    affine_multiply(motion->inverse, currentMatrix, a1);
  } else {
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 4; ++c)
        a1[r][c] = currentMatrix[r][c];
  }
  double b0[3][4], b1[3][4];
  if (motion != nullptr) {
    double placed[3][4];
    affine_multiply(motion->part[step], a0, placed);
    affine_multiply(back, placed, b0);
    affine_multiply(motion->part[step], a1, placed);
    affine_multiply(back, placed, b1);
  } else {
    affine_multiply(back, a0, b0);
    affine_multiply(back, a1, b1);
  }
  const auto apply = [](const double m[3][4], const float* p, double result[3]) {
    for (int r = 0; r < 3; ++r)
      result[r] = m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2] + m[r][3];
  };
  out.resize(static_cast<size_t>(count) * 3u);
  bool moves = false;
  for (uint32_t i = 0; i < count; ++i) {
    const float* p0 = before + i * 3u;
    const float* p1 = now + i * 3u;
    // How far it moved, in the previous camera's view: a particle flies at
    // most a few hundred units a frame.
    double u0[3], u1[3];
    apply(a0, p0, u0);
    apply(a1, p1, u1);
    const double moved = std::sqrt((u1[0] - u0[0]) * (u1[0] - u0[0]) + (u1[1] - u0[1]) * (u1[1] - u0[1]) +
                                   (u1[2] - u0[2]) * (u1[2] - u0[2]));
    const double distance = std::sqrt(u0[0] * u0[0] + u0[1] * u0[1] + u0[2] * u0[2]);
    if (moved > (tagged ? 0.5 * distance + 300.0 : 0.2 * distance + 100.0)) {
      out.clear(); // implausible, not merely still
      return false;
    }
    double q0[3], q1[3];
    apply(b0, p0, q0);
    apply(b1, p1, q1);
    for (int k = 0; k < 3; ++k) {
      const double q = q0[k] + (q1[k] - q0[k]) * t;
      out[i * 3u + k] = static_cast<float>(q);
      moves = moves || std::fabs(q - p1[k]) > 1e-3 * (1.0 + std::fabs(p1[k]));
    }
  }
  return moves;
}

void log_counts() {
  const auto& c = g_totalCounts;
  std::fprintf(stderr,
               "[frame-interp] frames=%llu interpolated=%llu draws=%llu matched=%llu "
               "blended=%llu identical=%llu rejected=%llu unmatched=%llu\n",
               static_cast<unsigned long long>(g_framesSeen),
               static_cast<unsigned long long>(g_framesInterpolated), static_cast<unsigned long long>(c.draws),
               static_cast<unsigned long long>(c.matched), static_cast<unsigned long long>(c.blended),
               static_cast<unsigned long long>(c.identical), static_cast<unsigned long long>(c.rejected),
               static_cast<unsigned long long>(c.unmatched));
}

} // namespace

bool enabled() noexcept { return g_enabled.load(std::memory_order_relaxed); }
void set_enabled(bool enabled) noexcept { g_enabled.store(enabled, std::memory_order_relaxed); }

static uint64_t draw_key_of(const gxc::DrawPlan& plan) noexcept {
  if (plan.match_payload == nullptr || plan.match_payload_size == 0)
    return 0;
  if (plan.draw_scope_part != 0) {
    // One of the draws a wake's emitter makes (its fans and strips), by its
    // place among them: its vertices are the wake's particles in order (see
    // blend_positions()). Or one of a cloth's strips: the same vertices of
    // the cloth every frame.
    const uint64_t h = mix64(0x5C0Au ^ (uint64_t(plan.draw_scope) << 8) ^ (uint64_t(plan.draw_scope_part) << 40));
    return h == 0 ? 1 : h;
  }
  if (plan.match_direct_position && plan.draw_tag == 0 && plan.match_primitive == 0x90 && plan.tex_address != 0 &&
      plan.constants.projection[3][2] != 0.f) {
    // A textured list of triangles in the scene: a real shadow cast onto the
    // triangles of the ground or sea under its object, in its own texture.
    // As many as lie there: a different count every few frames as the boat
    // sails, so the draw had a new key, went unmatched and kept the next
    // frame's shadow projection in the in-between frame while the boat was
    // halfway, and the shadow flickered at its edges. Paired by its texture
    // whatever the count (its triangles are not the same points from one
    // frame to the next, so only its matrices are blended).
    const uint64_t h = mix64(0x0B7Eu ^ (uint64_t(plan.tex_address) << 16) ^ (uint64_t(plan.match_vtx_fmt) << 8) ^
                             (uint64_t(plan.texmap_mask) << 48));
    return h == 0 ? 1 : h;
  }
  uint64_t h = mix64((uint64_t(plan.match_primitive) << 40) ^ (uint64_t(plan.match_vtx_fmt) << 32) ^
                     plan.vertex_count ^ (uint64_t(plan.match_payload_size) << 48));
  if (plan.match_direct_position) {
    // New positions every frame: the same shape with the same texture, and
    // for a particle the host tagged, that particle (see blend_positions()).
    h = mix64(h ^ 0xD1u ^ (uint64_t(plan.tex_address) << 8) ^ (uint64_t(plan.texmap_mask) << 40));
    if (plan.draw_tag != 0)
      h = mix64(h ^ (uint64_t(plan.draw_tag) << 20) ^ 0x7A6u);
    return h == 0 ? 1 : h;
  }
  const uint8_t* bytes = plan.match_payload;
  size_t size = plan.match_payload_size;
  while (size >= 8) {
    uint64_t word;
    std::memcpy(&word, bytes, 8);
    h = (h ^ mix64(word)) * 0x9E3779B97F4A7C15ull;
    bytes += 8;
    size -= 8;
  }
  uint64_t tail = 0;
  std::memcpy(&tail, bytes, size);
  h = mix64(h ^ tail ^ size);
  return h == 0 ? 1 : h;
}

uint64_t draw_key(const gxc::DrawPlan& plan) noexcept { return draw_key_of(plan); }

void capture_draw(const gxc::DrawPlan& plan, DrawInput& out) noexcept {
  out.key = draw_key_of(plan);
  out.usedMatrixRows = used_matrix_rows(plan);
  out.haveSamples = false;
  out.positions.clear();
  out.age = 0;
  if (out.key == 0)
    return;
  // Only the positions of particles the host tagged: an untagged draw of one
  // shape is not the same points from one frame to the next (the boat's
  // shadow is cast on the sea's triangles under it, a different list as it
  // moves), and blending one list toward the other drew the shadow torn.
  const size_t decoded = plan.vertices.size() / gxc::kVertexFloats;
  // A cloth's strip (a scope over draws that index their positions): the
  // game moves its vertices each frame, so they are blended one by one, as a
  // particle's are.
  const bool cloth = !plan.match_direct_position && plan.draw_scope_part != 0;
  if (plan.match_direct_position || cloth) {
    if ((plan.draw_tag != 0 || plan.draw_scope_part != 0) && decoded > 0 && decoded <= kMaxBlendedVertices) {
      out.positions.resize(decoded * 3u);
      for (size_t i = 0; i < decoded; ++i)
        std::memcpy(out.positions.data() + i * 3u,
                    plan.vertices.data() + i * gxc::kVertexFloats + gxc::kVertexPosOffset / sizeof(float),
                    sizeof(float) * 3);
      out.age = plan.draw_scope_part != 0 ? 0u : plan.draw_tag_age;
    }
    return;
  }
  // Three vertices, first, middle and last, as decoded (in the space the
  // position matrix takes them from), for blend_draw() to see whether the
  // game moved them itself (see vertex_motion()).
  const uint32_t count = plan.vertex_count;
  if (count >= 3 && plan.vertices.size() >= static_cast<size_t>(count) * gxc::kVertexFloats) {
    const uint32_t picks[3] = {0, count / 2, count - 1};
    for (int i = 0; i < 3; ++i)
      std::memcpy(&out.samples[i * 3],
                  plan.vertices.data() + static_cast<size_t>(picks[i]) * gxc::kVertexFloats +
                      gxc::kVertexPosOffset / sizeof(float),
                  sizeof(float) * 3);
    out.haveSamples = true;
  }
}

uint64_t used_matrix_rows(const gxc::DrawPlan& plan) noexcept {
  if (!plan.pipeline.shader.has_pos_mtx_idx || plan.match_payload == nullptr || plan.match_vertex_stride == 0)
    return 0;
  // PNMTXIDX is the first byte of each vertex; it holds the matrix's XF row.
  uint64_t rows = 0;
  for (uint32_t offset = 0; offset < plan.match_payload_size; offset += plan.match_vertex_stride) {
    const uint8_t row = plan.match_payload[offset];
    if (row <= 61)
      rows |= 1ull << row;
  }
  return rows;
}

static const gxc::VertexShaderConstants* match_draw(uint64_t key, uint64_t usedMatrixRows,
                                                   const gxc::VertexShaderConstants& current, bool repeatsLastDraw,
                                                   const VertexSamples* samples) {
  g_outcome = "no key";
  g_blendRepeated = false;
  const uint64_t serial = ++g_drawSerial;
  if (!enabled() || key == 0)
    return nullptr;
  const bool haveSamples = samples != nullptr;
  // A model drawn into a shadow map (an orthographic projection) and into the
  // scene sends the same vertices twice. They are not copies of one another:
  // the boat's hull, one draw of each in the shadow pass and the scene, was
  // two copies of itself, matched as copies are (see below).
  const bool perspective = current.projection[3][2] != 0.f;
  if (!perspective)
    key = mix64(key ^ 0x6F7274686F677261ull) | 1u;
  const float (*here)[4] = position_signature(current, usedMatrixRows);
  FrameRecords& frame = g_frames[g_current];
  // The pool keeps a run of identical blocks once. When the draw before this
  // one put its block there, the caller's comparison says whether this is it.
  const bool lastPooled = !frame.pool.empty() && g_pooledSerial + 1 == serial;
  if (frame.pool.empty() ||
      (lastPooled ? !repeatsLastDraw : std::memcmp(&frame.pool.back(), &current, sizeof(current)) != 0))
    frame.pool.push_back(current);
  g_pooledSerial = serial;
  // An in-between block made the way the draw before this one's was (same
  // path, same previous block, same motion, same matrices) from the same
  // constants is the one g_blended holds.
  const auto repeat = [&](BlendPath path, const void* previousBlock, const void* motion) {
    // g_movedPrevious is made anew for each draw whose vertices moved.
    const bool same = previousBlock != &g_movedPrevious && repeatsLastDraw && g_lastBlend.serial + 1 == serial &&
                      g_lastBlend.path == path && g_lastBlend.previous == previousBlock &&
                      g_lastBlend.motion == motion && g_lastBlend.motionGeneration == g_motionGeneration &&
                      g_lastBlend.rows == usedMatrixRows;
    g_lastBlend.serial = serial;
    if (same)
      return true;
    g_lastBlend.path = path;
    g_lastBlend.previous = previousBlock;
    g_lastBlend.motion = motion;
    g_lastBlend.motionGeneration = g_motionGeneration;
    g_lastBlend.rows = usedMatrixRows;
    return false;
  };
  const auto constantsIndex = static_cast<uint32_t>(frame.pool.size() - 1);
  uint32_t samplesIndex = kNoSamples;
  if (haveSamples) {
    samplesIndex = static_cast<uint32_t>(frame.samples.size());
    frame.samples.push_back(*samples);
  }
  frame.records.push_back(
      {key, constantsIndex, samplesIndex, {here[0][3], here[1][3], here[2][3]}, false, kNoSamples, 0, g_currentAge});
  ++g_frameCounts.draws;
  KeyState& state = g_keys[key];
  const uint32_t occurrence = state.occurrence++;
  if (!g_havePrevious) {
    ++g_frameCounts.unmatched;
    g_outcome = "no previous frame";
    return nullptr;
  }

  auto range =
      std::equal_range(g_previousIndex.begin(), g_previousIndex.end(), Record{key, 0, kNoSamples, {}, false, 0, 0, 0},
                       [](const Record& a, const Record& b) { return a.key < b.key; });
  // A particle's address is reused for a new one once it dies: a counterpart
  // older than it, or more than two game frames younger, is another particle.
  if (g_currentTagged && range.first != range.second &&
      (range.first->age > g_currentAge || g_currentAge - range.first->age > 2u))
    range.second = range.first;
  if (range.first == range.second) {
    ++g_frameCounts.unmatched;
    g_outcome = "unmatched";
    // No counterpart: a model come into view, or geometry whose vertex
    // stream changes every frame. The sea is 64 strips of the view matrix
    // whose texture coordinates follow the player's position (the grid moves
    // with him, even as the boat bobs), so no strip keeps its key. Drawn as it
    // is now, the whole sea would stay on the real frame's camera while the
    // boat, the islands and the sky are on the in-between camera: at sea the
    // waves step at 30 FPS when the camera turns, and the boat, which the
    // camera follows, stutters against the water. What stood still in the
    // world was where the camera's motion carries it from, so it is blended
    // from there.
    Motion* const view = perspective && usedMatrixRows == 0 ? camera_motion() : nullptr;
    if (view == nullptr)
      return nullptr;
    if (repeat(BlendPath::Unmatched, view, nullptr)) {
      g_outcome = "unmatched, stood still";
      g_blendRepeated = true;
      return &g_blended[0];
    }
    if (!stood_before(*view, current, g_stoodBefore)) {
      g_lastBlend.serial = 0;
      return nullptr;
    }
    for (int step = 0; step < g_frameSteps; ++step)
      blend_relative(*view, g_stoodBefore, current, step, g_blended[step], 0);
    g_outcome = "unmatched, stood still";
    return &g_blended[0];
  }
  g_rejectReason = "";
  const FrameRecords& previousFrame = g_frames[g_current ^ 1u];
  const Record* copies = &*range.first;
  const gxc::VertexShaderConstants* movedPrevious = nullptr; // copy 0 as this frame's vertices see it
  const auto constantsOf = [&](size_t copy) -> const gxc::VertexShaderConstants& {
    return copy == 0 && movedPrevious != nullptr ? *movedPrevious : previousFrame.pool[copies[copy].constants];
  };
  const size_t candidates = static_cast<size_t>(range.second - range.first);
  const double turnLimit = candidates == 1 && occurrence == 0 ? kOwnTurnLimit : kTurnLimit;
  const gxc::VertexShaderConstants* previous = nullptr;
  size_t chosen = 0; // previous's copy
  // A draw with a key of its own is the same object as its counterpart, so
  // it votes for the camera's motion whether or not the pair looks plausible
  // yet: a fast turn makes the scenery implausible until the camera's motion
  // is known. So do copies that all had one block (effects drawn with the
  // view matrix, like the wave crests at sea): whichever this one was, it
  // moved the same way. Their first copy votes, as the key's one draw would.
  const bool uniformCopies = candidates > 1 && copies[0].constants == copies[candidates - 1].constants;
  // Not a tagged particle: a key of its own each, and a billboard's matrix is
  // the identity, whose votes would outweigh the camera's.
  if ((candidates == 1 || uniformCopies) && occurrence == 0 && usedMatrixRows == 0 && perspective && !g_currentTagged &&
      constantsOf(0).projection[3][2] != 0.f)
    vote_motion(current.posnormalmatrix, constantsOf(0).posnormalmatrix);
  // A draw of its own whose vertices the game moved itself: the in-between
  // frame keeps this frame's vertices, so the frame before's matrices are
  // taken as they would place them (see vertex_motion()).
  if (candidates == 1 && occurrence == 0 && samplesIndex != kNoSamples && copies[0].samples != kNoSamples &&
      vertex_motion(previousFrame.samples[copies[0].samples], frame.samples[samplesIndex],
                    previousFrame.pool[copies[0].constants], usedMatrixRows, g_movedPrevious)) {
    movedPrevious = &g_movedPrevious;
    ++g_frameCounts.moved;
  }
  Motion* const view = perspective ? camera_motion() : nullptr;

  // Copies of one model (grass, bushes, trees): the copy that a motion several
  // draws share (the camera's, for anything standing still) carries onto
  // this one, within float rounding. The same place in the list, or the copy
  // after the last one carried (when copies before it left the view), usually
  // is; otherwise the closest.
  const Motion* motions[kMaxMotions];
  int motionCount = 0;
  const Motion* camera = nullptr;
  if (perspective && candidates > 1 && g_leadingMotion >= 0 && g_motions[g_leadingMotion].votes >= kMinMotionVotes) {
    camera = &g_motions[g_leadingMotion];
    motions[motionCount++] = camera;
    for (int i = 0; i < g_motionCount; ++i) {
      if (i != g_leadingMotion && g_motions[i].votes >= kMinMotionVotes)
        motions[motionCount++] = &g_motions[i];
    }
  }
  const double tolerance = 1.0 + 2e-5 * translation_length(here);
  const auto carried = [&](size_t copy) {
    for (int m = 0; m < motionCount; ++m) {
      if (motion_error(*motions[m], here, copies[copy].position) <= tolerance)
        return true;
    }
    return false;
  };
  // A copy the same size as this draw to within two percent, the nearest of
  // those to it in the view and near enough for a frame, is this one, and may
  // turn as far as a draw of its own. A broken pot's shards are one model
  // drawn at random sizes, flying apart and tumbling faster than copies are
  // let turn (kTurnLimit), so each was drawn where the next frame has it.
  size_t nearestCopy = SIZE_MAX;
  const auto size_of = [&](const float(*m)[4]) {
    double size = 0.0;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        size += double(m[r][c]) * m[r][c];
    return size;
  };
  const auto same_copy = [&](size_t copy) {
    if (nearestCopy == SIZE_MAX) {
      nearestCopy = candidates;
      const double size = size_of(here);
      double best = HUGE_VAL;
      for (size_t i = 0; i < std::min<size_t>(candidates, 256); ++i) {
        const double other = size_of(position_signature(constantsOf(i), usedMatrixRows));
        if (!(other > 1e-12) || std::fabs(size / other - 1.0) >= 0.04) // squared sizes: 2 percent
          continue;
        const double d = translation_distance2(here, copies[i].position);
        if (d < best) {
          best = d;
          nearestCopy = i;
        }
      }
    }
    const double reach = 40.0 + 0.01 * translation_length(here);
    return copy == nearestCopy && translation_distance2(here, copies[copy].position) <= reach * reach;
  };
  const auto plausible_copy = [&](size_t copy) {
    if (plausible_relative(view, current, constantsOf(copy), usedMatrixRows, turnLimit))
      return true;
    return turnLimit < kOwnTurnLimit && same_copy(copy) &&
           plausible_relative(view, current, constantsOf(copy), usedMatrixRows, kOwnTurnLimit);
  };
  size_t pick = candidates;
  // How far the camera's motion puts the nearest copy that stood still.
  double nearestStill = HUGE_VAL;
  if (camera != nullptr) {
    if (occurrence < candidates && carried(occurrence))
      pick = occurrence;
    else if (state.next < candidates && carried(state.next))
      pick = state.next;
    else if (constantsIndex == g_newInView)
      nearestStill = 0.0; // another part of a copy just found new in view
    else {
      const size_t first = static_cast<size_t>(range.first - g_previousIndex.begin());
      grid_copies(key, *camera, first, candidates);
      pick = find_carried(key, here, first, candidates, tolerance, nearestStill);
    }
  }
  const bool placed = pick < candidates;
  if (placed) {
    state.next = static_cast<uint32_t>(pick + 1);
    frame.records.back().still = motion_error(*camera, here, copies[pick].position) <= tolerance;
    const auto& candidate = constantsOf(pick);
    if (plausible_copy(pick)) {
      previous = &candidate;
      chosen = pick;
    }
  } else {
    // Without such a copy, a copy that stood still would have been carried
    // here had it been this one, unless it has just started to move (an idle
    // character swaying past the tolerance). Among copies that stood still
    // (grass, pots), one that moved is this draw only if it moved at an
    // actor's pace (a pot Link carries) and no copy that stood still is
    // nearer; otherwise this draw is a copy that has just come into view.
    // A few copies none of which stood still are one moving model drawn more
    // than once (the same parts in two passes): they move as fast as it
    // does, the boat under full sail 100 units a frame, so a plausible one
    // is usable at any pace.
    bool fewMoving = candidates <= kFewCopies;
    for (size_t i = 0; fewMoving && i < candidates; ++i)
      fewMoving = !copies[i].still;
    const auto usable = [&](size_t copy) {
      if (camera == nullptr)
        return true;
      const double moved = motion_error(*camera, here, copies[copy].position);
      if (copies[copy].still)
        return moved <= 4.0 * tolerance;
      // One that kept its place in the view: it moves with the camera, which
      // follows it (the boat's shadow volume, the same box for every real
      // shadow, as the boat sails 60 units a frame).
      const double reach = 40.0 + 0.01 * translation_length(here);
      if (translation_distance2(here, copies[copy].position) <= reach * reach)
        return true;
      // A copy that moved at an actor's pace (a pot Link carries), when no
      // copy that stood still is nearer.
      if (!(moved < nearestStill))
        return false;
      return moved <= 40.0 + 0.01 * translation_length(here) ||
             (fewMoving && plausible_relative(view, current, constantsOf(copy), usedMatrixRows, turnLimit));
    };
    bool anyUsable = false;
    // The same occurrence of the key: a model drawn from a fixed list (a
    // model's materials, moving copies) keeps its order.
    if (occurrence < candidates && usable(occurrence)) {
      anyUsable = true;
      const auto& candidate = constantsOf(occurrence);
      if (plausible_copy(occurrence)) {
        previous = &candidate;
        chosen = occurrence;
      }
    }
    if (previous == nullptr && candidates > 1) {
      // Copies that changed places (a depth-sorted list): the nearest plausible.
      const size_t limit = std::min<size_t>(candidates, 256);
      double bestDistance = 0.0;
      for (size_t i = 0; i < limit; ++i) {
        if (!usable(i))
          continue;
        anyUsable = true;
        const double d = translation_distance2(here, copies[i].position);
        if ((previous == nullptr || d < bestDistance) && plausible_copy(i)) {
          previous = &constantsOf(i);
          chosen = i;
          bestDistance = d;
        }
      }
    }
    if (camera != nullptr && !anyUsable) {
      // Counted with the rejected draws: after a camera cut, copies end here
      // instead of failing the plausibility test.
      g_newInView = constantsIndex;
      ++g_frameCounts.rejected;
      ++g_frameCounts.fresh;
      g_outcome = "new in view";
      // Drawn as it is now, it would sit half a camera step ahead of the
      // copies beside it (a pop along the screen's edge while the camera
      // turns). It stood where the camera's motion carries it from, so it is
      // blended from there.
      if (usedMatrixRows != 0 || !stood_before(*camera, current, g_stoodBefore) ||
          !plausible_relative(view, current, g_stoodBefore, 0, kTurnLimit))
        return nullptr;
      if (repeat(BlendPath::StoodBefore, camera, view)) {
        g_blendRepeated = true;
        return &g_blended[0];
      }
      for (int step = 0; step < g_frameSteps; ++step) {
        if (view != nullptr)
          blend_relative(*view, g_stoodBefore, current, step, g_blended[step], 0);
        else
          blend(g_stoodBefore, current, float(step_weight(step)), g_blended[step], 0);
      }
      return &g_blended[0];
    }
  }
  if (previous == nullptr && candidates == 1 && occurrence == 0 && usedMatrixRows == 0 && perspective &&
      constantsOf(0).projection[3][2] != 0.f &&
      plausible_projection(current.projection, constantsOf(0).projection)) {
    // A draw with a key of its own is the same object as its counterpart.
    // Before enough of the frame's draws agree on the camera's motion (the
    // sky, the first rooms of a fast turn), one whose change is rigid and a
    // turn rather than a cut is blended by half of its own motion, which for
    // scenery is the camera's.
    ++g_frameCounts.matched;
    if (repeat(BlendPath::OwnMotion, &constantsOf(0), nullptr)) {
      ++g_frameCounts.blended;
      g_outcome = "blended by its own motion";
      g_blendRepeated = true;
      g_match = {&copies[0], &constantsOf(0), &g_ownMotion}; // made for the same draw before this one
      return &g_blended[0];
    }
    Motion& own = g_ownMotion;
    own = Motion{};
    if (own_motion(constantsOf(0).posnormalmatrix, current.posnormalmatrix, own) && derive(own)) {
      g_match = {&copies[0], &constantsOf(0), &g_ownMotion};
      for (int step = 0; step < g_frameSteps; ++step)
        blend_relative(own, constantsOf(0), current, step, g_blended[step], usedMatrixRows);
      ++g_frameCounts.blended;
      g_outcome = "blended by its own motion";
      return &g_blended[0];
    }
    --g_frameCounts.matched;
    g_lastBlend.serial = 0;
  }
  if (previous == nullptr) {
    ++g_frameCounts.rejected;
    g_outcome = g_rejectReason;
    return nullptr;
  }
  ++g_frameCounts.matched;
  const bool relative = view != nullptr && previous->projection[3][2] != 0.f;
  g_match = {&copies[chosen], previous, relative ? view : nullptr};
  if (repeat(BlendPath::Main, previous, relative ? view : nullptr)) {
    if (g_lastBlend.identical) {
      ++g_frameCounts.identical;
      g_outcome = "identical";
      return nullptr;
    }
    ++g_frameCounts.blended;
    g_outcome = "blended";
    g_blendRepeated = true;
    return &g_blended[0];
  }
  g_lastBlend.identical = std::memcmp(previous, &current, sizeof(current)) == 0;
  if (g_lastBlend.identical) {
    ++g_frameCounts.identical;
    g_outcome = "identical";
    return nullptr;
  }
  for (int step = 0; step < g_frameSteps; ++step) {
    if (relative)
      blend_relative(*view, *previous, current, step, g_blended[step], usedMatrixRows);
    else
      blend(*previous, current, float(step_weight(step)), g_blended[step], usedMatrixRows);
  }
  ++g_frameCounts.blended;
  g_outcome = "blended";
  return &g_blended[0];
}

const gxc::VertexShaderConstants* blend_draw(const DrawInput& input, const gxc::VertexShaderConstants& current,
                                             bool repeatsLastDraw) {
  g_haveBlendedPositions = false;
  g_match = {};
  const bool havePositions = !input.positions.empty();
  const uint32_t count = static_cast<uint32_t>(input.positions.size() / 3u);
  g_currentTagged = havePositions;
  g_currentAge = havePositions ? input.age : 0u;
  FrameRecords& frame = g_frames[g_current];
  const size_t recorded = frame.records.size();
  const gxc::VertexShaderConstants* result = match_draw(input.key, input.usedMatrixRows, current, repeatsLastDraw,
                                                        input.haveSamples ? &input.samples : nullptr);
  g_currentTagged = false;
  g_currentAge = 0;
  // A particle's positions: kept for the next frame, and blended when the
  // draw was matched (in 3D or 2D, with one matrix).
  if (!havePositions || frame.records.size() == recorded || input.usedMatrixRows != 0)
    return result;
  Record& record = frame.records.back();
  record.positions = static_cast<uint32_t>(frame.positions.size() / 3u);
  record.positionCount = count;
  frame.positions.insert(frame.positions.end(), input.positions.begin(), input.positions.end());
  const Record* before = g_match.record;
  if (before == nullptr || g_match.block == nullptr || before->positions == kNoSamples ||
      before->positionCount != count)
    return result;
  const FrameRecords& previousFrame = g_frames[g_current ^ 1u];
  const float* now = frame.positions.data() + static_cast<size_t>(record.positions) * 3u;
  const float* then = previousFrame.positions.data() + static_cast<size_t>(before->positions) * 3u;
  // Each step's, through that step's in-between matrix; none unless every
  // step's is plausible and one moves.
  bool moves = false;
  for (int step = 0; step < g_frameSteps; ++step) {
    const float(*inBetween)[4] = result != nullptr ? g_blended[step].posnormalmatrix : current.posnormalmatrix;
    const bool moved = blend_positions(then, now, count, g_match.block->posnormalmatrix, current.posnormalmatrix,
                                       g_match.motion, step, inBetween, step_weight(step), true,
                                       g_blendedPositions[step]);
    if (!moved && g_blendedPositions[step].size() != static_cast<size_t>(count) * 3u)
      return result; // implausible
    moves = moves || moved;
  }
  g_haveBlendedPositions = moves;
  if (g_haveBlendedPositions)
    ++g_frameCounts.positions;
  return result;
}

const gxc::VertexShaderConstants* blend_draw(uint64_t key, uint64_t usedMatrixRows,
                                             const gxc::VertexShaderConstants& current, bool repeatsLastDraw) {
  DrawInput input;
  input.key = key;
  input.usedMatrixRows = usedMatrixRows;
  return blend_draw(input, current, repeatsLastDraw);
}

const float* blended_positions(int step) noexcept {
  return g_haveBlendedPositions && step < g_frameSteps ? g_blendedPositions[step].data() : nullptr;
}

const gxc::VertexShaderConstants* blended_step(int step) noexcept {
  return &g_blended[std::clamp(step, 0, kMaxSteps - 1)];
}

int frame_steps() noexcept { return g_frameSteps; }
void set_steps(int steps) noexcept { g_steps.store(std::clamp(steps, 1, kMaxSteps), std::memory_order_relaxed); }
int steps() noexcept { return g_steps.load(std::memory_order_relaxed); }
bool frame_skipped() noexcept { return g_frameSkipped; }

std::atomic<bool> g_cutRequested{false};
void request_cut() noexcept { g_cutRequested.store(true, std::memory_order_release); }
void note_overload(const char* why) noexcept {
  g_overloadWhy.store(why, std::memory_order_relaxed);
  g_overloaded.store(true, std::memory_order_relaxed);
}

// The game itself below full speed: the median gap between its last 60
// frames (2 s) over kSlowGapMs (under 28.2 a second against the game's
// 29.97) at two checks in a row, a second apart. On a CPU with few cores the
// in-between frames' work (each draw captured here and matched on the
// helper thread, the render worker drawing them again) takes cores the
// game's own thread needs, and the game ran in slow motion: 24-27 game
// frames a second on 4 of the i9's E-cores at 60 FPS, 30 with frame
// interpolation off. A gap of 150 ms or more (a pipeline compiled, a file
// loaded) is a hitch, not the game's speed, and is left out; the median and
// the second check keep one hitch, or a burst of a few, from reading as slow
// (an average of the last 8 gaps did: one 150 ms hitch held it over the
// limit for 13 frames and dropped the in-between frames for at least 3 s).
// The way DeepSea's frame60 governor judges speed (mod.c, 2026-09).
static bool slow_game(Pacing& p) {
  static constexpr double kSlowGapMs = 35.5;
  static constexpr double kHitchMs = 150.0;
  static constexpr int kCheckEvery = 30;
  const auto now = std::chrono::steady_clock::now();
  const auto last = p.lastEnd;
  p.lastEnd = now;
  if (last == std::chrono::steady_clock::time_point{})
    return false;
  const double ms = std::chrono::duration<double, std::milli>(now - last).count();
  if (ms < kHitchMs) {
    p.gapRing[static_cast<size_t>(p.gapNext)] = static_cast<float>(ms);
    p.gapNext = (p.gapNext + 1) % static_cast<int>(p.gapRing.size());
    p.gaps = std::min(p.gaps + 1, static_cast<int>(p.gapRing.size()));
  }
  if (++p.sinceCheck >= kCheckEvery && p.gaps == static_cast<int>(p.gapRing.size())) {
    p.sinceCheck = 0;
    std::array<float, 60> sorted = p.gapRing;
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
    p.gapMs = sorted[sorted.size() / 2];
    p.slowChecks = p.gapMs > kSlowGapMs ? p.slowChecks + 1 : 0;
  }
  return p.slowChecks >= 2;
}

// Recording thread, between game frames: the next frame's in-between frames.
// Overloads (the GPU a frame behind, or the game waiting on rendering) in 3 of
// the last 8 game frames take 60 Hz to none (and 120 Hz to 60, when pacing
// is asked for there); one now and then (a texture loading, a late drawable)
// is not a scene too heavy, and dropping on each one put 120 Hz at 60 most
// times it did not need to be. The frames in
// flight still report the overload a drop answers, so a second drop waits a
// few frames. Calm for 3 s brings one level back, and a level that overloads
// again soon after coming back waits twice as long the next time (up to 30 s),
// so a scene just too heavy for 120 settles at 60 instead of stuttering
// between them.
static void pace_steps() {
  static constexpr int kCalmStart = 90;  // game frames, 3 s
  static constexpr int kCalmMost = 900;  // 30 s
  static constexpr int kCalmMostSlow = 3600; // 2 min, after the game itself ran slow
  static constexpr uint64_t kSettle = 4; // game frames between drops
  static constexpr uint64_t kSoon = 300; // an overload within 10 s of coming back
  // DOL_AURORA_FRAME_INTERP_PACING: 0 never lowers them, 1 lowers 120 Hz's
  // too. Unset, only 60 Hz's are: 120 Hz stays 120, where a drop to 60 for a
  // stall of a few frames, over by the time it dropped, was felt more than the
  // stall.
  static const int setting = [] {
    const char* env = std::getenv("DOL_AURORA_FRAME_INTERP_PACING");
    return env != nullptr && env[0] != '\0' ? (env[0] != '0' ? 1 : 0) : -1;
  }();
  auto& p = g_pacing;
  ++p.frame;
  const int wanted = g_steps.load(std::memory_order_relaxed);
  const bool pacing = setting == 1 || (setting < 0 && wanted < 3);
  // The game running slow counts at any number of in-between frames (slow
  // motion is worse than fewer of them) unless pacing is off altogether.
  const bool slow = slow_game(p) && setting != 0;
  const bool overloaded = (g_overloaded.exchange(false, std::memory_order_relaxed) && pacing) || slow;
  p.recent = (p.recent << 1) | (overloaded ? 1u : 0u);
  const bool sustained = std::popcount(p.recent & 0xFFu) >= 3;
  if (wanted != p.wanted) {
    p = Pacing{.frame = p.frame, .wanted = wanted, .budget = wanted, .calmNeeded = kCalmStart,
               .lastEnd = p.lastEnd};
  } else if (sustained) {
    p.calm = 0;
    if (p.budget > 0 && p.frame - p.lastDrop >= kSettle) {
      if (p.lastRaise != 0 && p.frame - p.lastRaise < kSoon)
        p.calmNeeded = std::min(p.calmNeeded * 2, slow ? kCalmMostSlow : kCalmMost);
      const int before = p.budget;
      p.budget = p.budget > 1 ? 1 : 0;
      p.lastDrop = p.frame;
      p.recent = 0;
      const char* why = g_overloadWhy.load(std::memory_order_relaxed);
      if (slow) {
        std::fprintf(stderr,
                     "[interp-pace] in-between frames %d -> %d (the game ran below full speed, %.1f game "
                     "frames a second; back after %.0f s calm)\n",
                     before, p.budget, 1000.0 / p.gapMs, p.calmNeeded / 30.0);
        // The next drop on new frames' gaps.
        p.gaps = 0;
        p.sinceCheck = 0;
        p.slowChecks = 0;
      } else {
        std::fprintf(stderr, "[interp-pace] in-between frames %d -> %d (%s; back after %.0f s calm)\n", before,
                     p.budget, why != nullptr ? why : "rendering fell behind", p.calmNeeded / 30.0);
      }
    }
  } else if (p.budget < wanted) {
    if (++p.calm >= p.calmNeeded) {
      const int before = p.budget;
      p.budget = p.budget == 0 ? 1 : wanted;
      p.calm = 0;
      p.lastRaise = p.frame;
      std::fprintf(stderr, "[interp-pace] in-between frames %d -> %d\n", before, p.budget);
    }
  } else if (p.lastRaise != 0 && p.frame - p.lastRaise >= kSoon) {
    // Held since coming back: the next overload is a new scene's.
    p.calmNeeded = kCalmStart;
    p.lastRaise = 0;
  }
  g_frameSteps = std::clamp(std::min(p.budget, wanted), 1, kMaxSteps);
  g_frameSkipped = p.budget == 0;
}

bool last_blend_repeated() noexcept { return g_blendRepeated; }

bool frame_verdict() noexcept {
  g_lastVerdict = false;
  if (g_cutRequested.exchange(false, std::memory_order_acq_rel))
    return false;
  if (!enabled() || g_frameSkipped || !g_havePrevious || g_frameCounts.blended == 0)
    return false;
  // A cut: most draws have a counterpart that is not plausibly the same thing.
  // Copies just come into view count too (after a cut every copy of a model
  // looks new), unless the frame's camera motion is a turn: a fast turn can
  // bring a whole field of grass into view at once.
  const uint64_t rejected =
      g_frameCounts.rejected - (camera_motion() != nullptr ? g_frameCounts.fresh : 0u);
  const uint64_t considered = g_frameCounts.matched + rejected;
  if (considered >= 16 && rejected * 10 > considered * 4)
    return false;
  g_lastVerdict = true;
  return true;
}

void end_game_frame() noexcept {
  g_gameFrame.fetch_add(1, std::memory_order_relaxed);
  if (!enabled()) {
    if (g_havePrevious) {
      g_frames[0].clear();
      g_frames[1].clear();
      g_previousIndex.clear();
      g_keys.clear();
      g_motionCount = 0;
      g_leadingMotion = -1;
      g_havePredicted = false;
      g_havePrevious = false;
    }
    g_frameCounts = {};
    return;
  }
  ++g_framesSeen;
  const int g_lastVerdictSeen = g_lastVerdict ? 1 : 0;
  if (g_lastVerdict)
    ++g_framesInterpolated;
  g_lastVerdict = false;
  g_totalCounts.draws += g_frameCounts.draws;
  g_totalCounts.matched += g_frameCounts.matched;
  g_totalCounts.identical += g_frameCounts.identical;
  g_totalCounts.blended += g_frameCounts.blended;
  g_totalCounts.rejected += g_frameCounts.rejected;
  g_totalCounts.unmatched += g_frameCounts.unmatched;
  if (g_log && g_framesSeen % 60 == 0)
    log_counts();
  if (g_logFrames) {
    const Motion* camera = camera_motion();
    double angle = 0.0;
    if (camera != nullptr)
      angle = std::acos(std::clamp((camera->m[0][0] + camera->m[1][1] + camera->m[2][2] - 1.0) / 2.0, -1.0, 1.0)) *
              180.0 / M_PI;
    std::fprintf(stderr,
                 "[frame-interp-frame] frame=%llu verdict=%d draws=%llu blended=%llu rejected=%llu unmatched=%llu "
                 "camera=%s turn=%.1f votes=%u fresh=%llu moved=%llu positions=%llu\n",
                 static_cast<unsigned long long>(g_gameFrame.load(std::memory_order_relaxed)), g_lastVerdictSeen,
                 static_cast<unsigned long long>(g_frameCounts.draws),
                 static_cast<unsigned long long>(g_frameCounts.blended),
                 static_cast<unsigned long long>(g_frameCounts.rejected),
                 static_cast<unsigned long long>(g_frameCounts.unmatched),
                 camera == nullptr ? "none" : (camera == &g_predicted ? "predicted" : "voted"), angle,
                 camera == nullptr ? 0u : camera->votes, static_cast<unsigned long long>(g_frameCounts.fresh),
                 static_cast<unsigned long long>(g_frameCounts.moved),
                 static_cast<unsigned long long>(g_frameCounts.positions));
  }
  g_frameCounts = {};

  const FrameRecords& finished = g_frames[g_current];
  g_previousIndex.assign(finished.records.begin(), finished.records.end());
  std::stable_sort(g_previousIndex.begin(), g_previousIndex.end(),
                   [](const Record& a, const Record& b) { return a.key < b.key; });
  g_havePrevious = !finished.records.empty();
  g_current ^= 1u;
  g_frames[g_current].clear();
  // A large table's clear() frees it, and the next frame grew it back through
  // a dozen rehashes; room for as many keys as this frame had is one allocation.
  const size_t keys = g_keys.size();
  g_keys.clear();
  g_keys.reserve(keys);
  g_cellHead.clear();
  g_griddedKeys.clear();
  g_havePredicted = g_leadingMotion >= 0 && g_motions[g_leadingMotion].votes >= kMinMotionVotes;
  if (g_havePredicted) {
    g_predicted = g_motions[g_leadingMotion];
    g_predicted.derived = false; // its parts again, for the next frame's steps
  }
  pace_steps();
  g_motionCount = 0;
  g_leadingMotion = -1;
  g_newInView = UINT32_MAX;
}

bool encoding_interpolated() noexcept { return g_encodingInterpolated.load(std::memory_order_relaxed); }
void set_encoding_interpolated(bool value) noexcept {
  g_encodingInterpolated.store(value, std::memory_order_relaxed);
}

const char* dump_directory() noexcept {
  static const char* directory = std::getenv("DOL_AURORA_FRAME_INTERP_DUMP");
  return directory != nullptr && directory[0] != '\0' ? directory : nullptr;
}

bool dump_frame(uint64_t frame) noexcept {
  static const uint64_t from = env_u64("DOL_AURORA_FRAME_INTERP_DUMP_FROM", 0);
  static const uint64_t to = env_u64("DOL_AURORA_FRAME_INTERP_DUMP_TO", 0);
  return dump_directory() != nullptr && frame >= from && frame <= to;
}

const char* last_outcome() noexcept { return g_outcome; }
bool tracing() noexcept {
  const uint64_t frame = g_gameFrame.load(std::memory_order_relaxed);
  return g_trace.first != 0 && frame >= g_trace.first && frame <= g_trace.last;
}

uint64_t game_frame_number() noexcept { return g_gameFrame.load(std::memory_order_relaxed); }

} // namespace aurora::gfx::frame_interp

void aurora_set_frame_interpolation(bool enabled) { aurora::gfx::frame_interp::set_enabled(enabled); }
// Read from another thread for diagnostics; each value is one aligned word.
void aurora_get_frame_interp_totals(AuroraFrameInterpTotals* out) {
  using namespace aurora::gfx::frame_interp;
  out->frames = g_framesSeen;
  out->interpolated = g_framesInterpolated;
  out->draws = g_totalCounts.draws;
  out->rejected = g_totalCounts.rejected;
  out->unmatched = g_totalCounts.unmatched;
}
bool aurora_get_frame_interpolation() { return aurora::gfx::frame_interp::enabled(); }
void aurora_set_frame_interp_steps(int steps) { aurora::gfx::frame_interp::set_steps(steps); }
int aurora_get_frame_interp_steps() { return aurora::gfx::frame_interp::steps(); }
