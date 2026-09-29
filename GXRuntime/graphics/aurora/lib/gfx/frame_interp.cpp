#include "frame_interp.hpp"

#include <algorithm>
#include <array>
#include <atomic>
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
std::atomic_bool g_encodingInterpolated{false};
std::atomic<uint64_t> g_gameFrame{1};

// One game frame's draws: their constants (a consecutive run of identical
// blocks is stored once) and, per draw, its key and block, where its position
// matrix put it, and whether it stood still (the camera's motion carried a
// copy from the frame before onto it; see blend_draw).
struct Record {
  uint64_t key;
  uint32_t constants;
  bool still;
  float position[3];
};
struct FrameRecords {
  std::vector<gxc::VertexShaderConstants> pool;
  std::vector<Record> records;
  void clear() {
    pool.clear();
    records.clear();
  }
};

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
gxc::VertexShaderConstants g_blended;

// Most draws repeat the constants of the draw before them (a model's parts,
// copies drawn with one matrix): 96 percent in the Forsaken Fortress, where a
// frame has 17,500 draws. For those the caller's comparison is enough: the
// pool entry is not compared again, and an in-between block made from the same
// match the same way is the one g_blended already holds.
uint64_t g_drawSerial = 0;
uint64_t g_pooledSerial = 0; // the draw whose constants are the current pool's last
uint64_t g_motionGeneration = 0; // bumped when any motion's values change
enum class BlendPath { Main, OwnMotion, StoodBefore };
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
// draws matched without ambiguity, and how many draws agree with each.
struct Motion {
  double m[3][4];
  uint32_t votes;
  // For the camera's motion (see camera_motion()): its inverse, and the part
  // of it the in-between frame is at (D^t: the rotation turned by t of its
  // angle, and exactly half of D at t = 0.5, so D^0.5 * D^0.5 = D).
  bool derived;
  bool bounded;
  double inverse[3][4];
  double part[3][4];
};
constexpr int kMaxMotions = 6;
constexpr uint32_t kMinMotionVotes = 4;
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
};
Counts g_frameCounts;
Counts g_totalCounts;
uint64_t g_framesSeen = 0;
uint64_t g_framesInterpolated = 0;
bool g_lastVerdict = false;

const bool g_log = env_flag("DOL_AURORA_FRAME_INTERP_LOG", false);
// DOL_AURORA_FRAME_INTERP_LOG_FRAMES: a line per game frame (debug).
const bool g_logFrames = env_flag("DOL_AURORA_FRAME_INTERP_LOG_FRAMES", false);
const uint64_t g_traceFrame = env_u64("DOL_AURORA_FRAME_INTERP_TRACE", 0);
const char* g_outcome = "";
const char* g_rejectReason = "";

// --- Matching -----------------------------------------------------------------

double translation_length(const float rows[][4]) {
  return std::sqrt(double(rows[0][3]) * rows[0][3] + double(rows[1][3]) * rows[1][3] +
                   double(rows[2][3]) * rows[2][3]);
}

// Whether two 3x4 matrices can be one object's transform a frame apart. A
// camera cut or a reused draw for another object fails at least one bound:
// scale within 1.5x, the linear part within about 40 degrees of rotation, and
// the translation within a fifth of its distance from the camera plus 100
// units (a camera turning 10 degrees a frame moves a point about 0.17 of its
// distance; the boat at full sail covers about 60 units a frame).
bool plausible_matrix(const float current[][4], const float previous[][4]) {
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
  if (difference / axisScale > 1.0) {
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
               uint64_t usedMatrixRows) {
  if (!plausible_projection(current.projection, previous.projection))
    return false;
  if (usedMatrixRows == 0)
    return plausible_matrix(current.posnormalmatrix, previous.posnormalmatrix);
  // Per-vertex matrices (skinned models): the bank slots this draw's vertices
  // use. The other slots hold whatever other models loaded last.
  for (uint64_t rows = usedMatrixRows; rows != 0; rows &= rows - 1) {
    const int row = __builtin_ctzll(rows);
    if (!plausible_matrix(&current.transformmatrices[row], &previous.transformmatrices[row]))
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

void vote_motion(const float current[][4], const float previous[][4]) {
  int agreed = -1;
  if (g_leadingMotion >= 0 && motion_agrees(g_motions[g_leadingMotion].m, current, previous))
    agreed = g_leadingMotion;
  for (int i = 0; agreed < 0 && i < g_motionCount; ++i) {
    if (i != g_leadingMotion && motion_agrees(g_motions[i].m, current, previous))
      agreed = i;
  }
  if (agreed >= 0) {
    ++g_motions[agreed].votes;
  } else {
    double inverse[3][4];
    if (!affine_inverse(previous, inverse))
      return;
    // A new motion, in place of the one fewest draws agreed with.
    agreed = 0;
    if (g_motionCount < kMaxMotions) {
      agreed = g_motionCount++;
    } else {
      for (int i = 1; i < kMaxMotions; ++i) {
        if (g_motions[i].votes < g_motions[agreed].votes)
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
    g_motions[agreed].derived = false;
    ++g_motionGeneration;
  }
  // The motion most draws agree with is the camera's.
  if (g_leadingMotion < 0 || g_motions[agreed].votes > g_motions[g_leadingMotion].votes)
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

float blend_weight();

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
  // The rotation by t of the angle about the same axis.
  const double t = blend_weight();
  const double axis = std::sqrt(x * x + y * y + z * z);
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
  // that the half applied twice is D; otherwise t of it.
  double shiftPart[3] = {m[0][3] * t, m[1][3] * t, m[2][3] * t};
  if (t == 0.5f) {
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
  }
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k)
      motion.part[i][k] = r[i][k];
    motion.part[i][3] = shiftPart[i];
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
                        const gxc::VertexShaderConstants& previous, uint64_t usedMatrixRows) {
  if (camera == nullptr || previous.projection[3][2] == 0.f || current.projection[3][2] == 0.f)
    return plausible(current, previous, usedMatrixRows);
  if (!plausible_projection(current.projection, previous.projection))
    return false;
  float relative[3][4];
  if (usedMatrixRows == 0) {
    as_previous_camera(*camera, current.posnormalmatrix, relative);
    return plausible_matrix(relative, previous.posnormalmatrix);
  }
  for (uint64_t rows = usedMatrixRows; rows != 0; rows &= rows - 1) {
    const int row = __builtin_ctzll(rows);
    if (row > 61)
      continue;
    as_previous_camera(*camera, &current.transformmatrices[row], relative);
    if (!plausible_matrix(relative, &previous.transformmatrices[row]))
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

// DOL_AURORA_FRAME_INTERP_T: the blend weight toward this frame (debug; 0.5).
float blend_weight() {
  static const float weight = [] {
    const char* env = std::getenv("DOL_AURORA_FRAME_INTERP_T");
    return env != nullptr && env[0] != '\0' ? std::strtof(env, nullptr) : 0.5f;
  }();
  return weight;
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

void blend(const gxc::VertexShaderConstants& previous, const gxc::VertexShaderConstants& current, float t,
           gxc::VertexShaderConstants& out, uint64_t rows) {
  std::memcpy(&out, &current, sizeof(out));
  lerp_rows(out.posnormalmatrix, previous.posnormalmatrix, current.posnormalmatrix, 6, t);
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
      lerp_rows(&out.normalmatrices[m * 3], &previous.normalmatrices[m * 3], &current.normalmatrices[m * 3], 3, t);
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
                    const gxc::VertexShaderConstants& current, float t, gxc::VertexShaderConstants& out,
                    uint64_t rows) {
  blend(previous, current, t, out, rows);
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
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c)
        result[r][c] = static_cast<float>(camera.part[r][0] * mid[0][c] + camera.part[r][1] * mid[1][c] +
                                          camera.part[r][2] * mid[2][c]);
      result[r][3] = static_cast<float>(
          affine ? camera.part[r][0] * mid[0][3] + camera.part[r][1] * mid[1][3] + camera.part[r][2] * mid[2][3] +
                       camera.part[r][3]
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
      out.lights[l].pos[r] = static_cast<float>(camera.part[r][0] * midPos[0] + camera.part[r][1] * midPos[1] +
                                                camera.part[r][2] * midPos[2] + camera.part[r][3]);
      out.lights[l].dir[r] = static_cast<float>(camera.part[r][0] * midDir[0] + camera.part[r][1] * midDir[1] +
                                                camera.part[r][2] * midDir[2]);
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

uint64_t draw_key(const gxc::DrawPlan& plan) noexcept {
  if (plan.match_payload == nullptr || plan.match_payload_size == 0)
    return 0;
  uint64_t h = mix64((uint64_t(plan.match_primitive) << 40) ^ (uint64_t(plan.match_vtx_fmt) << 32) ^
                     plan.vertex_count ^ (uint64_t(plan.match_payload_size) << 48));
  if (plan.match_direct_position) {
    // New positions every frame: the same shape with the same texture. Its
    // matrices (usually the camera's) are blended, its vertices are not.
    h = mix64(h ^ 0xD1u ^ (uint64_t(plan.tex_address) << 8) ^ (uint64_t(plan.texmap_mask) << 40));
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

const gxc::VertexShaderConstants* blend_draw(uint64_t key, uint64_t usedMatrixRows,
                                             const gxc::VertexShaderConstants& current, bool repeatsLastDraw) {
  g_outcome = "no key";
  g_blendRepeated = false;
  const uint64_t serial = ++g_drawSerial;
  if (!enabled() || key == 0)
    return nullptr;
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
    const bool same = repeatsLastDraw && g_lastBlend.serial + 1 == serial && g_lastBlend.path == path &&
                      g_lastBlend.previous == previousBlock && g_lastBlend.motion == motion &&
                      g_lastBlend.motionGeneration == g_motionGeneration && g_lastBlend.rows == usedMatrixRows;
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
  frame.records.push_back({key, constantsIndex, false, {here[0][3], here[1][3], here[2][3]}});
  ++g_frameCounts.draws;
  KeyState& state = g_keys[key];
  const uint32_t occurrence = state.occurrence++;
  if (!g_havePrevious) {
    ++g_frameCounts.unmatched;
    g_outcome = "no previous frame";
    return nullptr;
  }

  const auto range = std::equal_range(g_previousIndex.begin(), g_previousIndex.end(), Record{key, 0, false, {}},
                                      [](const Record& a, const Record& b) { return a.key < b.key; });
  if (range.first == range.second) {
    ++g_frameCounts.unmatched;
    g_outcome = "unmatched";
    return nullptr;
  }
  g_rejectReason = "";
  const FrameRecords& previousFrame = g_frames[g_current ^ 1u];
  const Record* copies = &*range.first;
  const auto constantsOf = [&](size_t copy) -> const gxc::VertexShaderConstants& {
    return previousFrame.pool[copies[copy].constants];
  };
  const size_t candidates = static_cast<size_t>(range.second - range.first);
  const bool perspective = current.projection[3][2] != 0.f;
  const gxc::VertexShaderConstants* previous = nullptr;
  // A draw with a key of its own is the same object as its counterpart, so
  // it votes for the camera's motion whether or not the pair looks plausible
  // yet: a fast turn makes the scenery implausible until the camera's motion
  // is known.
  if (candidates == 1 && occurrence == 0 && usedMatrixRows == 0 && perspective &&
      constantsOf(0).projection[3][2] != 0.f)
    vote_motion(current.posnormalmatrix, constantsOf(0).posnormalmatrix);
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
    if (plausible_relative(view, current, candidate, usedMatrixRows))
      previous = &candidate;
  } else {
    // Without such a copy, a copy that stood still would have been carried
    // here had it been this one, unless it has just started to move (an idle
    // character swaying past the tolerance). Among copies that stood still
    // (grass, pots), one that moved is this draw only if it moved at an
    // actor's pace (a pot Link carries) and no copy that stood still is
    // nearer; otherwise this draw is a copy that has just come into view.
    const auto usable = [&](size_t copy) {
      if (camera == nullptr)
        return true;
      const double moved = motion_error(*camera, here, copies[copy].position);
      if (copies[copy].still)
        return moved <= 4.0 * tolerance;
      // A copy that moved at an actor's pace (a pot Link carries), when no
      // copy that stood still is nearer.
      return moved < nearestStill && moved <= 40.0 + 0.01 * translation_length(here);
    };
    bool anyUsable = false;
    // The same occurrence of the key: a model drawn from a fixed list (a
    // model's materials, moving copies) keeps its order.
    if (occurrence < candidates && usable(occurrence)) {
      anyUsable = true;
      const auto& candidate = constantsOf(occurrence);
      if (plausible_relative(view, current, candidate, usedMatrixRows))
        previous = &candidate;
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
        if ((previous == nullptr || d < bestDistance) &&
            plausible_relative(view, current, constantsOf(i), usedMatrixRows)) {
          previous = &constantsOf(i);
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
          !plausible_relative(view, current, g_stoodBefore, 0))
        return nullptr;
      if (repeat(BlendPath::StoodBefore, camera, view)) {
        g_blendRepeated = true;
        return &g_blended;
      }
      if (view != nullptr)
        blend_relative(*view, g_stoodBefore, current, blend_weight(), g_blended, 0);
      else
        blend(g_stoodBefore, current, blend_weight(), g_blended, 0);
      return &g_blended;
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
      return &g_blended;
    }
    Motion own{};
    if (own_motion(constantsOf(0).posnormalmatrix, current.posnormalmatrix, own) && derive(own)) {
      blend_relative(own, constantsOf(0), current, blend_weight(), g_blended, usedMatrixRows);
      ++g_frameCounts.blended;
      g_outcome = "blended by its own motion";
      return &g_blended;
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
  if (repeat(BlendPath::Main, previous, relative ? view : nullptr)) {
    if (g_lastBlend.identical) {
      ++g_frameCounts.identical;
      g_outcome = "identical";
      return nullptr;
    }
    ++g_frameCounts.blended;
    g_outcome = "blended";
    g_blendRepeated = true;
    return &g_blended;
  }
  g_lastBlend.identical = std::memcmp(previous, &current, sizeof(current)) == 0;
  if (g_lastBlend.identical) {
    ++g_frameCounts.identical;
    g_outcome = "identical";
    return nullptr;
  }
  if (relative)
    blend_relative(*view, *previous, current, blend_weight(), g_blended, usedMatrixRows);
  else
    blend(*previous, current, blend_weight(), g_blended, usedMatrixRows);
  ++g_frameCounts.blended;
  g_outcome = "blended";
  return &g_blended;
}

bool last_blend_repeated() noexcept { return g_blendRepeated; }

bool frame_verdict() noexcept {
  g_lastVerdict = false;
  if (!enabled() || !g_havePrevious || g_frameCounts.blended == 0)
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
                 "camera=%s turn=%.1f votes=%u fresh=%llu\n",
                 static_cast<unsigned long long>(g_gameFrame.load(std::memory_order_relaxed)), g_lastVerdictSeen,
                 static_cast<unsigned long long>(g_frameCounts.draws),
                 static_cast<unsigned long long>(g_frameCounts.blended),
                 static_cast<unsigned long long>(g_frameCounts.rejected),
                 static_cast<unsigned long long>(g_frameCounts.unmatched),
                 camera == nullptr ? "none" : (camera == &g_predicted ? "predicted" : "voted"), angle,
                 camera == nullptr ? 0u : camera->votes, static_cast<unsigned long long>(g_frameCounts.fresh));
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
  if (g_havePredicted)
    g_predicted = g_motions[g_leadingMotion];
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
bool tracing() noexcept { return g_traceFrame != 0 && g_gameFrame.load(std::memory_order_relaxed) == g_traceFrame; }

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
