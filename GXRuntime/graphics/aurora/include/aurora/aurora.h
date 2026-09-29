#ifndef AURORA_AURORA_H
#define AURORA_AURORA_H

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>

extern "C" {
#else
#include "stdbool.h"
#include "stddef.h"
#include "stdint.h"
#endif

typedef enum {
  SAMPLER_BILINEAR,
  SAMPLER_AREA,
} AuroraSampler;

typedef enum {
  BACKEND_AUTO,
  BACKEND_D3D11,
  BACKEND_D3D12,
  BACKEND_METAL,
  BACKEND_VULKAN,
  BACKEND_OPENGL,
  BACKEND_OPENGLES,
  BACKEND_WEBGPU,
  BACKEND_NULL,
} AuroraBackend;

typedef enum {
  LOG_DEBUG,
  LOG_INFO,
  LOG_WARNING,
  LOG_ERROR,
  LOG_FATAL,
} AuroraLogLevel;

typedef struct {
  int32_t x;
  int32_t y;
} AuroraWindowPos;

typedef struct {
  uint32_t width;
  uint32_t height;

  /**
   * Width of the main GX framebuffer.
   */
  uint32_t fb_width;

  /**
   * Height of the main GX framebuffer.
   */
  uint32_t fb_height;

  /**
   * The size of the framebuffer used to present to the operating system.
   * May differ from fb_width if Aurora is instructed to force an aspect ratio or resolution configuration.
   */
  uint32_t native_fb_width;

  /**
   * The size of the framebuffer used to present to the operating system.
   * May differ from fb_height if Aurora is instructed to force an aspect ratio or resolution configuration.
   */
  uint32_t native_fb_height;
  float scale;
} AuroraWindowSize;

typedef struct SDL_Window SDL_Window;
typedef struct AuroraEvent AuroraEvent;

typedef void (*AuroraLogCallback)(AuroraLogLevel level, const char* module, const char* message, unsigned int len);
typedef void (*AuroraImGuiInitCallback)(const AuroraWindowSize* size);

#define MEM1_DEFAULT_SIZE = 24 * 1024 * 1024;
#define ARAM_DEFAULT_SIZE = 16 * 1024 * 1024;

typedef struct {
  const char* appName;
  const char* userPath;
  const char* cachePath;
  const char* resourcesPath;
  AuroraBackend desiredBackend;
  uint32_t msaa;
  uint16_t maxTextureAnisotropy;
  bool vsync;
  bool startFullscreen;
  bool allowJoystickBackgroundEvents;
  bool pauseOnFocusLost;
  bool allowTextureDumps;
  bool allowCpuAdapter;
  int32_t windowPosX;
  int32_t windowPosY;
  uint32_t windowWidth;
  uint32_t windowHeight;
  void* iconRGBA8;
  uint32_t iconWidth;
  uint32_t iconHeight;
  AuroraLogCallback logCallback;
  AuroraLogLevel logLevel;
  AuroraImGuiInitCallback imGuiInitCallback;

  /*
   * The size of the GameCube's main memory, or MEM1 on the Wii.
   * Note that it will not be allocated at the exact 0x80000000 address, as that cannot be guaranteed.
   * This can be set to 0 to disable allocating this region.
   */
  uint32_t mem1Size;

  /*
   * The size of the GameCube's ARAM, or MEM2 on the Wii.
   * This can be set to 0 to disable allocating this region.
   */
  uint32_t mem2Size;
} AuroraConfig;

typedef struct {
  AuroraBackend backend;
  const char* userPath;
  const char* cachePath;
  SDL_Window* window;
  AuroraWindowSize windowSize;
} AuroraInfo;

AuroraInfo aurora_initialize(int argc, char* argv[], const AuroraConfig* config);
void aurora_shutdown();
const AuroraEvent* aurora_update();
bool aurora_begin_frame();
void aurora_end_frame();

void aurora_set_log_level(AuroraLogLevel level);
void aurora_set_pause_on_focus_lost(bool value);
void aurora_set_background_input(bool value);
void aurora_set_resampler(AuroraSampler sampler);
/* Internal render resolution as a multiple of the game's 640x480 frame
   buffer (3 renders 1920x1440 and scales the result to the window); 0 renders
   at the window's native pixel size. Takes effect at the next frame. */
void aurora_set_frame_buffer_scale(float scale);
/* While set, frames are rendered but not presented: no drawable is taken, so
   nothing waits for the display. The screen keeps the last presented frame. */
void aurora_set_present_suppressed(bool suppressed);
/* Anisotropic filtering for every mipmapped, linearly filtered texture, as
   Dolphin's "Anisotropic Filtering" enhancement: 1 (or 0) leaves the game's
   own sampler settings, 2-16 forces that many samples. Takes effect at the
   next draw. DOL_AURORA_FORCE_ANISO sets the starting value. */
void aurora_set_forced_anisotropy(unsigned samples);
/* In-between frames: each finished frame is drawn a second time with every
   draw's transforms blended halfway toward the previous frame and shown
   first, so a 30 FPS game presents 60 frames a second with its own logic
   unchanged (one extra half frame of display latency). Takes effect at the
   next frame. DOL_AURORA_FRAME_INTERP=1 sets the starting value. */
void aurora_set_frame_interpolation(bool enabled);
/* A frames-a-second counter at the top of the window (with the game's own
   rate beside it when in-between frames are on). DOL_AURORA_SHOW_FPS=1 sets
   the starting value. */
void aurora_set_fps_overlay(bool enabled);
bool aurora_get_frame_interpolation(void);
/* Frames presented to the window so far, in-between frames included (what
   the frames-a-second counter counts). */
unsigned long long aurora_get_shown_frames(void);
/* The in-between frames' running totals: game frames seen and interpolated,
   and their draws, the ones rejected as implausible and those with no
   counterpart in the frame before. Differences between two reads cover the
   interval. */
typedef struct AuroraFrameInterpTotals {
  unsigned long long frames, interpolated, draws, rejected, unmatched;
} AuroraFrameInterpTotals;
void aurora_get_frame_interp_totals(AuroraFrameInterpTotals* out);

AuroraBackend aurora_get_backend();
const AuroraBackend* aurora_get_available_backends(size_t* count);

#ifdef __cplusplus
}
#endif

#endif
