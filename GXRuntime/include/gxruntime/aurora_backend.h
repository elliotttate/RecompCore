// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef GXRUNTIME_AURORA_BACKEND_H
#define GXRUNTIME_AURORA_BACKEND_H

#include "gxruntime/platform.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AuroraBackendConfig {
    const char* app_name;
    unsigned window_width;
    unsigned window_height;
    bool vsync;
    bool allow_texture_dumps;
    bool info_logging;
    bool graphics_logging;
    bool force_untextured;
} AuroraBackendConfig;

bool dol_aurora_initialize(int argc, char** argv,
                           const AuroraBackendConfig* config);
void dol_aurora_shutdown(void);

// Host overlay hooks. The overlay callback runs on the main thread inside the
// open Aurora frame, just before it is submitted, so it may issue ImGui draw
// calls that composite over the game. The event observer sees every SDL event
// Aurora forwards (touch, keyboard, controller) as a const SDL_Event*.
typedef void (*DolAuroraOverlayFn)(void* user);
typedef void (*DolAuroraEventObserverFn)(const void* sdl_event, void* user);
void dol_aurora_set_overlay(DolAuroraOverlayFn draw, void* user);
void dol_aurora_set_event_observer(DolAuroraEventObserverFn observe, void* user);
// While no frame can be opened, the backend asks this predicate whether the
// host wants the guest held (for example an iOS app that is not active). If so
// it pumps events and sleeps on the main thread until the predicate clears.
typedef bool (*DolAuroraHoldFn)(void* user);
void dol_aurora_set_hold(DolAuroraHoldFn should_hold, void* user);
// While the guest is held at a present, keep drawing frames: the last picture
// with the host overlay over it, so an overlay menu drawn with ImGui stays live
// (a menu that pauses the game). Off by default: the hold then only waits.
void dol_aurora_set_hold_redraw(bool redraw);
// Fast-forward, for stretches with nothing to see or hear (a scene change's
// black): frames are rendered but not presented, and audio pushes are
// dropped, so neither the display nor the audio queue holds the guest to real
// time. The screen keeps the last presented frame.
void dol_aurora_set_fast_forward(bool on);

/* Cumulative main-thread frame timing, for per-second diagnostics: time spent
   waiting for the FIFO translation worker at the guest's GX barriers, time in
   the present (including that wait's own present), the part of it inside
   aurora_end_frame (GPU submission and waiting for a drawable), and the draw
   calls submitted. Differences between two reads cover the interval. */
typedef struct DolAuroraFrameTiming {
    unsigned long long presents;
    unsigned long long drain_us;
    unsigned long long present_us;
    unsigned long long end_frame_us;
    unsigned long long draws;
    unsigned long long display_copies;  /* the game's GXCopyDisp calls */
    unsigned long long audio_throttles; /* 1 ms waits for the audio queue to drain */
    unsigned long long audio_dropped;   /* pushes dropped on a full queue (no throttle) */
    int audio_queued_ms;                /* audio waiting in the device queue now */
    unsigned long long shown;           /* frames presented, in-between frames included */
    unsigned long long interp_frames;   /* game frames the in-between frames saw */
    unsigned long long interp_interpolated; /* ... and interpolated */
    unsigned long long interp_draws;
    unsigned long long interp_rejected; /* draws judged implausible (not blended) */
    unsigned long long interp_unmatched; /* draws with no counterpart the frame before */
} DolAuroraFrameTiming;
void dol_aurora_frame_timing(DolAuroraFrameTiming* out);

/* Presents a frame the FIFO worker has finished and requested, if any. The
   request is otherwise taken only at the next GX write; the host calls this
   at each retrace so a frame finished while the guest idles is not held. */
void aurora_backend_service_present(void);

#ifdef __cplusplus
}
#endif

#endif
