// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef GXRUNTIME_AURORA_BACKEND_H
#define GXRUNTIME_AURORA_BACKEND_H

#include "gxruntime/platform.h"

#include <stdbool.h>
#include <stddef.h>

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
    /* CPU time of the graphics threads, microseconds: the FIFO translation
       worker, Smooth Motion's helper and Aurora's render worker. */
    unsigned long long gx_worker_cpu_us;
    unsigned long long interp_helper_cpu_us;
    unsigned long long render_worker_cpu_us;
    /* The GX worker's second stage (the draw plans into Aurora), when it runs
       on a thread of its own; 0 otherwise. */
    unsigned long long gx_submit_cpu_us;
} DolAuroraFrameTiming;
void dol_aurora_frame_timing(DolAuroraFrameTiming* out);

/* Presents a frame the FIFO worker has finished and requested, if any. The
   request is otherwise taken only at the next GX write; the host calls this
   at each retrace so a frame finished while the guest idles is not held. */
void aurora_backend_service_present(void);

/* Save states (debugging). dol_aurora_gx_save_state drains the FIFO worker and
   returns, in a malloc'd blob (free it), the retail GX front end's register
   model with the bytes of any command the guest has not finished writing, and
   the gxcore sink's register state; 0 when there is nothing to save (the
   recomp path is off, or the front end has failed). dol_aurora_gx_load_state
   puts one back: resolved texture ranges are resolved again against the
   current guest memory and the texture cache is emptied. Renderer caches and
   EFB copies are not saved; they are rebuilt as the game draws. */
size_t dol_aurora_gx_save_state(void** out);
bool dol_aurora_gx_load_state(const void* data, size_t size);
/* Waits for the FIFO worker to finish what the guest has written: before a
   load replaces guest memory the worker must not still be reading it. */
void dol_aurora_gx_drain(void);

#ifdef __cplusplus
}
#endif

#endif
