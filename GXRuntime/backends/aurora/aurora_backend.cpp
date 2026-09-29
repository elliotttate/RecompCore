// SPDX-License-Identifier: GPL-3.0-or-later
#include "aurora_backend_private.h"
#include <aurora/aurora.h>
#include <aurora/event.h>
#include <aurora/gfx.h>
#include <aurora/texture.hpp>
#include <gxruntime/guest_memory_dirty.h>
#if GXRUNTIME_HAS_AURORA_RECOMP
#include <gfx/gxcore_draw.hpp>
#endif
#include <SDL3/SDL_init.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <window.hpp>
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

namespace gx_aurora {

bool g_initialized = false;
bool g_frame_open = false;
bool g_should_quit = false;
DolAuroraOverlayFn g_host_overlay = nullptr;
void* g_host_overlay_user = nullptr;
DolAuroraEventObserverFn g_host_event_observer = nullptr;
void* g_host_event_user = nullptr;
DolAuroraHoldFn g_host_hold = nullptr;
bool g_hold_redraw = false;
void* g_host_hold_user = nullptr;
bool g_graphics_log = false;
bool g_force_untextured = false;
bool g_gx_core_enabled = true;
bool g_draw_opcode_pending = false;
bool g_audio_queue_log = false;
bool g_frame_pacing_log = false;
unsigned long long g_present_count = 0;
unsigned long long g_fifo_bytes = 0;
unsigned long long g_audio_push_count = 0;
unsigned long long g_audio_throttle_count = 0;
unsigned long long g_audio_dropped_count = 0;
std::atomic_bool g_audio_discard{false};
unsigned long long g_audio_low_log_push = 0;
u32 g_audio_sample_rate = 32000;
SDL_AudioStream* g_audio_stream = nullptr;
bool g_audio_playing = false;
int g_audio_prebuffer_ms = 40;
unsigned long long g_audio_starved_count = 0;
unsigned long long g_audio_stretched_count = 0;
int g_audio_max_queue_ms = 250;
DolPlatformGuestAddressResolverFn g_guest_address_resolver = nullptr;
void* g_guest_address_resolver_user = nullptr;

std::array<PendingTextureMetadata, 8> g_pending_textures{};
std::array<PendingTlutMetadata, 20> g_pending_tluts{};

#if GXRUNTIME_HAS_AURORA_RECOMP
gxruntime::aurora_recomp::RetailGxFrontend g_shadow_frontend;
gxruntime::aurora_recomp::ConsumingAuroraRenderSink g_shadow_packet_sink;
bool g_shadow_frontend_enabled = false;
bool g_shadow_frontend_failed = false;
gxruntime::gxcore::GxCoreSink g_core_sink;
unsigned long long g_core_submitted = 0;
unsigned long long g_core_rejected = 0;
std::atomic<bool> g_display_copy_pending{false};

unsigned long long g_shadow_last_draw_total = 0;
unsigned long long g_shadow_last_vertex_total = 0;
unsigned long long g_shadow_draw_mismatch_frames = 0;
unsigned long long g_shadow_last_rawvert_total = 0;
unsigned long long g_shadow_last_topoidx_total = 0;
unsigned long long g_shadow_last_storage_total = 0;
unsigned long long g_shadow_vert_extent_mismatch_frames = 0;
unsigned long long g_shadow_last_zero_draw_total = 0;

bool g_shadow_prev_frame_valid = false;
unsigned long long g_shadow_prev_frame_index = 0;
unsigned long long g_shadow_prev_draws = 0;
unsigned long long g_shadow_prev_zero_draws = 0;
unsigned long long g_shadow_prev_verts = 0;
unsigned long long g_shadow_prev_rawvert = 0;
unsigned long long g_shadow_prev_topoidx = 0;
unsigned long long g_shadow_prev_storage = 0;

bool g_shadow_transform_log_enabled = false;
bool g_shadow_light_log_enabled = false;
bool g_shadow_light_log_lit_only = false;
bool g_shadow_transform_log_sequence_enabled = false;
unsigned long long g_shadow_transform_log_frame = 0;
unsigned long long g_shadow_transform_log_min_frame = 0;
long g_shadow_transform_log_draw = -1;
unsigned long long g_shadow_transform_log_sequence = 0;
unsigned long g_shadow_transform_log_limit = 1;
unsigned long g_shadow_transform_log_count = 0;
std::size_t g_shadow_transform_next_draw_index = 0;

bool g_trace_armed = false;
std::string g_trace_path;
unsigned long long g_trace_first_frame = 1ull;
unsigned long long g_trace_last_frame = ~0ull;
unsigned long long g_trace_present_scope_frame = 0ull;
bool g_trace_frame_begun = false;
unsigned long long g_trace_frames_recorded = 0ull;
gxruntime::aurora_recomp::trace::TraceWriter g_trace_writer;
std::unordered_map<u64, u64> g_trace_mem_dedup;
#endif

int audio_ms_env(const char* name, int fallback, int min_value, int max_value) {
    const char* text = std::getenv(name);
    if (text == nullptr || text[0] == '\0')
        return fallback;

    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (end == text)
        return fallback;
    if (value < min_value)
        value = min_value;
    if (value > max_value)
        value = max_value;
    return static_cast<int>(value);
}

unsigned long long ull_env(const char* name, unsigned long long fallback) {
    const char* text = std::getenv(name);
    if (text == nullptr || text[0] == '\0')
        return fallback;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    return end == text ? fallback : value;
}

long long_env(const char* name, long fallback) {
    const char* text = std::getenv(name);
    if (text == nullptr || text[0] == '\0')
        return fallback;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    return end == text ? fallback : value;
}

void log_callback(AuroraLogLevel level, const char* module, const char* message,
                  unsigned int) {
    const char* level_name = "info";
    switch (level) {
    case LOG_DEBUG:   level_name = "debug"; break;
    case LOG_INFO:    level_name = "info"; break;
    case LOG_WARNING: level_name = "warning"; break;
    case LOG_ERROR:   level_name = "error"; break;
    case LOG_FATAL:   level_name = "fatal"; break;
    }
    std::fprintf(level >= LOG_ERROR ? stderr : stdout, "[aurora:%s:%s] %s\n",
                 level_name, module ? module : "core", message ? message : "");
    if (level == LOG_FATAL)
        std::abort();
}

void poll_events() {
    const AuroraEvent* event = aurora_update();
    while (event != nullptr && event->type != AURORA_NONE) {
        if (event->type == AURORA_EXIT)
            g_should_quit = true;
        if (event->type == AURORA_SDL_EVENT && g_host_event_observer != nullptr)
            g_host_event_observer(&event->sdl, g_host_event_user);
        ++event;
    }
}

void run_host_overlay() {
    if (g_host_overlay != nullptr)
        g_host_overlay(g_host_overlay_user);
}

bool host_wants_hold() {
    return g_host_hold != nullptr && g_host_hold(g_host_hold_user);
}

void install_platform_ops() {


    const DolPlatformOps ops = {
        .should_quit = aurora_backend_should_quit,
        .present = aurora_backend_present,
        .mark_gx_begin = aurora_backend_mark_gx_begin,
        .gx_write = aurora_backend_gx_write,
        .gx_flush = aurora_backend_gx_flush,
        .call_display_list = aurora_backend_call_display_list,
        .set_array = aurora_backend_set_array,
        .set_array_guest = aurora_backend_set_array_guest,
        .load_texture = aurora_backend_load_texture,
        .load_texture_guest = aurora_backend_load_texture_guest,
        .load_tlut = aurora_backend_load_tlut,
        .load_tlut_guest = aurora_backend_load_tlut_guest,
        .set_copy_destination = aurora_backend_set_copy_destination,
        .set_copy_destination_guest = aurora_backend_set_copy_destination_guest,
        .set_guest_address_resolver = aurora_backend_set_guest_address_resolver,
        .configure_vi = aurora_backend_configure_vi,

        .pad_init = aurora_backend_pad_init,
        .pad_read = aurora_backend_pad_read,
        .pad_reset = aurora_backend_pad_reset,
        .pad_recalibrate = aurora_backend_pad_recalibrate,
        .pad_control_motor = aurora_backend_pad_control_motor,
        .pad_set_spec = aurora_backend_pad_set_spec,

        .audio_set_sample_rate = aurora_backend_audio_set_sample_rate,
        .audio_push = aurora_backend_audio_push,
    };
    dol_platform_install(&ops);
}

} // namespace gx_aurora

extern "C" {

bool aurora_backend_should_quit(void) {
    return gx_aurora::g_should_quit;
}

bool dol_aurora_initialize(int argc, char** argv,
                           const AuroraBackendConfig* backend_config) {
    if (gx_aurora::g_initialized)
        return true;

    const AuroraBackendConfig defaults = {
        .app_name = "GXRuntime",
        .window_width = 1280,
        .window_height = 960,
        .vsync = true,
        .allow_texture_dumps = false,
        .info_logging = false,
        .graphics_logging = false,
        .force_untextured = false,
    };
    if (backend_config == nullptr)
        backend_config = &defaults;

    AuroraConfig config{};
    config.appName =
        backend_config->app_name != nullptr ? backend_config->app_name
                                            : defaults.app_name;
    config.desiredBackend = BACKEND_AUTO;
    config.vsync = backend_config->vsync;
    config.windowWidth = backend_config->window_width != 0
                             ? backend_config->window_width
                             : defaults.window_width;
    config.windowHeight = backend_config->window_height != 0
                              ? backend_config->window_height
                              : defaults.window_height;
    // The display the picture is made for (desktop hosts; iOS fills its
    // screen). DOL_AURORA_WINDOW=WxH sizes the window in points; otherwise a
    // widened picture (DOL_AURORA_ASPECT_RATIO) gets a window of its shape at
    // the configured height, so 16:10 opens 1152x720 rather than 960x720.
    // DOL_AURORA_FULLSCREEN=1 starts fullscreen (a MacBook's fullscreen area,
    // below the camera housing, is 16:10).
    {
        const char* window_env = std::getenv("DOL_AURORA_WINDOW");
        unsigned width = 0, height = 0;
        if (window_env != nullptr && std::sscanf(window_env, "%ux%u", &width, &height) == 2 &&
            width >= 320 && height >= 240) {
            config.windowWidth = width;
            config.windowHeight = height;
        } else if (const char* ratio_env = std::getenv("DOL_AURORA_ASPECT_RATIO")) {
            const float ratio = std::strtof(ratio_env, nullptr);
            if (ratio > 1.0f && ratio < 4.0f)
                config.windowWidth = static_cast<uint32_t>(config.windowHeight * ratio + 0.5f);
        }
        const char* fullscreen_env = std::getenv("DOL_AURORA_FULLSCREEN");
        config.startFullscreen = fullscreen_env != nullptr && fullscreen_env[0] != '\0' && fullscreen_env[0] != '0';
    }
    config.allowTextureDumps = backend_config->allow_texture_dumps;
    config.logCallback = gx_aurora::log_callback;
    config.logLevel = backend_config->info_logging ? LOG_INFO : LOG_ERROR;
    config.mem1Size = 0;
    config.mem2Size = 0;

    const AuroraInfo info = aurora_initialize(argc, argv, &config);
    gx_aurora::g_initialized = info.window != nullptr;
    if (!gx_aurora::g_initialized)
        return false;

    // Keep the guest's configured frame aspect instead of stretching it to the
    // window. iPad and iPhone windows are rarely 4:3; the macOS window is
    // created 4:3 and stays unchanged unless DOL_AURORA_ASPECT_FIT is set.
    {
        bool aspect_fit = false;
#if defined(TARGET_OS_IOS) && TARGET_OS_IOS
        aspect_fit = true;
#endif
        const char* fit_env = std::getenv("DOL_AURORA_ASPECT_FIT");
        if (fit_env != nullptr && fit_env[0] != '\0')
            aspect_fit = fit_env[0] != '0';
        // DOL_AURORA_ASPECT_RATIO: a display aspect to letterbox to, such as
        // 1.7778 for a game patched to render anamorphic 16:9 (the
        // widescreen mod). It implies fitting.
        const char* ratio_env = std::getenv("DOL_AURORA_ASPECT_RATIO");
        const float ratio = ratio_env != nullptr ? std::strtof(ratio_env, nullptr) : 0.f;
        if (ratio > 0.f) {
            aspect_fit = true;
            aurora::window::set_frame_buffer_aspect_override(ratio);
        }
        aurora::window::set_frame_buffer_aspect_fit(aspect_fit);
    }
    // DOL_AURORA_RENDER_SCALE: the render resolution as a multiple of the
    // game's 480 lines (3 renders 1440 lines, the iOS default); 0 renders at
    // the window's own pixel height. A widened picture keeps its aspect.
    if (const char* scale_env = std::getenv("DOL_AURORA_RENDER_SCALE"); scale_env != nullptr && scale_env[0] != '\0') {
        const float scale = std::strtof(scale_env, nullptr);
        if (scale >= 0.f && scale <= 8.f)
            aurora_set_frame_buffer_scale(scale);
    }
    // DOL_AURORA_TEXTURE_PACK: a folder of Dolphin-format replacement
    // textures (tex1_WxH_hash[_tlut]_fmt.png or .dds, searched recursively,
    // with _mipN sidecars), such as an HD texture pack's GZL folder.
    if (const char* pack = std::getenv("DOL_AURORA_TEXTURE_PACK"); pack != nullptr && pack[0] != '\0') {
        static aurora::texture::ReplacementGroup s_texture_pack;
        s_texture_pack = aurora::texture::load_replacement_directory(pack);
        std::fprintf(stderr, "[mods] texture-pack=%s replacements=%zu\n", pack,
                     s_texture_pack.registrations.size());
    }

    gx_aurora::g_graphics_log = backend_config->graphics_logging;
    gx_aurora::g_force_untextured = backend_config->force_untextured;
    gx_aurora::g_frame_pacing_log = std::getenv("DOL_FRAME_PACING_LOG") != nullptr;
#if GXRUNTIME_HAS_AURORA_RECOMP
    gx_aurora::g_display_copy_pending = false;
    gx_aurora::g_shadow_light_log_enabled =
        std::getenv("DOL_AURORA_RECOMP_DRAW_LIGHT_LOG") != nullptr;
    gx_aurora::g_shadow_light_log_lit_only =
        std::getenv("DOL_AURORA_RECOMP_DRAW_LIGHT_LIT_ONLY") != nullptr;
    gx_aurora::g_shadow_transform_log_enabled =
        std::getenv("DOL_AURORA_RECOMP_DRAW_TRANSFORM_LOG") != nullptr ||
        gx_aurora::g_shadow_light_log_enabled;
    gx_aurora::g_shadow_transform_log_frame =
        gx_aurora::ull_env("DOL_AURORA_RECOMP_DRAW_TRANSFORM_FRAME", 0ull);
    gx_aurora::g_shadow_transform_log_min_frame =
        gx_aurora::ull_env("DOL_AURORA_RECOMP_DRAW_TRANSFORM_MIN_FRAME", 0ull);
    gx_aurora::g_shadow_transform_log_draw =
        gx_aurora::long_env("DOL_AURORA_RECOMP_DRAW_TRANSFORM_DRAW", -1);
    const char* transform_sequence =
        std::getenv("DOL_AURORA_RECOMP_DRAW_TRANSFORM_SEQUENCE");
    gx_aurora::g_shadow_transform_log_sequence_enabled =
        transform_sequence != nullptr && transform_sequence[0] != '\0';
    gx_aurora::g_shadow_transform_log_sequence =
        gx_aurora::ull_env("DOL_AURORA_RECOMP_DRAW_TRANSFORM_SEQUENCE", 0ull);
    gx_aurora::g_shadow_transform_log_limit = static_cast<unsigned long>(
        gx_aurora::ull_env("DOL_AURORA_RECOMP_DRAW_TRANSFORM_LIMIT", 1ull));
    if (gx_aurora::g_shadow_transform_log_limit == 0ul)
        gx_aurora::g_shadow_transform_log_limit = 1ul;
    gx_aurora::g_shadow_transform_log_count = 0;
    gx_aurora::g_shadow_transform_next_draw_index = 0;
    gx_aurora::g_trace_armed = false;
    gx_aurora::g_trace_path.clear();
    gx_aurora::g_trace_first_frame = 1ull;
    gx_aurora::g_trace_last_frame = ~0ull;
    gx_aurora::g_trace_present_scope_frame = 0ull;
    gx_aurora::g_trace_frame_begun = false;
    gx_aurora::g_trace_frames_recorded = 0ull;
    gx_aurora::g_trace_mem_dedup.clear();
    const char* trace_out = std::getenv("DOL_AURORA_RECOMP_TRACE_OUT");
    if (trace_out != nullptr && trace_out[0] != '\0') {
        gx_aurora::g_trace_path = trace_out;
        gx_aurora::g_trace_armed = true;
        const char* trace_window =
            std::getenv("DOL_AURORA_RECOMP_TRACE_FRAMES");
        if (trace_window != nullptr && trace_window[0] != '\0') {
            char* end = nullptr;
            const unsigned long long first =
                std::strtoull(trace_window, &end, 10);
            bool window_ok = end != trace_window && *end == ':';
            if (window_ok) {
                const char* second = end + 1;
                char* end2 = nullptr;
                const unsigned long long last =
                    std::strtoull(second, &end2, 10);
                window_ok = end2 != second && *end2 == '\0' && last >= first;
                if (window_ok) {
                    gx_aurora::g_trace_first_frame = first == 0ull ? 1ull : first;
                    gx_aurora::g_trace_last_frame = last;
                }
            }
            if (!window_ok)
                std::fprintf(stderr,
                             "[trace] ignoring malformed "
                             "DOL_AURORA_RECOMP_TRACE_FRAMES=%s (want A:B); "
                             "recording all frames\n",
                             trace_window);
        }
        std::fprintf(stderr, "[trace] recording armed path=%s frames=%llu:%llu\n",
                     gx_aurora::g_trace_path.c_str(), gx_aurora::g_trace_first_frame,
                     gx_aurora::g_trace_last_frame);
    }
    {
        const char* core_env = std::getenv("DOL_GX_CORE");
        gx_aurora::g_gx_core_enabled = !(core_env != nullptr && core_env[0] == '0');
    }
    gx_aurora::g_shadow_frontend_enabled =
        std::getenv("DOL_AURORA_RECOMP_FRONTEND_SHADOW") != nullptr ||
        gx_aurora::g_shadow_transform_log_enabled || gx_aurora::g_trace_armed || gx_aurora::g_gx_core_enabled;
    if (gx_aurora::g_gx_core_enabled) {
        dol_guest_memory_dirty_reset();
        aurora::gfx::gxcore::set_texture_dirty_epoch_observer(
            gx_aurora::core_texture_dirty_epoch);
        gx_aurora::g_core_sink.set_plan_observer(gx_aurora::core_plan_observer, nullptr);
        gx_aurora::g_core_sink.set_copy_observer(gx_aurora::core_copy_observer, nullptr);
        gx_aurora::g_core_submitted = 0;
        gx_aurora::g_core_rejected = 0;
        aurora::gfx::gxcore::reset_texture_cache();
        std::fprintf(stderr,
                     "[gx-core] renderer = gxcore (default): draws route "
                     "through the Dolphin-ported gxcore, live Aurora gx layer "
                     "bypassed\n");
    }
    // The FIFO translation runs on a worker now, so the reset has to wait for
    // it to be idle before the front end's state is cleared.
    gx_aurora::shadow_frontend_flush_pending();
    gx_aurora::g_shadow_frontend_failed = false;
    gx_aurora::g_shadow_frontend.reset(nullptr);
    gx_aurora::g_shadow_frontend.set_packet_drain_enabled(gx_aurora::g_shadow_frontend_enabled);
    // The frame ends at the display copy: stop each parse there so the draws
    // after it are recorded after the present, into the next frame.
    gx_aurora::g_shadow_frontend.set_stop_at_display_copy(gx_aurora::g_gx_core_enabled);
    gx_aurora::g_shadow_packet_sink.reset();
    gx_aurora::g_shadow_packet_sink.set_streaming(true);
    gx_aurora::g_shadow_packet_sink.set_draw_observer(
        gx_aurora::g_shadow_transform_log_enabled ? gx_aurora::shadow_transform_observer : nullptr,
        nullptr);
    gx_aurora::g_shadow_last_draw_total = 0;
    gx_aurora::g_shadow_last_vertex_total = 0;
    gx_aurora::g_shadow_draw_mismatch_frames = 0;
    gx_aurora::g_shadow_last_rawvert_total = 0;
    gx_aurora::g_shadow_last_topoidx_total = 0;
    gx_aurora::g_shadow_last_storage_total = 0;
    gx_aurora::g_shadow_vert_extent_mismatch_frames = 0;
    gx_aurora::g_shadow_last_zero_draw_total = 0;
    gx_aurora::g_shadow_prev_frame_valid = false;
    if (gx_aurora::g_shadow_transform_log_enabled) {
        if (gx_aurora::g_shadow_transform_log_sequence_enabled) {
            std::fprintf(stderr,
                         "[gfx] draw-transform log enabled frame=%llu "
                         "draw=%ld min_frame=%llu sequence=%llu limit=%lu\n",
                         gx_aurora::g_shadow_transform_log_frame,
                         gx_aurora::g_shadow_transform_log_draw,
                         gx_aurora::g_shadow_transform_log_min_frame,
                         gx_aurora::g_shadow_transform_log_sequence,
                         gx_aurora::g_shadow_transform_log_limit);
        } else {
            std::fprintf(stderr,
                         "[gfx] draw-transform log enabled frame=%llu "
                         "draw=%ld min_frame=%llu sequence=any limit=%lu\n",
                         gx_aurora::g_shadow_transform_log_frame,
                         gx_aurora::g_shadow_transform_log_draw,
                         gx_aurora::g_shadow_transform_log_min_frame,
                         gx_aurora::g_shadow_transform_log_limit);
        }
    }
#endif
    gx_aurora::g_audio_queue_log = std::getenv("DOL_AUDIO_QUEUE_LOG") != nullptr;
    gx_aurora::g_audio_prebuffer_ms = gx_aurora::audio_ms_env("DOL_AUDIO_PREBUFFER_MS", 40, 20, 500);
    gx_aurora::g_audio_max_queue_ms =
        gx_aurora::audio_ms_env("DOL_AUDIO_MAX_QUEUE_MS", 250, gx_aurora::g_audio_prebuffer_ms, 1000);
    gx_aurora::g_audio_push_count = 0;
    gx_aurora::g_audio_throttle_count = 0;
    gx_aurora::g_audio_starved_count = 0;
    gx_aurora::g_audio_stretched_count = 0;
    gx_aurora::g_audio_low_log_push = 0;
    gx_aurora::g_audio_sample_rate = 32000;

    if (SDL_InitSubSystem(SDL_INIT_AUDIO)) {
        SDL_AudioSpec spec{};
        spec.format = SDL_AUDIO_S16;
        spec.channels = 2;
        spec.freq = static_cast<int>(gx_aurora::g_audio_sample_rate);
        gx_aurora::g_audio_stream = SDL_OpenAudioDeviceStream(
            SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr, nullptr);
        if (gx_aurora::g_audio_stream == nullptr)
            std::fprintf(stderr, "[audio] SDL output unavailable: %s\n", SDL_GetError());
        else if (gx_aurora::g_audio_queue_log)
            std::fprintf(stderr, "[audio-queue] prebuffer_ms=%d max_queue_ms=%d\n",
                         gx_aurora::g_audio_prebuffer_ms, gx_aurora::g_audio_max_queue_ms);
    } else {
        std::fprintf(stderr, "[audio] SDL audio initialization failed: %s\n", SDL_GetError());
    }

    gx_aurora::poll_events();
    gx_aurora::g_frame_open = !gx_aurora::g_should_quit && aurora_begin_frame();
    // The first frame is opened here rather than by a present, so it must also
    // be opened to the FIFO translation worker. Otherwise every batch falls
    // back to the packet sink, the core sink never sees the display copy that
    // triggers the first present, and no frame is ever recorded or shown.
    gx_aurora::set_initial_frame_recording(gx_aurora::g_frame_open);
    gx_aurora::install_platform_ops();
    return true;
}

void dol_aurora_set_overlay(DolAuroraOverlayFn draw, void* user) {
    gx_aurora::g_host_overlay = draw;
    gx_aurora::g_host_overlay_user = user;
}

void dol_aurora_set_event_observer(DolAuroraEventObserverFn observe, void* user) {
    gx_aurora::g_host_event_observer = observe;
    gx_aurora::g_host_event_user = user;
}

void dol_aurora_set_fast_forward(bool on) {
    gx_aurora::g_audio_discard.store(on, std::memory_order_relaxed);
    aurora_set_present_suppressed(on);
}

void dol_aurora_set_hold_redraw(bool redraw) { gx_aurora::g_hold_redraw = redraw; }

void dol_aurora_set_hold(DolAuroraHoldFn should_hold, void* user) {
    gx_aurora::g_host_hold = should_hold;
    gx_aurora::g_host_hold_user = user;
}

void dol_aurora_shutdown(void) {
    if (!gx_aurora::g_initialized)
        return;
    // The translation worker has to be gone before the frame is closed and the
    // device destroyed, and before process exit destroys a std::thread that is
    // still joinable. Stopping it here also lets the bytes it has not parsed
    // yet be recorded into the frame that is about to be submitted.
    gx_aurora::shadow_frontend_stop_worker();
    dol_platform_reset();
    if (gx_aurora::g_frame_open) {
        aurora_end_frame();
        gx_aurora::g_frame_open = false;
    }
#if GXRUNTIME_HAS_AURORA_RECOMP
    gx_aurora::trace_close_and_log();
    if (gx_aurora::g_gx_core_enabled) {
        const auto& gaps = gx_aurora::g_core_sink.counters();
        std::fprintf(stderr,
                     "[gx-core] shutdown: submitted=%llu rejected=%llu "
                     "failed=%d planned=%llu skipped=%llu noops=%llu cull_all=%llu "
                     "missing_vcd=%llu vertex_decode_failures=%llu "
                     "projection_missing=%llu payload_empty=%llu "
                     "walk_underivable=%llu stride_mismatch=%llu "
                     "topology_unsupported=%llu payload_overrun=%llu "
                     "topology_zero_quads=%llu topology_zero_triangles=%llu "
                     "topology_zero_triangle_strip=%llu "
                     "topology_zero_triangle_fan=%llu "
                     "topology_zero_lines=%llu topology_zero_line_strip=%llu "
                     "topology_zero_points=%llu topology_zero_unknown=%llu "
                     "array_unresolved=%llu array_out_of_bounds=%llu "
                     "unsupported_texgen=%llu texgen_count_overflow=%llu "
                     "texgen_count_5=%llu texgen_count_6=%llu "
                     "texgen_count_7=%llu texgen_count_8plus=%llu "
                     "texgen_emboss_cached_nbt=%llu texgen_source_normal=%llu "
                     "texgen_source_normal_default=%llu "
                     "texgen_source_colors=%llu texgen_source_binormal=%llu "
                     "texgen_source_tex47=%llu texgen_source_unknown=%llu "
                     "per_vertex_tex_mtx=%llu per_vertex_normal_matrix=%llu "
                     "unresolved_tex_matrix=%llu normals_ignored=%llu "
                     "lighting_ignored=%llu tlut_texture=%llu "
                     "alpha_compare_ignored=%llu tev_stages_over=%llu "
                     "tev_multi_texmap=%llu texcoord_scale_active=%llu "
                     "texcoord_scale_mismatch=%llu efb_copy_ignored=%llu "
                     "efb_copies=%llu efb_copy_depth=%llu "
                     "efb_display_copies=%llu fog_ignored=%llu "
                     "indirect_active=%llu indirect_ignored=%llu "
                     "logic_op_ignored=%llu dst_alpha_active=%llu "
                     "early_depth_active=%llu ztexture_active=%llu "
                     "ztexture_ignored=%llu\n",
                     gx_aurora::g_core_submitted, gx_aurora::g_core_rejected,
                     gx_aurora::g_shadow_frontend_failed ? 1 : 0,
                     gaps.draws_planned, gaps.draws_skipped,
                     gaps.draws_noop, gaps.cull_all_draws, gaps.missing_vcd,
                     gaps.vertex_decode_failures,
                     gaps.vertex_projection_missing,
                     gaps.vertex_payload_empty,
                     gaps.vertex_walk_underivable,
                     gaps.vertex_stride_mismatch,
                     gaps.vertex_topology_unsupported,
                     gaps.vertex_payload_overrun,
                     gaps.topology_zero_quads,
                     gaps.topology_zero_triangles,
                     gaps.topology_zero_triangle_strip,
                     gaps.topology_zero_triangle_fan,
                     gaps.topology_zero_lines,
                     gaps.topology_zero_line_strip,
                     gaps.topology_zero_points,
                     gaps.topology_zero_unknown,
                     gaps.vertex_array_unresolved,
                     gaps.vertex_array_out_of_bounds,
                     gaps.unsupported_texgen,
                     gaps.texgen_count_overflow,
                     gaps.texgen_count_5,
                     gaps.texgen_count_6,
                     gaps.texgen_count_7,
                     gaps.texgen_count_8plus,
                     gaps.texgen_emboss_cached_nbt,
                     gaps.texgen_source_normal,
                     gaps.texgen_source_normal_default,
                     gaps.texgen_source_colors,
                     gaps.texgen_source_binormal,
                     gaps.texgen_source_tex47,
                     gaps.texgen_source_unknown,
                     gaps.per_vertex_tex_mtx, gaps.per_vertex_normal_matrix,
                     gaps.unresolved_tex_matrix,
                     gaps.normals_ignored, gaps.lighting_ignored,
                     gaps.tlut_texture, gaps.alpha_compare_ignored,
                     gaps.tev_stages_over, gaps.tev_multi_texmap,
                     gaps.texcoord_scale_active,
                     gaps.texcoord_scale_mismatch,
                     gaps.efb_copy_ignored, gaps.efb_copies,
                     gaps.efb_copy_depth, gaps.efb_display_copies,
                     gaps.fog_ignored, gaps.indirect_active,
                     gaps.indirect_ignored,
                     gaps.logic_op_ignored, gaps.dst_alpha_active,
                     gaps.early_depth_active, gaps.ztexture_active,
                     gaps.ztexture_ignored);
        std::fprintf(stderr,
                     "[gx-core] toon: texgen_color_lit=%llu texgen_color_unlit=%llu "
                     "lit_light_missing=%llu\n",
                     gaps.texgen_color_lit, gaps.texgen_color_unlit,
                     gaps.lit_light_missing);
        const auto& texture_stats =
            aurora::gfx::gxcore::texture_cache_stats();
        std::fprintf(stderr,
                     "[gx-core] texture-cache: uploads=%llu hits=%llu "
                     "ci_uploads=%llu raw_fallback=%llu hashed=%llu "
                     "palette_hashed=%llu generation_hits=%llu "
                     "generation_fallbacks=%llu\n",
                     texture_stats.uploads, texture_stats.hits,
                     texture_stats.ci_uploads, texture_stats.raw_fallback,
                     texture_stats.hashed_lookups,
                     texture_stats.palette_hashes,
                     texture_stats.generation_hits,
                     texture_stats.generation_fallbacks);
    }
#endif
    if (gx_aurora::g_audio_stream != nullptr) {
        std::fprintf(stderr, "[audio] summary pushes=%llu starved=%llu stretched=%llu throttles=%llu\n",
                     gx_aurora::g_audio_push_count, gx_aurora::g_audio_starved_count,
                     gx_aurora::g_audio_stretched_count, gx_aurora::g_audio_throttle_count);
        SDL_DestroyAudioStream(gx_aurora::g_audio_stream);
        gx_aurora::g_audio_stream = nullptr;
        gx_aurora::g_audio_playing = false;
    }
    aurora_shutdown();
    gx_aurora::g_pending_textures = {};
    gx_aurora::g_pending_tluts = {};
    gx_aurora::g_draw_opcode_pending = false;
    gx_aurora::g_initialized = false;
}

} // extern "C"
