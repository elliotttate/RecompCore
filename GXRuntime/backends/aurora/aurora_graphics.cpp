// SPDX-License-Identifier: GPL-3.0-or-later
#include "aurora_backend_private.h"
#include <aurora/aurora.h>
#include <aurora/gfx.h>
#include <dolphin/gx.h>
#include <dolphin/gx/GXAurora.h>
#include <dolphin/gx/GXCommandList.h>
#include <dolphin/vi.h>
#include <gx/fifo.hpp>
#include <gx/gx.hpp>
#include <gx/recomp.hpp>
#include <gxruntime/guest_memory_dirty.h>
#include <SDL3/SDL_timer.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#if GXRUNTIME_HAS_AURORA_RECOMP
// gxcore substrate submission lives in the Aurora fork (lib/gfx/gxcore_draw.cpp),
// linked into this binary via aurora::gx.
namespace aurora::gfx::gxcore {
bool submit_draw_plan(const gxruntime::gxcore::DrawPlan& plan);
void copy_efb_to_texture(const gxruntime::gxcore::EfbCopyCommand& cmd);
void reset_texture_cache();
void note_frame_presented();
void set_texture_dirty_epoch_observer(
    bool (*observer)(uint32_t, uint32_t, uint64_t*));
unsigned long long texture_upload_count();
} // namespace aurora::gfx::gxcore
#endif

namespace gx_aurora {

// True while the FIFO worker translates. Then the worker requests each present
// itself, once per display copy (wait_for_present), and the copy observer must
// not: its request, raised mid-parse, led to a second, empty present.
std::atomic<bool> g_worker_mode{false};

#if GXRUNTIME_HAS_AURORA_RECOMP
bool core_texture_dirty_epoch(uint32_t address, uint32_t size,
                              uint64_t* epoch) {
    return dol_guest_memory_dirty_epoch(address, size, epoch);
}

void core_plan_observer(const gxruntime::gxcore::DrawPlan& plan, void*) {
    if (!plan.ok)
        return; // skip reasons are tallied in the sink gap counters
    // DOL_GXCORE_PLAN_TEX=<hex guest address>: print the TEV setup of draws
    // that sample that texture (first 12), for comparing a draw with Dolphin.
    static const long long s_plan_tex = [] {
        const char* env = std::getenv("DOL_GXCORE_PLAN_TEX");
        return env != nullptr ? std::strtoll(env, nullptr, 16) : -1ll;
    }();
    static int s_plan_tex_reports = 0;
    if (s_plan_tex >= 0 && s_plan_tex_reports < 12 && plan.has_texture &&
        (plan.tex_address & 0x3FFFFFFFu) == (static_cast<u32>(s_plan_tex) & 0x3FFFFFFFu)) {
        ++s_plan_tex_reports;
        const auto& k = plan.pipeline.shader;
        std::fprintf(stderr, "[plan-tex] present=%llu fmt=%u %ux%u tlut=%08X tlutfmt=%u stages=%u tev_valid=%u chans=%u lit=%u blend=%u src=%u dst=%u alpha=%u/%u/%u\n",
                     g_present_count, plan.tex_format, plan.tex_width, plan.tex_height,
                     plan.tlut_address, plan.tlut_format, k.num_tev_stages, k.tev_valid,
                     k.num_color_chans, k.lit_valid, plan.pipeline.blend_enable,
                     plan.pipeline.src_factor, plan.pipeline.dst_factor,
                     k.alpha_comp0, k.alpha_comp1, k.alpha_logic);
        for (unsigned n = 0; n < k.num_tev_stages && n < 16u; ++n) {
            const auto& t = k.tev_stages[n];
            std::fprintf(stderr, "[plan-tex]   stage%u cc a=%u b=%u c=%u d=%u bias=%u op=%u scale=%u dest=%u | ac a=%u b=%u c=%u d=%u dest=%u | ksel kc=%u ka=%u order tc=%u map=%u chan=%u en=%u swap tex=%u%u%u%u ras=%u%u%u%u\n",
                         n, t.cc_a, t.cc_b, t.cc_c, t.cc_d, t.cc_bias, t.cc_op, t.cc_scale, t.cc_dest,
                         t.ac_a, t.ac_b, t.ac_c, t.ac_d, t.ac_dest, t.ksel_kc, t.ksel_ka,
                         t.tevorders_texcoord, t.tevorders_texmap, t.tevorders_colorchan, t.tevorders_enable,
                         t.tex_swap[0], t.tex_swap[1], t.tex_swap[2], t.tex_swap[3],
                         t.ras_swap[0], t.ras_swap[1], t.ras_swap[2], t.ras_swap[3]);
        }
        if (plan.tlut_data != nullptr && plan.tlut_entries != 0u) {
            const auto* t = static_cast<const std::uint8_t*>(plan.tlut_data);
            std::fprintf(stderr, "[plan-tex]   tlut entries=%u:", plan.tlut_entries);
            for (unsigned e = 0; e < plan.tlut_entries && e < 16u; ++e)
                std::fprintf(stderr, " %02X%02X", t[e * 2u], t[e * 2u + 1u]);
            std::fprintf(stderr, "\n");
        }
        if (plan.tex_data != nullptr) {
            const auto* t = static_cast<const std::uint8_t*>(plan.tex_data);
            std::fprintf(stderr, "[plan-tex]   texels:");
            for (unsigned b = 0; b < 2048u && b < plan.tex_size; ++b)
                std::fprintf(stderr, " %02X", t[b]);
            std::fprintf(stderr, "\n");
        }
        std::fprintf(stderr, "[plan-tex]   chans captured=%02X has_color0=%u has_color1=%u litchan(mat/amb/en)=",
                     k.chan_captured_mask, k.has_color0, k.has_color1);
        for (unsigned j = 0; j < 4u; ++j)
            std::fprintf(stderr, " %u/%u/%u", k.litchan[j].matsource, k.litchan[j].ambsource, k.litchan[j].enablelighting);
        {
            const unsigned stride = gxruntime::gxcore::kVertexFloats;
            for (unsigned v = 0; v < plan.vertices.size() / stride && v < 4u;
                 ++v) {
              const float *p = plan.vertices.data() + v * stride;
              std::fprintf(stderr,
                           "[plan-tex]   v%u pos=%.1f,%.1f,%.1f "
                           "col0=%.3f,%.3f,%.3f,%.3f\n",
                           v, p[0], p[1], p[2], p[4], p[5], p[6], p[7]);
            }
        }
        const auto& vc = plan.constants;
        std::fprintf(stderr, " mat0=%d,%d,%d,%d amb0=%d,%d,%d,%d\n",
                     vc.materials[2][0], vc.materials[2][1], vc.materials[2][2], vc.materials[2][3],
                     vc.materials[0][0], vc.materials[0][1], vc.materials[0][2], vc.materials[0][3]);
        const auto& c = plan.pixel_constants;
        std::fprintf(stderr, "[plan-tex]   c0=%d,%d,%d,%d c1=%d,%d,%d,%d c2=%d,%d,%d,%d prev=%d,%d,%d,%d k0=%d,%d,%d,%d k1=%d,%d,%d,%d\n",
                     c.colors[1][0], c.colors[1][1], c.colors[1][2], c.colors[1][3],
                     c.colors[2][0], c.colors[2][1], c.colors[2][2], c.colors[2][3],
                     c.colors[3][0], c.colors[3][1], c.colors[3][2], c.colors[3][3],
                     c.colors[0][0], c.colors[0][1], c.colors[0][2], c.colors[0][3],
                     c.kcolors[0][0], c.kcolors[0][1], c.kcolors[0][2], c.kcolors[0][3],
                     c.kcolors[1][0], c.kcolors[1][1], c.kcolors[1][2], c.kcolors[1][3]);
    }
    if (aurora::gfx::gxcore::submit_draw_plan(plan))
        ++g_core_submitted;
    else
        ++g_core_rejected;
}

void core_copy_observer(const gxruntime::gxcore::EfbCopyCommand& cmd, void*) {
    aurora::gfx::gxcore::copy_efb_to_texture(cmd);
    if (cmd.format == 0xFu && !g_worker_mode.load(std::memory_order_relaxed))
        g_display_copy_pending = true;
}
#endif

void write_aurora_command(u16 command) {
    GXParam1u8(GX_AURORA);
    GXParam1u16(command);
}

void flush_pending_resource_metadata() {
    for (u8 slot = 0; slot < g_pending_tluts.size(); ++slot) {
        auto& pending = g_pending_tluts[slot];
        if (!pending.valid)
            continue;
        write_aurora_command(GX_AURORA_LOAD_TLUT);
        GXParam1u8(slot);
        GXCmd1u64(reinterpret_cast<u64>(pending.data));
        GXParam1u32(pending.format);
        GXParam1u16(pending.entries);
        GXParam1u32(pending.object_id);
        GXParam1u32(pending.data_version);
        pending.valid = false;
    }

    for (u8 slot = 0; slot < g_pending_textures.size(); ++slot) {
        auto& pending = g_pending_textures[slot];
        if (!pending.valid)
            continue;
        write_aurora_command(GX_AURORA_LOAD_TEXOBJ);
        GXParam1u8(slot);
        GXCmd1u64(reinterpret_cast<u64>(pending.data));
        GXParam1u32(pending.width);
        GXParam1u32(pending.height);
        GXParam1u32(pending.format);
        GXParam1u32(pending.tlut);
        GXParam1u8(pending.mipmap ? 1 : 0);
        GXParam1u32(pending.object_id);
        GXParam1u32(pending.data_version);
        pending.valid = false;
    }
}

bool gx_attr_to_cp_array(u32 attr, u8* out) {
    if (out == nullptr)
        return false;
    GXAttr cp_attr = static_cast<GXAttr>(attr);
    if (cp_attr == GX_VA_NBT)
        cp_attr = GX_VA_NRM;
    if (cp_attr < GX_VA_POS)
        return false;
    const u32 cp_index = static_cast<u32>(cp_attr) - static_cast<u32>(GX_VA_POS);
    if (cp_index >= DOL_GX_RECOMP_CP_ARRAY_COUNT)
        return false;
    *out = static_cast<u8>(cp_index);
    return true;
}

DolGuestAddressSpace map_address_space(
    aurora::gx::recomp::AddressSpace space) {
    switch (space) {
    case aurora::gx::recomp::AddressSpace::Virtual:
        return DOL_GUEST_ADDRESS_VIRTUAL;
    case aurora::gx::recomp::AddressSpace::Physical:
        return DOL_GUEST_ADDRESS_PHYSICAL;
    case aurora::gx::recomp::AddressSpace::Auto:
    default:
        return DOL_GUEST_ADDRESS_AUTO;
    }
}

DolGuestResourceKind map_resource_kind(aurora::gx::recomp::ResourceKind kind) {
    switch (kind) {
    case aurora::gx::recomp::ResourceKind::Fifo:
        return DOL_GUEST_RESOURCE_FIFO;
    case aurora::gx::recomp::ResourceKind::DisplayList:
        return DOL_GUEST_RESOURCE_DISPLAY_LIST;
    case aurora::gx::recomp::ResourceKind::VertexArray:
        return DOL_GUEST_RESOURCE_VERTEX_ARRAY;
    case aurora::gx::recomp::ResourceKind::Texture:
        return DOL_GUEST_RESOURCE_TEXTURE;
    case aurora::gx::recomp::ResourceKind::Tlut:
        return DOL_GUEST_RESOURCE_TLUT;
    case aurora::gx::recomp::ResourceKind::CopyDestination:
        return DOL_GUEST_RESOURCE_COPY_DESTINATION;
    case aurora::gx::recomp::ResourceKind::Generic:
    default:
        return DOL_GUEST_RESOURCE_GENERIC;
    }
}

bool aurora_guest_address_resolver_bridge(
    void*, std::uint32_t address, std::uint32_t size,
    aurora::gx::recomp::AddressSpace space,
    aurora::gx::recomp::ResourceKind resource, const void** data,
    std::uint32_t* available) {
    if (g_guest_address_resolver == nullptr)
        return false;
    return g_guest_address_resolver(
        g_guest_address_resolver_user, address, size, map_address_space(space),
        map_resource_kind(resource), data, available);
}

#if GXRUNTIME_HAS_AURORA_RECOMP
void shadow_frontend_fail_metadata(const char* reason, u32 attr,
                                   u32 guest_address, u32 value) {
    g_shadow_frontend_failed = true;
    std::fprintf(stderr,
                 "[aurora-recomp] shadow RetailGxFrontend rejected metadata; "
                 "reason=%s attr=%u guest=0x%08X value=%u; "
                 "live Aurora path remains active\n",
                 reason != nullptr ? reason : "unknown", attr, guest_address,
                 value);
}

unsigned long long trace_current_frame() {
    return g_trace_present_scope_frame != 0ull ? g_trace_present_scope_frame
                                               : g_present_count + 1ull;
}

bool trace_open_now() {
    if (g_trace_writer.is_open())
        return true;
    gxruntime::aurora_recomp::trace::TraceHeader header{};
    header.mem1_size = 0x01800000u;
    if (g_guest_address_resolver != nullptr) {
        const void* data = nullptr;
        u32 available = 0;
        if (g_guest_address_resolver(g_guest_address_resolver_user, 0x80000000u,
                                     sizeof header.game_id,
                                     DOL_GUEST_ADDRESS_VIRTUAL,
                                     DOL_GUEST_RESOURCE_GENERIC, &data,
                                     &available) &&
            data != nullptr && available >= sizeof header.game_id) {
            std::memcpy(header.game_id, data, sizeof header.game_id);
        }
    }
    if (!g_trace_writer.open(g_trace_path.c_str(), header)) {
        std::fprintf(stderr, "[trace] failed to open %s; recording disabled\n",
                     g_trace_path.c_str());
        g_trace_armed = false;
        return false;
    }
    return true;
}

bool trace_should_record() {
    if (!g_trace_armed)
        return false;
    const unsigned long long frame = trace_current_frame();
    if (frame < g_trace_first_frame || frame > g_trace_last_frame)
        return false;
    if (!trace_open_now())
        return false;
    if (!g_trace_frame_begun) {
        g_trace_writer.frame_begin(static_cast<u32>(frame));
        g_trace_frame_begun = true;
        ++g_trace_frames_recorded;
    }
    return true;
}

void trace_record_mem_update(u32 address, u32 size, const void* data) {
    if (size == 0 || data == nullptr || !trace_should_record())
        return;
    const u64 key = (static_cast<u64>(address) << 32) | size;
    u64 hash = 1469598103934665603ull;
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (u32 i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    const auto it = g_trace_mem_dedup.find(key);
    if (it != g_trace_mem_dedup.end() && it->second == hash)
        return;
    g_trace_mem_dedup[key] = hash;
    g_trace_writer.mem_update(address, data, size);
}

void trace_close_and_log() {
    if (!g_trace_writer.is_open()) {
        g_trace_armed = false;
        return;
    }
    const unsigned long long records = g_trace_writer.records_written();
    const unsigned long long bytes = g_trace_writer.bytes_written();
    const bool ok = g_trace_writer.close();
    g_trace_armed = false;
    std::fprintf(stderr,
                 "[trace] wrote %s frames=%llu records=%llu bytes=%llu%s\n",
                 g_trace_path.c_str(), g_trace_frames_recorded, records, bytes,
                 ok ? "" : " (WRITE ERRORS)");
}

void trace_on_present() {
    if (g_trace_armed && trace_should_record()) {
        const AuroraStats* stats = aurora_get_stats();
        gxruntime::aurora_recomp::trace::PresentStats ps{};
        ps.frame_index = static_cast<u32>(trace_current_frame());
        ps.queued_pipelines = stats->queuedPipelines;
        ps.created_pipelines = stats->createdPipelines;
        ps.draw_call_count = stats->drawCallCount;
        ps.merged_draw_call_count = stats->mergedDrawCallCount;
        ps.last_vert_size = stats->lastVertSize;
        ps.last_uniform_size = stats->lastUniformSize;
        ps.last_index_size = stats->lastIndexSize;
        ps.last_storage_size = stats->lastStorageSize;
        ps.last_texture_upload_size = stats->lastTextureUploadSize;
        g_trace_writer.present_stats(ps);
    }
    if (g_trace_armed && g_present_count >= g_trace_last_frame)
        trace_close_and_log();
    g_trace_present_scope_frame = 0ull;
    g_trace_frame_begun = false;
}

// Defined below, next to the write path it batches. The guest-visible
// synchronizations call it before they change or consume front-end state.
static void shadow_frontend_flush(void);

// The FIFO's translation runs on a worker rather than on the thread that
// executes the guest.
//
// The front end's state - the parse position, the GX register image and its own
// buffer - is touched only from that worker, so nothing guards it; the lock
// guards the hand-off buffer alone. The guest-visible synchronizations drain
// first, through shadow_frontend_flush, and that is the only place the main
// thread waits; the byte budget only wakes the worker. Aurora's submission is a
// queue drained by its own render worker, so calling the sink from here is the
// path Aurora expects, and the trace writer stays on the calling thread because
// it is a file writer and the parse is what feeds it.
// docs/status/CURRENT.md, 2026-09-22.
namespace {

std::mutex g_fifo_worker_mutex;
std::condition_variable g_fifo_worker_cv;
std::condition_variable g_fifo_worker_idle_cv;
std::vector<std::uint8_t> g_fifo_handoff;
std::thread g_fifo_worker_thread;
bool g_fifo_worker_started = false;
bool g_fifo_worker_stop = false;
bool g_fifo_work_pending = false;
bool g_fifo_worker_idle = true;
// True only while the worker is blocked on g_fifo_worker_cv. The enqueue path
// runs once per guest FIFO write; signalling a worker that is already awake
// cost 2.1% of the game thread in pthread_cond_signal (iOS simulator sample).
bool g_fifo_worker_sleeping = false;
std::uint64_t g_fifo_appended = 0;
std::uint64_t g_fifo_parsed = 0;

// Aurora records a frame into a packet that exists only between begin_frame and
// end_frame, and aurora_backend_present() runs both ends in one function: the
// finished frame is submitted and then the next one is opened. The translation
// worker records draws on its own thread, so without this lock a worker batch
// can straddle that window and dereference the packet pointer Aurora has just
// cleared - measured on 2026-09-22 as EXC_BAD_ACCESS at 0x28 inside
// aurora::gfx::get_render_target_size(), which is g_recordingFrame being null
// (the release build compiles the CHECK out). The main thread holds this lock
// across the transition and the worker holds it across one batch, so neither
// can observe the other's half-state. It cannot deadlock: the main thread's
// only wait on the worker - the drain at the guest-visible barriers - is never
// taken from inside the transition.
std::mutex g_aurora_recording_mutex;
// Whether Aurora has a frame packet open for the worker to record into. Read by
// the main thread in the write path and by the worker in the batch gate, always
// under g_aurora_recording_mutex or on the thread that owns the frame.
bool g_aurora_recording_open = false;
// True only while the main thread is inside the transition above. The drain
// reads it on the main thread alone, so it needs no lock of its own.
bool g_aurora_recording_in_transition = false;
// Batches that arrived with no frame to record into and were parsed by the
// packet sink instead. Reported so a run that starts dropping draws is not
// silent, because before this gate existed that state was a null dereference.
std::atomic<std::uint64_t> g_aurora_unframed_batches = 0;

// Display-copy hand-off (worker mode). The frontend stops a parse right after a
// display copy; the worker then leaves the recording lock and waits here until
// the main thread has presented, so the next frame's first draws are recorded
// into the next frame. Before this the worker ran on into them and they landed
// in the frame being presented: the Wind Waker title intermittently showed its
// logo and island over a black sky and sea (the sky is drawn first).
std::condition_variable g_present_cv;
std::uint64_t g_presents_done = 0;      // under g_fifo_worker_mutex
bool g_worker_waiting_present = false;  // under g_fifo_worker_mutex
std::uint64_t g_present_wait_seen = 0;  // g_presents_done when the wait began
void wait_for_present(std::unique_lock<std::mutex>& recording);

void g_fifo_translate(std::vector<std::uint8_t>& batch) {
    if (batch.empty())
        return;
    bool flushed = true;
    // One batch, one frame: the lock keeps the main thread from closing the
    // packet underneath this recording, and the gate below keeps this batch
    // from recording into a packet that does not exist yet.
    std::unique_lock<std::mutex> recording(g_aurora_recording_mutex);
    bool record_into_aurora = g_gx_core_enabled && g_aurora_recording_open;
    if (g_gx_core_enabled && !record_into_aurora)
        g_aurora_unframed_batches.fetch_add(1, std::memory_order_relaxed);
    // Feed the front end in the slices the single-threaded path flushes at.
    // Its per-flush event trace holds DOL_GX_RECOMP_MAX_TRACE_EVENTS (8,192)
    // events and silently drops the rest, so one heavy batch (60 KB measured in
    // the Outset play scene) overflowed it, failed the flush and, because the
    // failure is sticky, froze every later frame. A command split across two
    // slices is carried over by the parser's partial-command handling.
    constexpr std::size_t kSlice = 1024u;
    for (std::size_t offset = 0; offset < batch.size() && flushed; offset += kSlice) {
        const std::size_t size = std::min(kSlice, batch.size() - offset);
        const std::span<const std::uint8_t> bytes(batch.data() + offset, size);
        if (!g_shadow_frontend.write_fifo(bytes)) {
            flushed = false;
            break;
        }
        if (record_into_aurora)
            flushed = g_shadow_frontend.flush(&g_core_sink);
        else
            flushed = g_shadow_frontend.flush(&g_shadow_packet_sink);
        // A parse that stopped at a display copy resumes on the buffered rest
        // only after the present, and may stop again at the next one.
        while (flushed && g_shadow_frontend.display_copy_stopped()) {
            wait_for_present(recording);
            record_into_aurora = g_gx_core_enabled && g_aurora_recording_open;
            flushed = record_into_aurora ? g_shadow_frontend.flush(&g_core_sink)
                                         : g_shadow_frontend.flush(&g_shadow_packet_sink);
        }
    }
    if (!flushed) {
        std::lock_guard<std::mutex> lock(g_fifo_worker_mutex);
        if (!g_shadow_frontend_failed) {
            // The single-threaded path names its failure; the worker has to as
            // well, because a failed front end stops every later present.
            std::fprintf(stderr,
                         "[gx-core] worker frontend rejected FIFO: %s "
                         "(opcode=0x%02X offset=%llu a=0x%08X b=0x%08X "
                         "c=0x%08X d=0x%08X) consumer=%s recording=%d "
                         "batch=%zu present=%llu\n",
                         g_shadow_frontend.last_error() != nullptr
                             ? g_shadow_frontend.last_error()
                             : "none",
                         static_cast<unsigned>(g_shadow_frontend.last_error_opcode()),
                         static_cast<unsigned long long>(
                             g_shadow_frontend.last_error_offset()),
                         g_shadow_frontend.last_error_a(),
                         g_shadow_frontend.last_error_b(),
                         g_shadow_frontend.last_error_c(),
                         g_shadow_frontend.last_error_d(),
                         record_into_aurora
                             ? (g_core_sink.failure_reason() != nullptr
                                    ? g_core_sink.failure_reason()
                                    : "none")
                             : (g_shadow_packet_sink.failure_reason() != nullptr
                                    ? g_shadow_packet_sink.failure_reason()
                                    : "none"),
                         record_into_aurora ? 1 : 0, batch.size(),
                         static_cast<unsigned long long>(g_present_count));
        }
        g_shadow_frontend_failed = true;
    }
}

void g_fifo_worker_main() {
    for (;;) {
        std::vector<std::uint8_t> batch;
        std::uint64_t parsed = 0;
        {
            std::unique_lock<std::mutex> lock(g_fifo_worker_mutex);
            while (!g_fifo_work_pending && !g_fifo_worker_stop) {
                g_fifo_worker_sleeping = true;
                g_fifo_worker_cv.wait(lock);
                g_fifo_worker_sleeping = false;
            }
            if (!g_fifo_work_pending)
                return;
            g_fifo_work_pending = false;
            batch.swap(g_fifo_handoff);
            parsed = g_fifo_appended;
        }
        // A slow batch holds the game thread at its next draw-done; say what
        // the worker made in it (pipelines, decoded textures) so the cause
        // is in the session log. DOL_GX_SLOW_BATCH_MS sets the bar (20).
        static const long slow_ms = [] {
            const char* env = std::getenv("DOL_GX_SLOW_BATCH_MS");
            return env != nullptr ? std::strtol(env, nullptr, 10) : 20L;
        }();
        const auto t0 = std::chrono::steady_clock::now();
        const uint32_t pipelines0 = aurora_get_stats()->createdPipelines;
        const unsigned long long uploads0 = aurora::gfx::gxcore::texture_upload_count();
        g_fifo_translate(batch);
        const long ms = static_cast<long>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                              std::chrono::steady_clock::now() - t0)
                                              .count());
        if (slow_ms > 0 && ms >= slow_ms) {
            std::fprintf(stderr, "[gx-slow] batch_ms=%ld bytes=%zu pipelines=%u textures=%llu\n", ms,
                         batch.size(), aurora_get_stats()->createdPipelines - pipelines0,
                         aurora::gfx::gxcore::texture_upload_count() - uploads0);
            // A batch that held the game for a tenth of a second or more: its
            // first bytes (the GX commands), so what it waited on can be told.
            if (ms >= 100) {
                char hex[3 * 256 + 1];
                size_t n = 0;
                for (size_t i = 0; i < batch.size() && i < 256; ++i)
                    n += static_cast<size_t>(std::snprintf(hex + n, sizeof hex - n, "%02X ", batch[i]));
                hex[n] = '\0';
                std::fprintf(stderr, "[gx-slow-bytes] %s\n", hex);
            }
        }
        {
            std::lock_guard<std::mutex> lock(g_fifo_worker_mutex);
            g_fifo_parsed = parsed;
            if (!g_fifo_work_pending)
                g_fifo_worker_idle = true;
        }
        g_fifo_worker_idle_cv.notify_all();
    }
}

// Called by the worker with the recording lock held, after a parse stopped at a
// display copy. Lock order is recording, then g_fifo_worker_mutex, everywhere.
void wait_for_present(std::unique_lock<std::mutex>& recording) {
    std::unique_lock<std::mutex> lock(g_fifo_worker_mutex);
    if (g_fifo_worker_stop)
        return;
    const std::uint64_t seen = g_presents_done;
    g_worker_waiting_present = true;
    g_present_wait_seen = seen;
    // One request per stop: the main thread presents at its next GX write, or
    // in a drain that is waiting on this worker (woken below), whichever comes
    // first; the other finds the request already served.
    g_display_copy_pending = true;
    g_fifo_worker_idle_cv.notify_all();
    recording.unlock();
    g_present_cv.wait(lock, [seen] { return g_presents_done != seen || g_fifo_worker_stop; });
    g_worker_waiting_present = false;
    lock.unlock();
    recording.lock();
}

void g_fifo_worker_start() {
    if (g_fifo_worker_started || trace_should_record())
        return;
    // DOL_GX_FIFO_WORKER=0 translates on the guest thread (A/B and diagnosis).
    static const bool disabled = [] {
        const char* env = std::getenv("DOL_GX_FIFO_WORKER");
        return env != nullptr && env[0] == '0';
    }();
    if (disabled)
        return;
    g_fifo_worker_started = true;
    g_worker_mode = true;
    g_fifo_worker_thread = std::thread(g_fifo_worker_main);
}

// Stops the translation worker and waits for it, so that nothing is recording
// into Aurora while the device is destroyed and so that no worker thread
// outlives the process's statics. A std::thread still joinable when exit
// destroys it calls std::terminate: every rendered run of this worker before
// this existed ended in SIGABRT after a normal guest stop (measured 2026-09-22),
// and a batch in flight during device teardown is a use-after-free on top of it.
void g_fifo_publish_local();

void g_fifo_worker_stop_and_join() {
    g_fifo_publish_local();
    std::thread worker;
    {
        std::lock_guard<std::mutex> lock(g_fifo_worker_mutex);
        if (!g_fifo_worker_started)
            return;
        g_fifo_worker_stop = true;
        // Wake the worker even with no bytes pending, so it is not left in the
        // predicate wait when this joins it.
        g_fifo_work_pending = true;
        worker.swap(g_fifo_worker_thread);
        g_fifo_worker_started = false;
        g_worker_mode = false;
    }
    g_fifo_worker_cv.notify_all();
    g_present_cv.notify_all();
    if (worker.joinable())
        worker.join();
    std::lock_guard<std::mutex> lock(g_fifo_worker_mutex);
    // A later initialization may start the worker again.
    g_fifo_worker_stop = false;
    g_fifo_worker_idle = true;
}

// Gather-pipe bytes collect on the main thread and reach the worker in batches:
// taking the worker mutex once per 1-4 byte guest write cost several percent of
// the game thread. Every barrier that needs the worker to have seen the bytes
// (g_fifo_drain, shutdown) publishes the local batch first.
// A fixed buffer: a vector insert per 1-8 byte write was 1.7 percent of the
// game thread (iPad simulator sample, heavy Outset view).
constexpr std::size_t kFifoLocalBatch = 1024u;
std::uint8_t g_fifo_local[kFifoLocalBatch + 8u];
std::size_t g_fifo_local_size = 0;

void g_fifo_publish_local() {
    if (g_fifo_local_size == 0)
        return;
    bool wake;
    {
        std::lock_guard<std::mutex> lock(g_fifo_worker_mutex);
        g_fifo_handoff.insert(g_fifo_handoff.end(), g_fifo_local,
                              g_fifo_local + g_fifo_local_size);
        ++g_fifo_appended;
        g_fifo_work_pending = true;
        g_fifo_worker_idle = false;
        wake = g_fifo_worker_sleeping;
    }
    g_fifo_local_size = 0;
    if (wake)
        g_fifo_worker_cv.notify_one();
}

// DOL_GX_FIFO_BATCH=N publishes every N bytes instead of kFifoLocalBatch, so
// the worker sees the stream cut at other places (a test of the parser's
// carry-over across batches; the guest's work does not depend on it).
static std::size_t fifo_local_batch_limit() {
    static const std::size_t limit = [] {
        const char* env = std::getenv("DOL_GX_FIFO_BATCH");
        const long value = env != nullptr ? std::strtol(env, nullptr, 10) : 0;
        return value > 0 && value <= static_cast<long>(kFifoLocalBatch)
                   ? static_cast<std::size_t>(value)
                   : kFifoLocalBatch;
    }();
    return limit;
}

inline void g_fifo_enqueue(const std::uint8_t* bytes, u8 size) {
    std::memcpy(g_fifo_local + g_fifo_local_size, bytes, size);
    g_fifo_local_size += size;
    if (g_fifo_local_size >= fifo_local_batch_limit())
        g_fifo_publish_local();
}

// Waits until the worker has translated everything appended so far. Called at
// the barriers and only there.
static void g_fifo_drain_impl();
std::atomic<unsigned long long> g_timing_drain_us{0};
std::atomic<unsigned long long> g_timing_present_us{0};
std::atomic<unsigned long long> g_timing_end_frame_us{0};
std::atomic<unsigned long long> g_timing_draws{0};
std::atomic<unsigned long long> g_timing_present_in_drain_us{0};
bool g_timing_in_drain = false;  // main thread only
static unsigned long long timing_now_us() {
    return static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
static void g_fifo_drain() {
    const unsigned long long start = timing_now_us();
    g_timing_in_drain = true;
    g_fifo_drain_impl();
    g_timing_in_drain = false;
    g_timing_drain_us += timing_now_us() - start;
}
static void g_fifo_drain_impl() {
    g_fifo_publish_local();
    std::unique_lock<std::mutex> lock(g_fifo_worker_mutex);
    // Called from the guest's synchronization points, which never run inside
    // the present transition; if one ever does, waiting here would deadlock
    // against the recording lock the transition holds. Say so rather than hang.
    if (g_aurora_recording_in_transition) {
        static bool s_warned = false;
        if (!s_warned) {
            s_warned = true;
            std::fprintf(stderr,
                         "[gx] FIFO drain requested inside the frame "
                         "transition; not waiting for the worker\n");
        }
        return;
    }
    const std::uint64_t target = g_fifo_appended;
    if (g_fifo_worker_idle && g_fifo_parsed >= target)
        return;
    g_fifo_work_pending = true;
    g_fifo_worker_cv.notify_one();
    for (;;) {
        g_fifo_worker_idle_cv.wait(lock, [target] {
            return (g_fifo_worker_idle && g_fifo_parsed >= target) ||
                   g_worker_waiting_present;
        });
        if (g_fifo_worker_idle && g_fifo_parsed >= target)
            return;
        if (g_presents_done != g_present_wait_seen) {
            // Already presented for this stop; the worker is waking up.
            g_fifo_worker_idle_cv.wait_for(lock, std::chrono::milliseconds(1));
            continue;
        }
        // The worker stopped at a display copy and waits for the present that
        // ends the frame; this thread is the one that presents.
        lock.unlock();
        // No present has happened since the wait began, so this is the one.
        g_display_copy_pending = false;
        aurora_backend_present();
        lock.lock();
    }
}

}  // namespace

void shadow_frontend_set_array(u32 attr, u32 guest_address, u8 stride) {
    if (!g_shadow_frontend_enabled || g_shadow_frontend_failed)
        return;
    // The mirror must not run ahead of bytes the parser has not seen, or a
    // draw already in the FIFO would be decoded with this array's successor.
    shadow_frontend_flush();
    u8 cp_attr = 0;
    if (!gx_attr_to_cp_array(attr, &cp_attr)) {
        shadow_frontend_fail_metadata("unsupported GX array attribute", attr,
                                      guest_address, stride);
        return;
    }
    const u32 physical = dol_gx_recomp_guest_to_physical(guest_address);
    if (trace_should_record())
        g_trace_writer.set_array(cp_attr, physical, stride);
    if (!g_shadow_frontend.set_cp_array(cp_attr, physical, stride)) {
        shadow_frontend_fail_metadata("failed to mirror CP array state", attr,
                                      guest_address, stride);
    }
}

bool frontend_guest_address_resolver_bridge(
    void*, u32 address, u32 size, DolGuestAddressSpace space,
    DolGuestResourceKind resource, DolGuestResolvedRange* out) {
    if (g_guest_address_resolver == nullptr || out == nullptr)
        return false;
    const void* data = nullptr;
    u32 available = 0;
    if (!g_guest_address_resolver(g_guest_address_resolver_user, address, size,
                                  space, resource, &data, &available) ||
        data == nullptr || available < size)
        return false;
    *out = {
        .data = const_cast<void*>(data),
        .address = address,
        .size = size,
        .available = available,
        .space = space,
        .resource = resource,
    };
    trace_record_mem_update(address, size, data);
    return true;
}

void shadow_frontend_call_display_list(const void* data, u32 size) {
    if (!g_shadow_frontend_enabled || g_shadow_frontend_failed)
        return;
    // Same ordering rule as the array mirror: the HLE path parses the list
    // immediately, so anything already in the FIFO has to be parsed first.
    shadow_frontend_flush();
    const std::span<const std::uint8_t> bytes(
        static_cast<const std::uint8_t*>(data), size);
    if (g_gx_core_enabled) {
        if (!g_shadow_frontend.write_display_list(bytes, &g_core_sink)) {
            g_shadow_frontend_failed = true;
            std::fprintf(stderr,
                         "[gx-core] frontend rejected HLE display list "
                         "(%u bytes): %s (opcode=0x%02X)\n",
                         size,
                         g_shadow_frontend.last_error() != nullptr
                             ? g_shadow_frontend.last_error()
                             : "none",
                         static_cast<unsigned>(
                             g_shadow_frontend.last_error_opcode()));
        }
        return;
    }
    if (!g_shadow_frontend.write_display_list(bytes, &g_shadow_packet_sink)) {
        g_shadow_frontend_failed = true;
        std::fprintf(stderr,
                     "[aurora-recomp] shadow RetailGxFrontend rejected HLE "
                     "display list (%u bytes): parse_error=%s opcode=0x%02X "
                     "offset=%llu a=%u b=%u c=%u d=%u; live Aurora path "
                     "remains active\n",
                     size,
                     g_shadow_frontend.last_error() != nullptr
                         ? g_shadow_frontend.last_error()
                         : "none",
                     static_cast<unsigned>(
                         g_shadow_frontend.last_error_opcode()),
                     static_cast<unsigned long long>(
                         g_shadow_frontend.last_error_offset()),
                     g_shadow_frontend.last_error_a(),
                     g_shadow_frontend.last_error_b(),
                     g_shadow_frontend.last_error_c(),
                     g_shadow_frontend.last_error_d());
    }
}

// The FIFO is parsed in batches rather than once per guest store.
//
// A store that leaves a partial command in the buffer is re-parsed by the next
// flush, so flushing per store re-scans that partial once per byte written, and
// every store also pays the flush path's notification, packet-emission and drain
// scans. The batch is bounded in bytes, and the guest-visible synchronizations
// flush first: the CP array mirror and the HLE display-list path here, and the
// draw-done commit in the host through dol_platform_gx_flush. Nothing else the
// guest can observe depends on the parse having happened, because the FIFO's own
// read-back is serviced by the front end's state and the parse does not run
// backwards. docs/status/CURRENT.md, 2026-09-22.
static constexpr std::size_t kShadowFrontendFlushBytes = 1024u;
static std::size_t g_shadow_frontend_pending_bytes;
static u64 g_shadow_frontend_last_write_value;
static u8 g_shadow_frontend_last_write_size;

static void shadow_frontend_flush(void);

// shadow_frontend_write for a word the running translation worker takes (the
// front end live, a 1, 2, 4 or 8-byte write): the same bookkeeping, and the
// word's bytes appended to the worker's batch in guest order directly.
static inline void shadow_frontend_enqueue_word(u64 value, u8 size) {
    g_shadow_frontend_last_write_value = value;
    g_shadow_frontend_last_write_size = size;
    g_shadow_frontend_pending_bytes += size;
    std::uint8_t* const out = g_fifo_local + g_fifo_local_size;
    switch (size) {
    case 1:
        out[0] = static_cast<std::uint8_t>(value);
        break;
    case 2: {
        const std::uint16_t word = __builtin_bswap16(static_cast<std::uint16_t>(value));
        std::memcpy(out, &word, 2);
        break;
    }
    case 4: {
        const std::uint32_t word = __builtin_bswap32(static_cast<std::uint32_t>(value));
        std::memcpy(out, &word, 4);
        break;
    }
    default: {
        const std::uint64_t word = __builtin_bswap64(value);
        std::memcpy(out, &word, 8);
        break;
    }
    }
    g_fifo_local_size += size;
    if (g_fifo_local_size >= fifo_local_batch_limit())
        g_fifo_publish_local();
}

void shadow_frontend_write(u64 value, u8 size) {
    if (!g_shadow_frontend_enabled || g_shadow_frontend_failed)
        return;
    if (!g_fifo_worker_started)
        g_fifo_worker_start();
    std::uint8_t bytes[8] = {};
    switch (size) {
    case 1:
        bytes[0] = static_cast<std::uint8_t>(value);
        break;
    case 2:
        bytes[0] = static_cast<std::uint8_t>(value >> 8u);
        bytes[1] = static_cast<std::uint8_t>(value);
        break;
    case 4:
        bytes[0] = static_cast<std::uint8_t>(value >> 24u);
        bytes[1] = static_cast<std::uint8_t>(value >> 16u);
        bytes[2] = static_cast<std::uint8_t>(value >> 8u);
        bytes[3] = static_cast<std::uint8_t>(value);
        break;
    case 8:
        for (unsigned i = 0; i < 8; ++i)
            bytes[i] =
                static_cast<std::uint8_t>(value >> ((7u - i) * 8u));
        break;
    default:
        return;
    }
    const std::span<const std::uint8_t> fragment(bytes, size);
    g_shadow_frontend_last_write_value = value;
    g_shadow_frontend_last_write_size = size;
    g_shadow_frontend_pending_bytes += size;
    if (g_fifo_worker_started) {
        g_fifo_enqueue(bytes, size);
        return;
    }
    if (!g_shadow_frontend.write_fifo(fragment)) {
        g_shadow_frontend_failed = true;
        std::fprintf(stderr,
                     "[gx] shadow RetailGxFrontend refused a %u-byte FIFO "
                     "fragment\n",
                     static_cast<unsigned>(size));
        return;
    }
    if (g_shadow_frontend_pending_bytes >= kShadowFrontendFlushBytes)
        shadow_frontend_flush();
}

static void shadow_frontend_flush(void) {
    if (!g_shadow_frontend_enabled || g_shadow_frontend_failed)
        return;
    if (g_fifo_worker_started) {
        g_fifo_drain();
        return;
    }
    g_shadow_frontend_pending_bytes = 0u;
    const u64 value = g_shadow_frontend_last_write_value;
    const u8 size = g_shadow_frontend_last_write_size;
    if (g_gx_core_enabled) {
        if (!g_shadow_frontend.flush(&g_core_sink)) {
            g_shadow_frontend_failed = true;
            std::fprintf(stderr,
                         "[gx-core] frontend rejected FIFO after %llu byte(s): "
                         "%s (opcode=0x%02X offset=%llu a=0x%08X b=0x%08X "
                         "c=0x%08X d=0x%08X); consumer=%s\n",
                         g_fifo_bytes,
                         g_shadow_frontend.last_error() != nullptr
                             ? g_shadow_frontend.last_error()
                             : "none",
                         static_cast<unsigned>(
                             g_shadow_frontend.last_error_opcode()),
                         static_cast<unsigned long long>(
                             g_shadow_frontend.last_error_offset()),
                         g_shadow_frontend.last_error_a(),
                         g_shadow_frontend.last_error_b(),
                         g_shadow_frontend.last_error_c(),
                         g_shadow_frontend.last_error_d(),
                         g_core_sink.failure_reason() != nullptr
                             ? g_core_sink.failure_reason()
                             : "none");
        }
        return;
    }
    if (!g_shadow_frontend.flush(&g_shadow_packet_sink)) {
        g_shadow_frontend_failed = true;
        const auto& failed = g_shadow_packet_sink.failed_packet();
        std::fprintf(stderr,
                     "[aurora-recomp] shadow RetailGxFrontend rejected FIFO "
                     "after %llu byte(s), packets=%llu "
                     "(stream=%llu state=%llu resource=%llu draw=%llu); "
                     "write_size=%u write_value=0x%016llX; "
                     "parse_error=%s parse_opcode=0x%02X "
                     "parse_offset=%llu pending=%llu "
                     "detail=(%u,%u,%u,%u); "
                     "reason=%s seq=%llu event=%s(%u) "
                     "a=%u b=%u c=%u d=%u; "
                     "live Aurora path remains active\n",
                     g_fifo_bytes, g_shadow_packet_sink.packets(),
                     g_shadow_packet_sink.stream_packets(),
                     g_shadow_packet_sink.state_packets(),
                     g_shadow_packet_sink.resource_packets(),
                     g_shadow_packet_sink.draw_packets(),
                     static_cast<unsigned>(size),
                     static_cast<unsigned long long>(value),
                     g_shadow_frontend.last_error() != nullptr
                         ? g_shadow_frontend.last_error()
                         : "none",
                     static_cast<unsigned>(
                         g_shadow_frontend.last_error_opcode()),
                     static_cast<unsigned long long>(
                         g_shadow_frontend.last_error_offset()),
                     static_cast<unsigned long long>(
                         g_shadow_frontend.pending_fifo_size()),
                     g_shadow_frontend.last_error_a(),
                     g_shadow_frontend.last_error_b(),
                     g_shadow_frontend.last_error_c(),
                     g_shadow_frontend.last_error_d(),
                     g_shadow_packet_sink.failure_reason() != nullptr
                         ? g_shadow_packet_sink.failure_reason()
                         : "frontend-parse",
                     static_cast<unsigned long long>(failed.sequence),
                     gxruntime::aurora_recomp::trace_event_name(
                         failed.event.kind),
                     static_cast<unsigned>(failed.event.kind), failed.event.a,
                     failed.event.b, failed.event.c, failed.event.d);
    }
}

// Called by the host at the draw-done commit, where the guest is about to
// observe the PE finish, and by the platform layer's flush hook.
void shadow_frontend_flush_pending(void) {
    shadow_frontend_flush();
}

// Called by the platform layer before the device is destroyed.
void shadow_frontend_stop_worker(void) {
    g_fifo_worker_stop_and_join();
}

void set_initial_frame_recording(bool open) {
    std::lock_guard<std::mutex> recording(g_aurora_recording_mutex);
    g_aurora_recording_open = open;
}

// A present opens the next frame, and only the core sink's display copy
// triggers a present. When begin_frame fails - the window is minimized or the
// iOS app is in the background, so the surface is released - no frame is open,
// every batch goes to the packet sink, no display copy reaches the core sink,
// and nothing would ever open a frame again: the game keeps running behind a
// blank window. The worker counts those unframed batches; when that count moves
// the main thread retries begin_frame here, once per new unframed batch.
void reopen_frame_if_unframed() {
    static std::uint64_t s_seen_unframed = 0;
    if (g_frame_open || g_should_quit)
        return;
    const std::uint64_t unframed =
        g_aurora_unframed_batches.load(std::memory_order_relaxed);
    if (unframed == s_seen_unframed)
        return;
    s_seen_unframed = unframed;
    std::lock_guard<std::mutex> recording(g_aurora_recording_mutex);
    poll_events();
    if (host_wants_hold()) {
        const Uint64 start = SDL_GetTicks();
        std::fprintf(stderr, "[gfx] guest held while the host is inactive\n");
        while (!g_should_quit && host_wants_hold()) {
            SDL_Delay(50);
            poll_events();
        }
        std::fprintf(stderr, "[gfx] guest released after %.1f s\n",
                     (SDL_GetTicks() - start) / 1000.0);
    }
    if (g_should_quit)
        return;
    g_frame_open = aurora_begin_frame();
    g_aurora_recording_open = g_frame_open;
    if (g_frame_open)
        std::fprintf(stderr, "[gfx] frame reopened after %llu unframed batch(es)\n",
                     static_cast<unsigned long long>(unframed));
}

unsigned long long shadow_transform_frame_number() {
    return g_frame_open ? g_present_count + 1ull : g_present_count;
}

void log_transform_matrix(const char* label, unsigned index,
                           const float* values) {
    std::fprintf(stderr,
                 "[gfx] draw-transform %s[%u] "
                 "%.8g %.8g %.8g %.8g | %.8g %.8g %.8g %.8g | "
                 "%.8g %.8g %.8g %.8g\n",
                 label, index, values[0], values[1], values[2], values[3],
                 values[4], values[5], values[6], values[7], values[8],
                 values[9], values[10], values[11]);
}

void shadow_transform_observer(
    const gxruntime::aurora_recomp::ConsumedDraw& draw,
    unsigned long long cumulative_draw, void*) {
    const std::size_t frame_draw = g_shadow_transform_next_draw_index++;
    if (!g_shadow_transform_log_enabled ||
        g_shadow_transform_log_count >= g_shadow_transform_log_limit)
        return;
    const unsigned long long frame = shadow_transform_frame_number();
    if (g_shadow_transform_log_min_frame != 0ull &&
        frame < g_shadow_transform_log_min_frame)
        return;
    if (g_shadow_transform_log_frame != 0ull &&
        g_shadow_transform_log_frame != frame)
        return;
    if (g_shadow_transform_log_draw >= 0 &&
        static_cast<unsigned long>(g_shadow_transform_log_draw) != frame_draw)
        return;
    if (g_shadow_transform_log_sequence_enabled &&
        g_shadow_transform_log_sequence != draw.sequence)
        return;
    if (g_shadow_light_log_lit_only) {
        bool lit = false;
        for (unsigned c = 0; c < 4 && !lit; ++c) {
            if ((draw.chan_reg_mask & (1u << (5u + c))) == 0u)
                continue;
            const std::uint32_t ctrl = draw.chan_regs[5u + c];
            const std::uint32_t mask =
                ((ctrl >> 2u) & 0xFu) | (((ctrl >> 11u) & 0xFu) << 4u);
            lit = ((ctrl >> 1u) & 0x1u) != 0u && mask != 0u;
        }
        if (!lit)
            return;
    }

    ++g_shadow_transform_log_count;
    std::uint32_t pn_used_mask = 0u;
    if ((draw.transform_flags &
         gxruntime::aurora_recomp::kDrawTransformPayloadPnMatrixValid) != 0u &&
        draw.payload_pn_matrix_mask != 0u) {
        pn_used_mask = draw.payload_pn_matrix_mask;
    } else if (draw.current_pn_matrix < DOL_GX_RECOMP_POSITION_MATRIX_COUNT) {
        pn_used_mask = 1u << draw.current_pn_matrix;
    }
    const std::uint32_t pn_valid_used =
        pn_used_mask & draw.position_matrix_valid_mask;
    std::fprintf(stderr,
                 "[gfx] draw-transform frame=%llu draw=%zu seq=%llu "
                 "total=%llu prim=0x%02X fmt=%u count=%u vsize=%u "
                 "payload=%zu cull=%d tex=%u:0x%08X flags=0x%X "
                 "current_pn=%u payload_pn_mask=0x%03X pn_used=0x%03X "
                 "pn_valid=0x%03X pos_valid=0x%03X arrays=%u active=0x%04X\n",
                 frame, frame_draw,
                 static_cast<unsigned long long>(draw.sequence), cumulative_draw,
                 draw.primitive, draw.vtx_fmt, draw.vertex_count,
                 draw.vertex_size, draw.vertex_payload.size(),
                 draw.cull_all ? 1 : 0, draw.texture.slot,
                 draw.texture.address, draw.transform_flags,
                 draw.current_pn_matrix, draw.payload_pn_matrix_mask,
                 pn_used_mask, pn_valid_used,
                 draw.position_matrix_valid_mask, draw.array_input_count,
                 draw.active_array_mask);
    if ((draw.transform_flags &
         gxruntime::aurora_recomp::kDrawTransformViewportValid) != 0u) {
        std::fprintf(stderr,
                     "[gfx] draw-transform viewport %.8g %.8g %.8g %.8g "
                     "%.8g %.8g\n",
                     draw.viewport[0], draw.viewport[1], draw.viewport[2],
                     draw.viewport[3], draw.viewport[4], draw.viewport[5]);
    } else {
        std::fprintf(stderr, "[gfx] draw-transform viewport invalid\n");
    }
    if ((draw.transform_flags &
         gxruntime::aurora_recomp::kDrawTransformProjectionValid) != 0u) {
        std::fprintf(stderr,
                     "[gfx] draw-transform projection type=%u %.8g %.8g "
                     "%.8g %.8g %.8g %.8g\n",
                     draw.projection_type, draw.projection[0],
                     draw.projection[1], draw.projection[2],
                     draw.projection[3], draw.projection[4],
                     draw.projection[5]);
    } else {
        std::fprintf(stderr, "[gfx] draw-transform projection invalid\n");
    }
    for (unsigned i = 0; i < DOL_GX_RECOMP_POSITION_MATRIX_COUNT; ++i) {
        const std::uint32_t bit = 1u << i;
        if ((pn_used_mask & bit) == 0u)
            continue;
        if ((draw.position_matrix_valid_mask & bit) == 0u) {
            std::fprintf(stderr,
                         "[gfx] draw-transform pnmtx[%u] invalid\n", i);
            continue;
        }
        log_transform_matrix("pnmtx", i, draw.position_matrices[i]);
    }
    if (!g_shadow_light_log_enabled)
        return;
    std::fprintf(stderr,
                 "[gfx] draw-light frame=%llu draw=%zu chan_mask=0x%03X "
                 "numchans=%u amb0=0x%08X amb1=0x%08X mat0=0x%08X "
                 "mat1=0x%08X\n",
                 frame, frame_draw, draw.chan_reg_mask,
                 (draw.chan_reg_mask & 0x1u) != 0u ? draw.chan_regs[0] : 0u,
                 draw.chan_regs[1], draw.chan_regs[2], draw.chan_regs[3],
                 draw.chan_regs[4]);
    static const char* const kChanNames[4] = {"cctrl0", "cctrl1", "actrl0",
                                               "actrl1"};
    for (unsigned c = 0; c < 4; ++c) {
        if ((draw.chan_reg_mask & (1u << (5u + c))) == 0u)
            continue;
        const std::uint32_t ctrl = draw.chan_regs[5u + c];
        const unsigned matsrc = ctrl & 0x1u;
        const unsigned lit = (ctrl >> 1u) & 0x1u;
        const unsigned ambsrc = (ctrl >> 6u) & 0x1u;
        const unsigned diffunc = (ctrl >> 7u) & 0x3u;
        const unsigned attnfunc = (ctrl >> 9u) & 0x3u;
        const unsigned light_mask =
            ((ctrl >> 2u) & 0xFu) | (((ctrl >> 11u) & 0xFu) << 4u);
        std::fprintf(stderr,
                     "[gfx] draw-light %s=0x%08X lit=%u matsrc=%u ambsrc=%u "
                     "diffunc=%u attnfunc=%u lights=0x%02X\n",
                     kChanNames[c], ctrl, lit, matsrc, ambsrc, diffunc,
                     attnfunc, light_mask);
    }
    for (unsigned i = 0; i < DOL_GX_RECOMP_LIGHT_COUNT; ++i) {
        if (draw.light_word_mask[i] == 0u)
            continue;
        float f[16];
        std::memcpy(f, draw.light_words[i], sizeof(f));
        std::fprintf(stderr,
                     "[gfx] draw-light light[%u] mask=0x%04X col=0x%08X "
                     "cos=%.8g,%.8g,%.8g dist=%.8g,%.8g,%.8g "
                     "pos=%.8g,%.8g,%.8g dir=%.8g,%.8g,%.8g\n",
                     i, draw.light_word_mask[i], draw.light_words[i][3],
                     f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11], f[12],
                     f[13], f[14], f[15]);
    }
}
#endif

} // namespace gx_aurora

extern "C" {

static void aurora_backend_present_impl(void);

// Timed for the per-second frame diagnostics (dol_aurora_frame_timing).
void aurora_backend_present(void) {
    const unsigned long long start = gx_aurora::timing_now_us();
    aurora_backend_present_impl();
    const unsigned long long spent = gx_aurora::timing_now_us() - start;
    gx_aurora::g_timing_present_us += spent;
    if (gx_aurora::g_timing_in_drain)
        gx_aurora::g_timing_present_in_drain_us += spent;
}

void dol_aurora_frame_timing(DolAuroraFrameTiming* out) {
    out->presents = gx_aurora::g_present_count;
    out->drain_us = gx_aurora::g_timing_drain_us - gx_aurora::g_timing_present_in_drain_us;
    out->present_us = gx_aurora::g_timing_present_us;
    out->end_frame_us = gx_aurora::g_timing_end_frame_us;
    out->draws = gx_aurora::g_timing_draws;
    out->audio_throttles = gx_aurora::g_audio_throttle_count;
    out->audio_dropped = gx_aurora::g_audio_dropped_count;
    out->shown = aurora_get_shown_frames();
    AuroraFrameInterpTotals interp{};
    aurora_get_frame_interp_totals(&interp);
    out->interp_frames = interp.frames;
    out->interp_interpolated = interp.interpolated;
    out->interp_draws = interp.draws;
    out->interp_rejected = interp.rejected;
    out->interp_unmatched = interp.unmatched;
    out->audio_queued_ms = 0;
    if (gx_aurora::g_audio_stream != nullptr && gx_aurora::g_audio_sample_rate != 0u) {
        const int queued = SDL_GetAudioStreamQueued(gx_aurora::g_audio_stream);
        out->audio_queued_ms =
            queued > 0 ? static_cast<int>(queued * 1000ll / (gx_aurora::g_audio_sample_rate * 4ll)) : 0;
    }
#if GXRUNTIME_HAS_AURORA_RECOMP
    out->display_copies = gx_aurora::g_shadow_frontend.display_copies();
#else
    out->display_copies = 0;
#endif
}

// A present the worker requested at a display copy is taken at the main
// thread's next GX write. If the worker finishes the frame while the guest is
// idle waiting for the next retrace, there is no write until the game starts
// its next frame, so the finished frame waited up to a frame time: two or
// three frames a second stayed on screen for 60 ms while the game was on time
// ([late] lines, 2026-09-26). The host calls this at every retrace as well.
void aurora_backend_service_present(void) {
#if GXRUNTIME_HAS_AURORA_RECOMP
    if (!gx_aurora::g_initialized || !gx_aurora::g_gx_core_enabled)
        return;
    if (gx_aurora::g_display_copy_pending.load(std::memory_order_relaxed) &&
        gx_aurora::g_display_copy_pending.exchange(false))
        aurora_backend_present();
#endif
}

static void aurora_backend_present_impl(void) {
    if (!gx_aurora::g_initialized)
        return;
    // Everything below up to the next begin_frame runs with no frame packet for
    // the translation worker to record into. The lock is what makes that window
    // invisible to the worker: it waits here for a batch in flight, and a batch
    // that starts afterwards sees g_aurora_recording_open false and parses
    // without offering draws to a renderer that has no frame.
    std::unique_lock<std::mutex> aurora_recording(gx_aurora::g_aurora_recording_mutex);
    gx_aurora::g_aurora_recording_open = false;
    gx_aurora::g_aurora_recording_in_transition = true;
#if GXRUNTIME_HAS_AURORA_RECOMP
    if (gx_aurora::g_gx_core_enabled && !gx_aurora::g_shadow_frontend_failed)
        gx_aurora::g_core_sink.flush_frame();
    if (gx_aurora::g_gx_core_enabled) {
        static int s_cutscene_diag = -1;
        if (s_cutscene_diag < 0)
            s_cutscene_diag = std::getenv("STRIKERS_CUTSCENE_DIAG") != nullptr ? 1 : 0;
        if (s_cutscene_diag) {
            static gxruntime::gxcore::GapCounters s_prev{};
            const gxruntime::gxcore::GapCounters& c = gx_aurora::g_core_sink.counters();
            std::fprintf(stderr,
                "[cutscene] present=%llu planned=%llu skipped=%llu cull=%llu "
                "missing_vcd=%llu vtx_fail=%llu\n",
                gx_aurora::g_present_count + 1,
                c.draws_planned - s_prev.draws_planned,
                c.draws_skipped - s_prev.draws_skipped,
                c.cull_all_draws - s_prev.cull_all_draws,
                c.missing_vcd - s_prev.missing_vcd,
                c.vertex_decode_failures - s_prev.vertex_decode_failures);
            s_prev = c;
        }
    }
#endif
    static const char *capture_path = std::getenv("DOL_GXCORE_CAPTURE_PATH");
    static const unsigned long long capture_frame = [] {
      const char *value = std::getenv("DOL_GXCORE_CAPTURE_FRAME");
      return value ? std::strtoull(value, nullptr, 10) : 0ull;
    }();
    const bool capture =
        capture_path && capture_frame == gx_aurora::g_present_count + 1;
    if (gx_aurora::g_frame_open) {
      if (capture)
        aurora_request_framebuffer_readback();
      gx_aurora::run_host_overlay();
      gx_aurora::g_timing_draws += aurora_get_stats()->drawCallCount;
      const unsigned long long end_frame_start = gx_aurora::timing_now_us();
      aurora_end_frame();
      gx_aurora::g_timing_end_frame_us +=
          gx_aurora::timing_now_us() - end_frame_start;
      gx_aurora::g_frame_open = false;
      // Opt-in diagnostic: read the real EFB, including native simulation
      // frames that never pass through the interpolation capture path.
      if (capture) {
        const std::uint8_t *rgba = nullptr;
        u32 width = 0, height = 0;
        bool ready = false;
        for (unsigned attempt = 0; attempt < 20000; ++attempt) {
          if (aurora_take_framebuffer_readback(&rgba, &width, &height)) {
            ready = true;
            break;
          }
          aurora_pump_framebuffer_readback();
          std::this_thread::sleep_for(std::chrono::microseconds(250));
        }
        if (ready) {
          if (FILE *file = std::fopen(capture_path, "wb")) {
            std::fprintf(file,
                         "P7\nWIDTH %u\nHEIGHT %u\nDEPTH 4\nMAXVAL "
                         "255\nTUPLTYPE RGB_ALPHA\nENDHDR\n",
                         width, height);
            const auto count =
                std::fwrite(rgba, 4, std::size_t(width) * height, file);
            std::fclose(file);
            std::fprintf(
                stderr,
                "[gx-capture] frame=%llu pixels=%zu size=%ux%u path=%s\n",
                capture_frame, count, width, height, capture_path);
          }
        } else
          std::fprintf(stderr, "[gx-capture] frame=%llu readback timed out\n",
                       capture_frame);
      }
    }
    ++gx_aurora::g_present_count;
#if GXRUNTIME_HAS_AURORA_RECOMP
    aurora::gfx::gxcore::note_frame_presented();
#endif
    if (gx_aurora::g_frame_pacing_log &&
        (gx_aurora::g_present_count <= 10 || (gx_aurora::g_present_count % 60) == 0)) {
        const AuroraStats* stats = aurora_get_stats();
        std::fprintf(stderr,
                     "[frame-pacing] frame=%llu fps=%.1f draws=%u "
                     "texture-upload=%u fifo=%llu\n",
                     gx_aurora::g_present_count, aurora_get_fps(), stats->drawCallCount,
                     stats->lastTextureUploadSize, gx_aurora::g_fifo_bytes);
    }
    if (gx_aurora::g_graphics_log) {
        const AuroraStats* s = aurora_get_stats();
        std::fprintf(stderr,
                     "[gfxN] present=%llu draws=%u merged=%u verts=%u indices=%u "
                     "fifo=%llu\n",
                     gx_aurora::g_present_count, s->drawCallCount, s->mergedDrawCallCount,
                     s->lastVertSize, s->lastIndexSize, gx_aurora::g_fifo_bytes);
    }
#if GXRUNTIME_HAS_AURORA_RECOMP
    gx_aurora::g_trace_present_scope_frame = gx_aurora::g_present_count;
    if (gx_aurora::g_shadow_frontend_enabled && !gx_aurora::g_gx_core_enabled) {
        gx_aurora::g_shadow_packet_sink.flush_assembly();
        const AuroraStats* astats = aurora_get_stats();
        const unsigned long long fe_draw_total = gx_aurora::g_shadow_packet_sink.draw_packets();
        const unsigned long long fe_vtx_total = gx_aurora::g_shadow_packet_sink.vertex_inputs();
        const unsigned long long fe_zero_draw_total =
            gx_aurora::g_shadow_frontend.zero_vertex_draws();
        const unsigned long long fe_frame_draws =
            fe_draw_total - gx_aurora::g_shadow_last_draw_total;
        const unsigned long long fe_frame_verts =
            fe_vtx_total - gx_aurora::g_shadow_last_vertex_total;
        const unsigned long long fe_frame_zero_draws =
            fe_zero_draw_total - gx_aurora::g_shadow_last_zero_draw_total;
        gx_aurora::g_shadow_last_draw_total = fe_draw_total;
        gx_aurora::g_shadow_last_vertex_total = fe_vtx_total;
        gx_aurora::g_shadow_last_zero_draw_total = fe_zero_draw_total;
        const unsigned long long fe_rawvert_total =
            gx_aurora::g_shadow_packet_sink.raw_vertex_bytes();
        const unsigned long long fe_topoidx_total =
            gx_aurora::g_shadow_packet_sink.topology_index_bytes();
        const unsigned long long fe_storage_total =
            gx_aurora::g_shadow_packet_sink.storage_bytes();
        const unsigned long long fe_frame_rawvert =
            fe_rawvert_total - gx_aurora::g_shadow_last_rawvert_total;
        const unsigned long long fe_frame_topoidx =
            fe_topoidx_total - gx_aurora::g_shadow_last_topoidx_total;
        const unsigned long long fe_frame_storage =
            fe_storage_total - gx_aurora::g_shadow_last_storage_total;
        gx_aurora::g_shadow_last_rawvert_total = fe_rawvert_total;
        gx_aurora::g_shadow_last_topoidx_total = fe_topoidx_total;
        gx_aurora::g_shadow_last_storage_total = fe_storage_total;
        const bool have_prev = gx_aurora::g_shadow_prev_frame_valid;
        const bool draw_match =
            !have_prev ||
            (!gx_aurora::g_shadow_frontend_failed &&
             gx_aurora::g_shadow_prev_draws + gx_aurora::g_shadow_prev_zero_draws ==
                 astats->drawCallCount);
        if (!draw_match)
            ++gx_aurora::g_shadow_draw_mismatch_frames;
        const unsigned long long vert_align_slack =
            4ull * gx_aurora::g_shadow_prev_draws + 4ull;
        const bool vert_extent_match =
            !have_prev ||
            (!gx_aurora::g_shadow_frontend_failed &&
             astats->lastVertSize >= gx_aurora::g_shadow_prev_rawvert &&
             (astats->lastVertSize - gx_aurora::g_shadow_prev_rawvert) <=
                 vert_align_slack);
        if (!vert_extent_match)
            ++gx_aurora::g_shadow_vert_extent_mismatch_frames;
        if (gx_aurora::g_graphics_log && have_prev &&
            (gx_aurora::g_present_count <= 10 || gx_aurora::g_present_count % 60 == 0 ||
             !draw_match || !vert_extent_match)) {
            std::fprintf(stderr,
                         "[gfx] shadow-consume failed=%d packets=%llu "
                         "draws=%llu textures=%llu copies=%llu spans=%llu "
                         "array-inputs=%llu/%llu (resolved/unresolved) "
                         "assembled=%llu/%llu elems=%llu (ok/fail)\n",
                         gx_aurora::g_shadow_frontend_failed ? 1 : 0,
                         gx_aurora::g_shadow_packet_sink.packets(),
                         fe_draw_total,
                         gx_aurora::g_shadow_packet_sink.texture_count(),
                         gx_aurora::g_shadow_packet_sink.copy_count(),
                         gx_aurora::g_shadow_packet_sink.indexed_span_count(),
                         gx_aurora::g_shadow_packet_sink.resolved_array_inputs(),
                         gx_aurora::g_shadow_packet_sink.unresolved_array_inputs(),
                         gx_aurora::g_shadow_packet_sink.assembled_draws() -
                             gx_aurora::g_shadow_packet_sink.assemble_failed_draws(),
                         gx_aurora::g_shadow_packet_sink.assemble_failed_draws(),
                         gx_aurora::g_shadow_packet_sink.assembled_elements());
            std::fprintf(stderr,
                         "[gfx] shadow-diff frame=%llu fe_draws=%llu "
                         "fe_zdraws=%llu aurora_draws=%u match=%d "
                         "fe_verts=%llu mismatch_frames=%llu\n",
                         gx_aurora::g_shadow_prev_frame_index, gx_aurora::g_shadow_prev_draws,
                         gx_aurora::g_shadow_prev_zero_draws, astats->drawCallCount,
                         draw_match ? 1 : 0, gx_aurora::g_shadow_prev_verts,
                         gx_aurora::g_shadow_draw_mismatch_frames);
            std::fprintf(stderr,
                         "[gfx] shadow-extent frame=%llu fe_vertB=%llu "
                         "aurora_vertB=%u vmatch=%d vmiss=%llu | fe_idxB=%llu "
                         "aurora_idxB=%u | fe_storeB=%llu aurora_storeB=%u\n",
                         gx_aurora::g_shadow_prev_frame_index, gx_aurora::g_shadow_prev_rawvert,
                         astats->lastVertSize, vert_extent_match ? 1 : 0,
                         gx_aurora::g_shadow_vert_extent_mismatch_frames,
                         gx_aurora::g_shadow_prev_topoidx, astats->lastIndexSize,
                         gx_aurora::g_shadow_prev_storage, astats->lastStorageSize);
        }
        gx_aurora::g_shadow_prev_frame_valid = true;
        gx_aurora::g_shadow_prev_frame_index = gx_aurora::g_present_count;
        gx_aurora::g_shadow_prev_draws = fe_frame_draws;
        gx_aurora::g_shadow_prev_zero_draws = fe_frame_zero_draws;
        gx_aurora::g_shadow_prev_verts = fe_frame_verts;
        gx_aurora::g_shadow_prev_rawvert = fe_frame_rawvert;
        gx_aurora::g_shadow_prev_topoidx = fe_frame_topoidx;
        gx_aurora::g_shadow_prev_storage = fe_frame_storage;
    }
    gx_aurora::trace_on_present();
#endif
    if (gx_aurora::g_graphics_log && (gx_aurora::g_present_count <= 10 || gx_aurora::g_present_count % 60 == 0)) {
        const AuroraStats* stats = aurora_get_stats();
        const auto& gx = aurora::gx::g_gxState;
        std::fprintf(stderr,
                     "[gfx] frame=%llu fifo=%llu draws=%u merged=%u verts=%u "
                     "uniforms=%u indices=%u textures=%u fps=%.1f\n",
                     gx_aurora::g_present_count, gx_aurora::g_fifo_bytes, stats->drawCallCount,
                     stats->mergedDrawCallCount, stats->lastVertSize,
                     stats->lastUniformSize, stats->lastIndexSize,
                     stats->lastTextureUploadSize, aurora_get_fps());
        std::fprintf(
            stderr,
            "[gfx] parsed-state tev=%u texgen=%u chans=%u ind=%u cull=%u "
            "blend=%u/%u/%u depth=%u/%u/%u write=%u/%u "
            "viewport=(%.1f,%.1f %.1fx%.1f %.3f..%.3f) "
            "scissor=(%d,%d %dx%d) pnmtx=%u\n",
            gx.numTevStages, gx.numTexGens, gx.numChans, gx.numIndStages,
            static_cast<unsigned>(gx.cullMode),
            static_cast<unsigned>(gx.blendMode),
            static_cast<unsigned>(gx.blendFacSrc),
            static_cast<unsigned>(gx.blendFacDst), gx.depthCompare,
            static_cast<unsigned>(gx.depthFunc), gx.depthUpdate, gx.colorUpdate,
            gx.alphaUpdate, gx.logicalViewport.left, gx.logicalViewport.top,
            gx.logicalViewport.width, gx.logicalViewport.height,
            gx.logicalViewport.znear, gx.logicalViewport.zfar,
            gx.logicalScissor.x, gx.logicalScissor.y,
            gx.logicalScissor.width, gx.logicalScissor.height, gx.currentPnMtx);
        std::fprintf(stderr, "[gfx] projection");
        for (unsigned row = 0; row < 4; row++)
            for (unsigned col = 0; col < 4; col++)
                std::fprintf(stderr, " %.5g", gx.proj[row][col]);
        fputc('\n', stderr);
        std::fprintf(stderr, "[gfx] pnmtx[%u]", gx.currentPnMtx);
        const auto& pos = gx.pnMtx[gx.currentPnMtx].pos;
        const float* pos_values = reinterpret_cast<const float*>(&pos);
        for (unsigned i = 0; i < 12; i++)
            std::fprintf(stderr, " %.5g", pos_values[i]);
        fputc('\n', stderr);
        for (unsigned attr : {9u, 11u, 13u}) {
            const auto& fmt = gx.vtxFmts[0].attrs[attr];
            const auto& array = gx.arrays[attr];
            std::fprintf(stderr,
                         "[gfx] attr=%u desc=%u cnt=%u type=%u frac=%u "
                         "array=%p size=%u stride=%u le=%u\n",
                         attr, static_cast<unsigned>(gx.vtxDesc[attr]),
                         static_cast<unsigned>(fmt.cnt),
                         static_cast<unsigned>(fmt.type), fmt.frac, array.data,
                         array.size, array.stride, array.le);
        }
        if (gx.numTexGens != 0) {
            const auto& tcg = gx.tcgs[0];
            std::fprintf(stderr,
                         "[gfx] tcg0 type=%u src=%u mtx=%u post=%u norm=%u\n",
                         static_cast<unsigned>(tcg.type),
                         static_cast<unsigned>(tcg.src),
                         static_cast<unsigned>(tcg.mtx),
                         static_cast<unsigned>(tcg.postMtx), tcg.normalize);
        }
        if (gx.numTevStages != 0) {
            const auto& tev = gx.tevStages[0];
            std::fprintf(
                stderr,
                "[gfx] tev0 texcoord=%u texmap=%u chan=%u "
                "color=%u,%u,%u,%u alpha=%u,%u,%u,%u\n",
                static_cast<unsigned>(tev.texCoordId),
                static_cast<unsigned>(tev.texMapId),
                static_cast<unsigned>(tev.channelId),
                static_cast<unsigned>(tev.colorPass.a),
                static_cast<unsigned>(tev.colorPass.b),
                static_cast<unsigned>(tev.colorPass.c),
                static_cast<unsigned>(tev.colorPass.d),
                static_cast<unsigned>(tev.alphaPass.a),
                static_cast<unsigned>(tev.alphaPass.b),
                static_cast<unsigned>(tev.alphaPass.c),
                static_cast<unsigned>(tev.alphaPass.d));
        }
    }
    const unsigned long long present_fifo = gx_aurora::g_fifo_bytes;
    gx_aurora::g_fifo_bytes = 0;
    gx_aurora::poll_events();
    // The host may hold the guest at this frame boundary: a menu or layout
    // editor is open, or the app is leaving the foreground. Events keep being
    // pumped so the host UI stays live, the last frame stays on screen, and
    // no GPU work is issued until the hold ends.
    if (!gx_aurora::g_should_quit && gx_aurora::host_wants_hold()) {
        const Uint64 start = SDL_GetTicks();
        std::fprintf(stderr, "[gfx] guest held by the host at present=%llu\n",
                     gx_aurora::g_present_count);
        while (!gx_aurora::g_should_quit && gx_aurora::host_wants_hold()) {
            // dol_aurora_set_hold_redraw: a frame with no game drawing presents
            // the last picture, and the host overlay (a menu) is drawn over
            // it. The worker records nothing meanwhile (recording is closed
            // until the begin_frame below) and the display paces the loop.
            if (gx_aurora::g_hold_redraw && aurora_begin_frame()) {
                gx_aurora::run_host_overlay();
                aurora_end_frame();
            } else {
                SDL_Delay(16);
            }
            gx_aurora::poll_events();
        }
        std::fprintf(stderr, "[gfx] guest released after %.1f s\n",
                     (SDL_GetTicks() - start) / 1000.0);
    }
    if (!gx_aurora::g_should_quit) {
        gx_aurora::g_frame_open = aurora_begin_frame();
        // The frame the worker records into exists from here until the next
        // present's end_frame, and this is what opens it to the worker.
        gx_aurora::g_aurora_recording_open = gx_aurora::g_frame_open;
        // One shout if draws were ever offered to a window with no frame open.
        // That state used to be a null dereference; it is now a fallback parse,
        // and a run that hits it should say so rather than lose draws quietly.
        static bool s_unframed_warned = false;
        const std::uint64_t unframed =
            gx_aurora::g_aurora_unframed_batches.load(std::memory_order_relaxed);
        if (unframed != 0 && !s_unframed_warned) {
            s_unframed_warned = true;
            std::fprintf(stderr,
                         "[gx] %llu FIFO batch(es) arrived with no frame packet; "
                         "parsed without offering their draws to the renderer "
                         "(present=%llu)\n",
                         static_cast<unsigned long long>(unframed),
                         gx_aurora::g_present_count);
        }
#if GXRUNTIME_HAS_AURORA_RECOMP
        gx_aurora::g_shadow_transform_next_draw_index = 0;
#endif
        if (gx_aurora::g_graphics_log && !gx_aurora::g_frame_open)
            std::fprintf(stderr,
                         "[gfx] WARN aurora_begin_frame()=false present=%llu "
                         "fifo_this_interval=%llu (FIFO will accumulate "
                         "undrained)\n",
                         gx_aurora::g_present_count, present_fifo);
    }
    if (gx_aurora::g_graphics_log && present_fifo > 50000ull)
        std::fprintf(stderr, "[gfx] LARGE present=%llu fifo_this_interval=%llu\n",
                     gx_aurora::g_present_count, present_fifo);
    // The transition is over: a packet exists again (or the window is not
    // presentable), and the worker may record.
    gx_aurora::g_aurora_recording_in_transition = false;
    aurora_recording.unlock();
    {
        std::lock_guard<std::mutex> lock(gx_aurora::g_fifo_worker_mutex);
        ++gx_aurora::g_presents_done;
    }
    gx_aurora::g_present_cv.notify_all();
}

void aurora_backend_mark_gx_begin(void) {
    gx_aurora::g_draw_opcode_pending = true;
}

void aurora_backend_call_display_list(const void* data, u32 size) {
    if (!gx_aurora::g_initialized || data == nullptr || size == 0)
        return;
#if GXRUNTIME_HAS_AURORA_RECOMP
    gx_aurora::shadow_frontend_call_display_list(data, size);
    if (gx_aurora::trace_should_record())
        gx_aurora::g_trace_writer.call_display_list(0u, data, size);
    if (gx_aurora::g_gx_core_enabled) {
        gx_aurora::g_fifo_bytes += size;
        return;
    }
#endif
    gx_aurora::flush_pending_resource_metadata();
    aurora::gx::fifo::write_data(data, size);
    gx_aurora::g_fifo_bytes += size;
}

// Parses whatever the FIFO write path has buffered. The host calls this at the
// draw-done commit, where the guest is about to observe GPU progress, so the
// translation is never behind a wait it is supposed to satisfy. Without Aurora
// there is no front end to drain.
void aurora_backend_gx_flush(void) {
#if GXRUNTIME_HAS_AURORA_RECOMP
    if (!gx_aurora::g_initialized)
        return;
    gx_aurora::shadow_frontend_flush_pending();
#endif
}

void aurora_backend_gx_write(u64 value, u8 size) {
#if GXRUNTIME_HAS_AURORA_RECOMP
    // Every gather-pipe word in play: the recorder open, the translation
    // worker running, nothing traced and no present waiting on this write.
    // The path below does just this for such a word - counts it and hands its
    // bytes to the worker's batch - through three calls and a byte-by-byte
    // copy, about 3 percent of the game thread at native 60 Hz.
    if (gx_aurora::g_initialized && gx_aurora::g_gx_core_enabled && gx_aurora::g_frame_open &&
        !gx_aurora::g_trace_armed && gx_aurora::g_fifo_worker_started &&
        gx_aurora::g_shadow_frontend_enabled && !gx_aurora::g_shadow_frontend_failed &&
        (size == 1u || size == 2u || size == 4u || size == 8u) &&
        !gx_aurora::g_display_copy_pending.load(std::memory_order_relaxed)) {
        gx_aurora::g_fifo_bytes += size;
        gx_aurora::shadow_frontend_enqueue_word(value, size);
        return;
    }
#endif
    if (!gx_aurora::g_initialized)
        return;
    gx_aurora::g_fifo_bytes += size;
#if GXRUNTIME_HAS_AURORA_RECOMP
    gx_aurora::shadow_frontend_write(value, size);
    if (gx_aurora::g_trace_armed && gx_aurora::trace_should_record())
        gx_aurora::g_trace_writer.gx_write(size, value);
    if (gx_aurora::g_gx_core_enabled && !gx_aurora::g_frame_open)
        gx_aurora::reopen_frame_if_unframed();
    // A relaxed load first: the flag is almost always clear, and an atomic
    // exchange at every gather-pipe write is a read-modify-write the core
    // pays for even when there is nothing to take. Only the exchange claims
    // the request, so a present is still taken exactly once.
    if (gx_aurora::g_gx_core_enabled &&
        gx_aurora::g_display_copy_pending.load(std::memory_order_relaxed) &&
        gx_aurora::g_display_copy_pending.exchange(false)) {
        aurora_backend_present();
    }
    if (gx_aurora::g_gx_core_enabled)
        return;
#endif

    static u32 s_last_zmode = 0x40000017;
    static u32 s_last_cmode0 = 0x410004BC;
    static bool s_cull_all_active = false;
    static u8 s_last_opcode = 0;

    bool display_copy = false;
    switch (size) {
    case 1: {
        const u8 command = static_cast<u8>(value);
        s_last_opcode = command;
        if (gx_aurora::g_draw_opcode_pending && command >= 0x80u) {
            gx_aurora::g_draw_opcode_pending = false;
            gx_aurora::flush_pending_resource_metadata();
            if (gx_aurora::g_force_untextured) {
                GXSetCullMode(GX_CULL_NONE);
                GXSetZMode(GX_FALSE, GX_ALWAYS, GX_FALSE);
                GXSetBlendMode(GX_BM_NONE, GX_BL_ONE, GX_BL_ZERO, GX_LO_CLEAR);
                GXSetColorUpdate(GX_TRUE);
                GXSetAlphaUpdate(GX_TRUE);
                GXSetNumTexGens(0);
                GXSetNumChans(1);
                GXSetNumTevStages(1);
                GXSetTevOrder(GX_TEVSTAGE0, GX_TEXCOORD_NULL, GX_TEXMAP_NULL,
                              GX_COLOR0A0);
                GXSetTevOp(GX_TEVSTAGE0, GX_PASSCLR);
            }
        }
        GXParam1u8(command);
        break;
    }
    case 2: GXParam1u16(static_cast<u16>(value)); break;
    case 4: {
        u32 val32 = static_cast<u32>(value);

        if (s_last_opcode == 0x61) {
            u8 regId = (val32 >> 24) & 0xFF;
            display_copy = regId == 0x52u && (val32 & (1u << 14)) != 0;

            if (regId == 0x40) {
                s_last_zmode = val32;
                if (s_cull_all_active) {
                    val32 &= ~0x11;
                }
                GXParam1u32(val32);
            } else if (regId == 0x41) {
                s_last_cmode0 = val32;
                if (s_cull_all_active) {
                    val32 &= ~0x18;
                }
                GXParam1u32(val32);
            } else if (regId == 0x00) {
                u32 hwCull = (val32 >> 14) & 3;
                if (hwCull == 3) {
                    val32 &= ~(3u << 14);
                    GXParam1u32(val32);

                    if (!s_cull_all_active) {
                        s_cull_all_active = true;
                        GXParam1u8(0x61);
                        GXParam1u32(s_last_zmode & ~0x11);
                        GXParam1u8(0x61);
                        GXParam1u32(s_last_cmode0 & ~0x18);
                    }
                } else {
                    GXParam1u32(val32);

                    if (s_cull_all_active) {
                        s_cull_all_active = false;
                        GXParam1u8(0x61);
                        GXParam1u32(s_last_zmode);
                        GXParam1u8(0x61);
                        GXParam1u32(s_last_cmode0);
                    }
                }
            } else {
                GXParam1u32(val32);
            }
        } else {
            GXParam1u32(val32);
        }
        break;
    }
    case 8: GXCmd1u64(value); break;
    default: break;
    }
    if (display_copy)
        aurora_backend_present();
}

void aurora_backend_set_array(u32 attr, const void* data, u32 size, u8 stride) {
    aurora_backend_set_array_guest(attr, 0u, data, size, stride);
}

void aurora_backend_set_array_guest(u32 attr, u32 guest_address,
                                    const void* data, u32 size, u8 stride) {
    if (!gx_aurora::g_initialized || data == nullptr)
        return;
#if GXRUNTIME_HAS_AURORA_RECOMP
    if (guest_address != 0u)
        gx_aurora::shadow_frontend_set_array(attr, guest_address, stride);
#endif
    GXSetArray(static_cast<GXAttr>(attr), data, size, stride, false);
}

void aurora_backend_load_texture(u8 slot, const void* data, u32 width, u32 height,
                                 u32 format, u32 tlut, bool mipmap, u32 object_id,
                                 u32 data_version) {
    aurora_backend_load_texture_guest(slot, 0u, data, width, height, format,
                                      tlut, mipmap, object_id, data_version);
}

void aurora_backend_load_texture_guest(u8 slot, u32 guest_address,
                                       const void* data, u32 width, u32 height,
                                       u32 format, u32 tlut, bool mipmap,
                                       u32 object_id, u32 data_version) {
    (void)guest_address;
    if (!gx_aurora::g_initialized || data == nullptr || slot >= 8)
        return;
    gx_aurora::g_pending_textures[slot] = {
        .valid = true,
        .data = data,
        .width = width,
        .height = height,
        .format = format,
        .tlut = tlut,
        .mipmap = mipmap,
        .object_id = object_id,
        .data_version = data_version,
    };
}

void aurora_backend_load_tlut(u8 slot, const void* data, u32 format, u16 entries,
                              u32 object_id, u32 data_version) {
    aurora_backend_load_tlut_guest(slot, 0u, data, format, entries, object_id,
                                   data_version);
}

void aurora_backend_load_tlut_guest(u8 slot, u32 guest_address,
                                    const void* data, u32 format, u16 entries,
                                    u32 object_id, u32 data_version) {
    (void)guest_address;
    if (!gx_aurora::g_initialized || data == nullptr || slot >= gx_aurora::g_pending_tluts.size())
        return;
    gx_aurora::g_pending_tluts[slot] = {
        .valid = true,
        .data = data,
        .format = format,
        .entries = entries,
        .object_id = object_id,
        .data_version = data_version,
    };
}

void aurora_backend_set_copy_destination(const void* data) {
    aurora_backend_set_copy_destination_guest(0u, data);
}

void aurora_backend_set_copy_destination_guest(u32 guest_address,
                                               const void* data) {
    (void)guest_address;
    if (!gx_aurora::g_initialized || data == nullptr)
        return;
    gx_aurora::write_aurora_command(GX_AURORA_LOAD_COPY_DEST);
    GXCmd1u64(reinterpret_cast<u64>(data));
}

void aurora_backend_set_guest_address_resolver(
    DolPlatformGuestAddressResolverFn resolve, void* user) {
    gx_aurora::g_guest_address_resolver = resolve;
    gx_aurora::g_guest_address_resolver_user = user;
#if GXRUNTIME_HAS_AURORA_RECOMP
    if (resolve != nullptr) {
        DolGuestAddressResolver frontend_resolver;
        dol_guest_address_resolver_init_callback(
            &frontend_resolver, gx_aurora::frontend_guest_address_resolver_bridge,
            nullptr);
        gx_aurora::g_shadow_frontend.reset(&frontend_resolver);
        gx_aurora::g_shadow_frontend.set_packet_drain_enabled(gx_aurora::g_shadow_frontend_enabled);
        gx_aurora::g_shadow_packet_sink.reset();
        gx_aurora::g_shadow_packet_sink.set_guest_resolver(
            &gx_aurora::g_shadow_frontend.state().resolver);
        if (gx_aurora::g_gx_core_enabled)
            gx_aurora::g_core_sink.set_guest_resolver(&gx_aurora::g_shadow_frontend.state().resolver);
        gx_aurora::g_shadow_frontend_failed = false;
    } else {
        gx_aurora::g_shadow_frontend.reset(nullptr);
        gx_aurora::g_shadow_frontend.set_packet_drain_enabled(gx_aurora::g_shadow_frontend_enabled);
        gx_aurora::g_shadow_packet_sink.reset();
        gx_aurora::g_shadow_packet_sink.set_guest_resolver(nullptr);
    }
    gx_aurora::g_shadow_last_draw_total = 0;
    gx_aurora::g_shadow_last_vertex_total = 0;
    gx_aurora::g_shadow_last_rawvert_total = 0;
    gx_aurora::g_shadow_last_topoidx_total = 0;
    gx_aurora::g_shadow_last_storage_total = 0;
#endif
    if (resolve != nullptr) {
        aurora::gx::recomp::set_guest_address_resolver(
            gx_aurora::aurora_guest_address_resolver_bridge, nullptr);
    } else {
        aurora::gx::recomp::clear_guest_address_resolver();
    }
}

void aurora_backend_configure_vi(u32 tv_mode, u16 fb_width, u16 efb_height,
                                 u16 xfb_height, u16 vi_width, u16 vi_height) {
    if (!gx_aurora::g_initialized)
        return;
    GXRenderModeObj mode{};
    mode.viTVmode = static_cast<VITVMode>(tv_mode);
    mode.fbWidth = fb_width;
    mode.efbHeight = efb_height;
    mode.xfbHeight = xfb_height;
    mode.viWidth = vi_width;
    mode.viHeight = vi_height;
    VIConfigure(&mode);
}

} // extern "C"
