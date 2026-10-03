// SPDX-License-Identifier: GPL-3.0-or-later
// The ubershader: one WGSL module that draws any gxcore shader key, reading
// the key at run time (Dolphin's ubershaders, UberShaderVertex/Pixel). A draw
// whose specialized pipeline is still compiling is drawn with it instead of
// being left out of the frame. Each part below is the specialized generator's
// (gxcore_shader.cpp) with its choices made per draw: the same operations in
// the same order, so it computes what the specialized shader would; where it
// does not (Z textures, early depth), a draw is a frame or two from its own
// pipeline.
#include "gxruntime/gxcore/shader.hpp"

#include <cstddef>
#include <cstdio>
#include <string>

namespace gxruntime::gxcore {
namespace {

void define(std::string& out, const char* name, std::size_t value) {
  char line[96];
  std::snprintf(line, sizeof line, "const %s: u32 = %zuu;\n", name, value);
  out += line;
}

// The key's layout, for kb(): byte offsets of each field.
void emit_key_layout(std::string& out) {
#define K(field) define(out, "K_" #field, offsetof(ShaderKey, field))
  K(num_tex_gens);
  K(has_pos_mtx_idx);
  K(has_tex_mtx_idx);
  K(has_color0);
  K(has_color1);
  K(uv_mask);
  K(textured);
  K(tev_valid);
  K(num_tev_stages);
  K(num_ind_stages);
  K(alpha_comp0);
  K(alpha_comp1);
  K(alpha_logic);
  K(lit_valid);
  K(num_color_chans);
  K(fog_fsel);
  K(fog_proj);
  K(fog_range);
  K(tex_mtx_idx_mask);
  K(chan_captured_mask);
  K(has_vertex_normal);
  K(has_vertex_binormal);
  K(has_vertex_tangent);
  K(dst_alpha);
  K(litchan);
  K(tex_gens);
  K(ind_stages);
  K(tev_stages);
#undef K
#define L(field) define(out, "LC_" #field, offsetof(LightChanKey, field))
  define(out, "LC_SIZE", sizeof(LightChanKey));
  L(enablelighting);
  L(matsource);
  L(ambsource);
  L(diffusefunc);
  L(attnfunc);
  L(light_mask);
#undef L
#define T(field) define(out, "TG_" #field, offsetof(TexGenKey, field))
  define(out, "TG_SIZE", sizeof(TexGenKey));
  T(texgentype);
  T(sourcerow);
  T(inputform);
  T(projection);
  T(embosssourceshift);
  T(embosslightshift);
#undef T
#define I(field) define(out, "IS_" #field, offsetof(IndirectStageKey, field))
  define(out, "IS_SIZE", sizeof(IndirectStageKey));
  I(texmap);
  I(texcoord);
  I(scale_s);
  I(scale_t);
#undef I
#define S(field) define(out, "TS_" #field, offsetof(TevStageKey, field))
  define(out, "TS_SIZE", sizeof(TevStageKey));
  S(cc_a);
  S(cc_b);
  S(cc_c);
  S(cc_d);
  S(cc_bias);
  S(cc_op);
  S(cc_clamp);
  S(cc_scale);
  S(cc_dest);
  S(ac_a);
  S(ac_b);
  S(ac_c);
  S(ac_d);
  S(ac_bias);
  S(ac_op);
  S(ac_clamp);
  S(ac_scale);
  S(ac_dest);
  S(ksel_kc);
  S(ksel_ka);
  S(ras_swap);
  S(tex_swap);
  S(tevorders_texcoord);
  S(tevorders_texmap);
  S(tevorders_colorchan);
  S(tevorders_enable);
  S(ind_stage);
  S(ind_format);
  S(ind_bias);
  S(ind_bump_alpha);
  S(ind_matrix_index);
  S(ind_matrix_id);
  S(ind_wrap_s);
  S(ind_wrap_t);
  S(ind_add_prev);
#undef S
  define(out, "MAX_TEXGENS", kMaxTexGens);
  define(out, "MAX_TEV", kMaxTevStages);
  define(out, "KEY_WORDS", kUberKeyWords);
}

// Everything but the key layout: bindings, helpers, the stages.
constexpr const char* kBody = R"WGSL(
struct Light {
    color: vec4i,
    cosatt: vec4f,
    distatt: vec4f,
    pos: vec4f,
    dir: vec4f,
};
struct VertexShaderConstants {
    posnormalmatrix: array<vec4f, 6>,
    projection: array<vec4f, 4>,
    texmatrices: array<vec4f, 24>,
    transformmatrices: array<vec4f, 64>,
    lights: array<Light, 8>,
    materials: array<vec4i, 4>,
    cached_normal: vec4f,
    cached_tangent: vec4f,
    cached_binormal: vec4f,
    normalmatrices: array<vec4f, 32>,
};
@group(1) @binding(0) var<uniform> vsc: VertexShaderConstants;
// The pixel constants, then the draw's shader key and what the CPU derived
// from it (extra.x: more than one texmap, sampled where each is bound).
struct PixelShaderConstants {
    colors: array<vec4i, 4>,
    kcolors: array<vec4i, 4>,
    alpha_ref: vec4i,
    fogcolor: vec4i,
    fogi: vec4i,
    fogf: vec4f,
    fogrange: array<vec4f, 3>,
    zbias: vec4i,
    texdims: array<vec4i, 8>,
    indtexmtx: array<vec4i, 6>,
    key: array<vec4u, KEY_WORDS>,
    extra: vec4u,
};
@group(2) @binding(0) var<uniform> psc: PixelShaderConstants;
@group(3) @binding(0) var tex0: texture_2d<f32>;
@group(3) @binding(1) var samp0: sampler;
@group(3) @binding(2) var tex1: texture_2d<f32>;
@group(3) @binding(3) var samp1: sampler;
@group(3) @binding(4) var tex2: texture_2d<f32>;
@group(3) @binding(5) var samp2: sampler;
@group(3) @binding(6) var tex3: texture_2d<f32>;
@group(3) @binding(7) var samp3: sampler;
@group(3) @binding(8) var tex4: texture_2d<f32>;
@group(3) @binding(9) var samp4: sampler;
@group(3) @binding(10) var tex5: texture_2d<f32>;
@group(3) @binding(11) var samp5: sampler;
@group(3) @binding(12) var tex6: texture_2d<f32>;
@group(3) @binding(13) var samp6: sampler;
@group(3) @binding(14) var tex7: texture_2d<f32>;
@group(3) @binding(15) var samp7: sampler;

// One byte of the key.
fn kb(off: u32) -> u32 {
    let w = psc.key[off >> 4u][(off >> 2u) & 3u];
    return (w >> ((off & 3u) * 8u)) & 0xFFu;
}

fn texmap_sample(unit: u32, uv: vec2f) -> vec4f {
    switch unit {
        case 0u: { return textureSample(tex0, samp0, uv); }
        case 1u: { return textureSample(tex1, samp1, uv); }
        case 2u: { return textureSample(tex2, samp2, uv); }
        case 3u: { return textureSample(tex3, samp3, uv); }
        case 4u: { return textureSample(tex4, samp4, uv); }
        case 5u: { return textureSample(tex5, samp5, uv); }
        case 6u: { return textureSample(tex6, samp6, uv); }
        default: { return textureSample(tex7, samp7, uv); }
    }
}

fn texmap_dims(unit: u32) -> vec2u {
    switch unit {
        case 0u: { return textureDimensions(tex0); }
        case 1u: { return textureDimensions(tex1); }
        case 2u: { return textureDimensions(tex2); }
        case 3u: { return textureDimensions(tex3); }
        case 4u: { return textureDimensions(tex4); }
        case 5u: { return textureDimensions(tex5); }
        case 6u: { return textureDimensions(tex6); }
        default: { return textureDimensions(tex7); }
    }
}

fn gx_texdims(logical: vec2i, actual: vec2u) -> vec2f {
    if (logical.x > 0 && logical.y > 0) { return vec2f(logical); }
    return vec2f(actual);
}

struct VertexIn {
    @location(0) rawpos: vec3f,
    @location(1) posmtx: u32,
    @location(2) rawcolor0: vec4f,
    @location(3) rawcolor1: vec4f,
    @location(4) rawtex0: vec2f,
    @location(5) rawtex1: vec2f,
    @location(6) rawtex2: vec2f,
    @location(7) rawtex3: vec2f,
    @location(8) rawnormal: vec3f,
    @location(9) texmtxidx: u32,
    @location(10) rawbinormal: vec3f,
    @location(11) rawtangent: vec3f,
    @location(12) rawtex4: vec2f,
    @location(13) texmtxidx_hi: u32,
};
struct VertexOut {
    @builtin(position) pos: vec4f,
    @location(0) color0: vec4f,
    @location(1) color1: vec4f,
    @location(2) uv0: vec3f,
    @location(3) uv1: vec3f,
    @location(4) uv2: vec3f,
    @location(5) uv3: vec3f,
    @location(6) uv4: vec3f,
};

fn rawtex(in: VertexIn, n: u32) -> vec2f {
    switch n {
        case 0u: { return in.rawtex0; }
        case 1u: { return in.rawtex1; }
        case 2u: { return in.rawtex2; }
        case 3u: { return in.rawtex3; }
        default: { return in.rawtex4; }
    }
}

// The channel takes the full lighting path (channel_lit_path).
fn chan_lit(j: u32) -> bool {
    if (((kb(K_chan_captured_mask) >> j) & 1u) == 0u) { return false; }
    let col = K_litchan + j * LC_SIZE;
    let alp = K_litchan + (j + 2u) * LC_SIZE;
    return kb(col + LC_enablelighting) != 0u || kb(alp + LC_enablelighting) != 0u ||
           kb(col + LC_matsource) == 0u || kb(alp + LC_matsource) == 0u;
}

// One light's term (emit_light): its direction and attenuation.
fn light_term(i: u32, ch: u32, pos: vec3f, _normal: vec3f) -> f32 {
    let attnfn = kb(ch + LC_attnfunc);
    let diff = kb(ch + LC_diffusefunc);
    var ldir: vec3f;
    var attn: f32;
    if (attnfn == 0u || attnfn == 2u) {
        ldir = normalize(vsc.lights[i].pos.xyz - pos);
        attn = 1.0;
        if (length(ldir) == 0.0) { ldir = _normal; }
    } else if (attnfn == 1u) {
        ldir = normalize(vsc.lights[i].pos.xyz - pos);
        attn = select(0.0, max(0.0, dot(_normal, vsc.lights[i].dir.xyz)), dot(_normal, ldir) >= 0.0);
        let cosAttn = vsc.lights[i].cosatt.xyz;
        var distAttn: vec3f;
        if (diff == 0u) { distAttn = vsc.lights[i].distatt.xyz; } else { distAttn = normalize(vsc.lights[i].distatt.xyz); }
        attn = max(0.0, dot(cosAttn, vec3f(1.0, attn, attn*attn))) / dot(distAttn, vec3f(1.0, attn, attn*attn));
    } else {
        ldir = vsc.lights[i].pos.xyz - pos;
        let dist2 = dot(ldir, ldir);
        let dist = sqrt(dist2);
        ldir = ldir / dist;
        attn = max(0.0, dot(ldir, vsc.lights[i].dir.xyz));
        attn = max(0.0, vsc.lights[i].cosatt.x + vsc.lights[i].cosatt.y*attn + vsc.lights[i].cosatt.z*attn*attn) /
               dot(vsc.lights[i].distatt.xyz, vec3f(1.0, dist, dist2));
    }
    if (diff == 1u) { return attn * (dot(ldir, _normal)); }
    if (diff == 2u) { return attn * max(0.0, dot(ldir, _normal)); }
    return attn;
}

// calc_lighting_chn{j} (emit_lighting_chan).
fn calc_lighting(j: u32, base_color: vec4f, pos: vec3f, _normal: vec3f) -> vec4f {
    let col = K_litchan + j * LC_SIZE;
    let alp = K_litchan + (j + 2u) * LC_SIZE;
    var lacc: vec4i;
    var mat: vec4i;
    if (kb(col + LC_matsource) == 1u) { mat = vec4i(round(base_color * 255.0)); }
    else { mat = vsc.materials[j + 2u]; }
    if (kb(col + LC_enablelighting) != 0u) {
        if (kb(col + LC_ambsource) == 1u) { lacc = vec4i(round(base_color * 255.0)); }
        else { lacc = vsc.materials[j]; }
    } else {
        lacc = vec4i(255, 255, 255, 255);
    }
    if (kb(alp + LC_matsource) != kb(col + LC_matsource)) {
        if (kb(alp + LC_matsource) == 1u) { mat.w = i32(round(base_color.w * 255.0)); }
        else { mat.w = vsc.materials[j + 2u].w; }
    }
    if (kb(alp + LC_enablelighting) != 0u) {
        if (kb(alp + LC_ambsource) == 1u) { lacc.w = i32(round(base_color.w * 255.0)); }
        else { lacc.w = vsc.materials[j].w; }
    } else {
        lacc.w = 255;
    }
    if (kb(col + LC_enablelighting) != 0u) {
        let mask = kb(col + LC_light_mask);
        for (var i = 0u; i < 8u; i++) {
            if (((mask >> i) & 1u) != 0u) {
                let t = light_term(i, col, pos, _normal);
                lacc = lacc + vec4i(vec3i(round(t * vec3f(vsc.lights[i].color.rgb))), 0);
            }
        }
    }
    if (kb(alp + LC_enablelighting) != 0u) {
        let mask = kb(alp + LC_light_mask);
        for (var i = 0u; i < 8u; i++) {
            if (((mask >> i) & 1u) != 0u) {
                let t = light_term(i, alp, pos, _normal);
                lacc.a = lacc.a + i32(round(t * f32(vsc.lights[i].color.a)));
            }
        }
    }
    lacc = clamp(lacc, vec4<i32>(0), vec4<i32>(255));
    return vec4f((mat * (lacc + (lacc >> vec4u(7)))) >> vec4u(8)) / 255.0;
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var o: VertexOut;
    let has_posidx = kb(K_has_pos_mtx_idx) != 0u;
    let lit = kb(K_lit_valid) != 0u;
    let ntg = min(kb(K_num_tex_gens), MAX_TEXGENS);
    var has_emboss = false;
    for (var i = 0u; i < ntg; i++) {
        if (kb(K_tex_gens + i * TG_SIZE + TG_texgentype) == 1u) { has_emboss = true; }
    }
    let needs_normal_bank = has_posidx && (lit || has_emboss);
    var posidx = 0;
    var normidx = 0;
    var p0: vec4f;
    var p1: vec4f;
    var p2: vec4f;
    if (has_posidx) {
        posidx = i32(in.posmtx);
        p0 = vsc.transformmatrices[posidx];
        p1 = vsc.transformmatrices[posidx + 1];
        p2 = vsc.transformmatrices[posidx + 2];
        normidx = posidx & 31;
    } else {
        p0 = vsc.posnormalmatrix[0];
        p1 = vsc.posnormalmatrix[1];
        p2 = vsc.posnormalmatrix[2];
    }
    let pos4 = vec4f(in.rawpos, 1.0);
    let viewpos = vec4f(dot(p0, pos4), dot(p1, pos4), dot(p2, pos4), 1.0);
    var clip = vec4f(dot(vsc.projection[0], viewpos), dot(vsc.projection[1], viewpos),
                     dot(vsc.projection[2], viewpos), dot(vsc.projection[3], viewpos));
    clip.z = -clip.z;
    o.pos = clip;

    let hc0 = kb(K_has_color0) != 0u;
    let hc1 = kb(K_has_color1) != 0u;
    var vc0 = vec4f(1.0);
    if (hc0) { vc0 = in.rawcolor0; } else if (hc1) { vc0 = in.rawcolor1; }
    var vc1 = vec4f(1.0);
    if (hc0 && hc1) { vc1 = in.rawcolor1; }
    let has_normal = kb(K_has_vertex_normal) != 0u;
    var normal_in = vsc.cached_normal.xyz;
    if (has_normal) { normal_in = in.rawnormal; }
    var tangent_in = vsc.cached_tangent.xyz;
    if (kb(K_has_vertex_tangent) != 0u) { tangent_in = in.rawtangent; }
    var binormal_in = vsc.cached_binormal.xyz;
    if (kb(K_has_vertex_binormal) != 0u) { binormal_in = in.rawbinormal; }
    let lit0 = chan_lit(0u);
    let lit1 = chan_lit(1u);
    var _normal = vec3f(0.0, 0.0, 1.0);
    if ((lit0 || lit1) && lit) {
        if (needs_normal_bank) {
            _normal = normalize(vec3f(dot(vsc.normalmatrices[normidx].xyz, normal_in),
                                      dot(vsc.normalmatrices[normidx + 1].xyz, normal_in),
                                      dot(vsc.normalmatrices[normidx + 2].xyz, normal_in)));
        } else {
            _normal = normalize(vec3f(dot(vsc.posnormalmatrix[3].xyz, normal_in),
                                      dot(vsc.posnormalmatrix[4].xyz, normal_in),
                                      dot(vsc.posnormalmatrix[5].xyz, normal_in)));
        }
    }
    if (lit0) { o.color0 = calc_lighting(0u, vc0, viewpos.xyz, _normal); } else { o.color0 = vc0; }
    if (lit1) { o.color1 = calc_lighting(1u, vc1, viewpos.xyz, _normal); } else { o.color1 = vc1; }
    let nchans = kb(K_num_color_chans);
    if (nchans == 0u) { o.color0 = vec4f(0.0); }
    if (nchans <= 1u) { o.color1 = vec4f(0.0); }

    var uvs: array<vec3f, 5>;
    let uv_mask = kb(K_uv_mask);
    let has_tmi = kb(K_has_tex_mtx_idx) != 0u;
    let tmi_mask = kb(K_tex_mtx_idx_mask);
    for (var i = 0u; i < ntg; i++) {
        let tg = K_tex_gens + i * TG_SIZE;
        var coord = vec4f(0.0, 0.0, 1.0, 1.0);
        let row = kb(tg + TG_sourcerow);
        if (row == 0u) {
            coord = vec4f(in.rawpos, 1.0);
        } else if (row == 1u) {
            if (has_normal) { coord = vec4f(in.rawnormal, 1.0); }
        } else if (row >= 5u && row < 5u + MAX_TEXGENS) {
            let texnum = row - 5u;
            if (((uv_mask >> texnum) & 1u) != 0u) {
                let r = rawtex(in, texnum);
                coord = vec4f(r.x, r.y, 1.0, 1.0);
            }
        }
        if (kb(tg + TG_inputform) == 0u) { coord.z = 1.0; }
        let ty = kb(tg + TG_texgentype);
        if (ty == 0u) {
            let proj = kb(tg + TG_projection) != 0u;
            var uv: vec3f;
            if (has_tmi && ((tmi_mask >> i) & 1u) != 0u) {
                var ti: u32;
                if (i < 4u) { ti = (in.texmtxidx >> (8u * i)) & 0xFFu; }
                else { ti = (in.texmtxidx_hi >> (8u * (i - 4u))) & 0xFFu; }
                let m0 = vsc.transformmatrices[ti];
                let m1 = vsc.transformmatrices[ti + 1u];
                let m2 = vsc.transformmatrices[ti + 2u];
                if (proj) { uv = vec3f(dot(coord, m0), dot(coord, m1), dot(coord, m2)); }
                else { uv = vec3f(dot(coord, m0), dot(coord, m1), 1.0); }
            } else if (proj) {
                uv = vec3f(dot(coord, vsc.texmatrices[3u * i]), dot(coord, vsc.texmatrices[3u * i + 1u]),
                           dot(coord, vsc.texmatrices[3u * i + 2u]));
            } else {
                uv = vec3f(dot(coord, vsc.texmatrices[3u * i]), dot(coord, vsc.texmatrices[3u * i + 1u]), 1.0);
            }
            if (uv.z == 0.0) { uv = vec3f(clamp(uv.xy * 0.5, vec2f(-1.0), vec2f(1.0)), uv.z); }
            uvs[i] = uv;
        } else if (ty == 2u) {
            uvs[i] = vec3f(o.color0.x, o.color0.y, 1.0);
        } else if (ty == 3u) {
            uvs[i] = vec3f(o.color1.x, o.color1.y, 1.0);
        } else {
            var tn: vec3f;
            var bn: vec3f;
            if (needs_normal_bank) {
                tn = vec3f(dot(vsc.normalmatrices[normidx].xyz, tangent_in),
                           dot(vsc.normalmatrices[normidx + 1].xyz, tangent_in),
                           dot(vsc.normalmatrices[normidx + 2].xyz, tangent_in));
                bn = vec3f(dot(vsc.normalmatrices[normidx].xyz, binormal_in),
                           dot(vsc.normalmatrices[normidx + 1].xyz, binormal_in),
                           dot(vsc.normalmatrices[normidx + 2].xyz, binormal_in));
            } else {
                tn = vec3f(dot(vsc.posnormalmatrix[3].xyz, tangent_in), dot(vsc.posnormalmatrix[4].xyz, tangent_in),
                           dot(vsc.posnormalmatrix[5].xyz, tangent_in));
                bn = vec3f(dot(vsc.posnormalmatrix[3].xyz, binormal_in), dot(vsc.posnormalmatrix[4].xyz, binormal_in),
                           dot(vsc.posnormalmatrix[5].xyz, binormal_in));
            }
            let ld = normalize(vsc.lights[kb(tg + TG_embosslightshift)].pos.xyz - viewpos.xyz);
            uvs[i] = uvs[min(kb(tg + TG_embosssourceshift), 4u)] + vec3f(dot(ld, tn), dot(ld, bn), 0.0);
        }
    }
    o.uv0 = uvs[0];
    o.uv1 = uvs[1];
    o.uv2 = uvs[2];
    o.uv3 = uvs[3];
    o.uv4 = uvs[4];
    return o;
}

fn uv_of(in: VertexOut, n: u32) -> vec3f {
    switch n {
        case 0u: { return in.uv0; }
        case 1u: { return in.uv1; }
        case 2u: { return in.uv2; }
        case 3u: { return in.uv3; }
        default: { return in.uv4; }
    }
}

fn swz(v: vec4i, off: u32) -> vec4i {
    return vec4i(v[kb(off) & 3u], v[kb(off + 1u) & 3u], v[kb(off + 2u) & 3u], v[kb(off + 3u) & 3u]);
}

fn konst_frac(sel: u32) -> i32 {
    var f = array<i32, 8>(255, 223, 191, 159, 128, 96, 64, 32);
    return f[sel & 7u];
}

fn konst_c(sel: u32) -> vec3i {
    if (sel < 8u) { return vec3i(konst_frac(sel)); }
    if (sel < 12u) { return vec3i(0); }
    let k = psc.kcolors[sel & 3u];
    if (sel < 16u) { return k.rgb; }
    return vec3i(k[((sel - 16u) >> 2u) & 3u]);
}

fn konst_a(sel: u32) -> i32 {
    if (sel < 8u) { return konst_frac(sel); }
    if (sel < 16u) { return 0; }
    return psc.kcolors[sel & 3u][((sel - 16u) >> 2u) & 3u];
}

// The TEV registers, prev c0 c1 c2, and a stage's inputs (kTevCInput, kTevAInput).
fn cin(arg: u32, r: array<vec4i, 4>, tex: vec4i, ras: vec4i, konst: vec4i) -> vec3i {
    switch arg {
        case 0u: { return r[0].rgb; }
        case 1u: { return vec3i(r[0].a); }
        case 2u: { return r[1].rgb; }
        case 3u: { return vec3i(r[1].a); }
        case 4u: { return r[2].rgb; }
        case 5u: { return vec3i(r[2].a); }
        case 6u: { return r[3].rgb; }
        case 7u: { return vec3i(r[3].a); }
        case 8u: { return tex.rgb; }
        case 9u: { return vec3i(tex.a); }
        case 10u: { return ras.rgb; }
        case 11u: { return vec3i(ras.a); }
        case 12u: { return vec3i(255, 255, 255); }
        case 13u: { return vec3i(128, 128, 128); }
        case 14u: { return konst.rgb; }
        default: { return vec3i(0, 0, 0); }
    }
}

fn ain(arg: u32, r: array<vec4i, 4>, tex: vec4i, ras: vec4i, konst: vec4i) -> i32 {
    switch arg {
        case 0u: { return r[0].a; }
        case 1u: { return r[1].a; }
        case 2u: { return r[2].a; }
        case 3u: { return r[3].a; }
        case 4u: { return tex.a; }
        case 5u: { return ras.a; }
        case 6u: { return konst.a; }
        default: { return 0; }
    }
}

// tev_regular, for rgb and for alpha.
fn tev_regular3(a: vec3i, b: vec3i, c: vec3i, d: vec3i, bias: u32, op: u32, scale: u32) -> vec3i {
    let div2 = scale == 3u;
    var sh = 0u;
    if (scale == 1u) { sh = 1u; } else if (scale == 2u) { sh = 2u; }
    var lerp = (a << vec3u(8u)) + (b - a) * (c + (c >> vec3u(7u)));
    if (!div2) {
        if (sh != 0u) { lerp = lerp << vec3u(sh); }
        lerp = lerp + vec3i(select(128, 127, op == 1u));
    }
    lerp = lerp >> vec3u(8u);
    var dterm = d;
    if (bias == 1u) { dterm = dterm + vec3i(128); } else if (bias == 2u) { dterm = dterm - vec3i(128); }
    if (!div2 && sh != 0u) { dterm = dterm << vec3u(sh); }
    var res = dterm + lerp;
    if (op == 1u) { res = dterm - lerp; }
    if (div2) { res = res >> vec3u(1u); }
    return res;
}

fn tev_regular1(a: i32, b: i32, c: i32, d: i32, bias: u32, op: u32, scale: u32) -> i32 {
    let div2 = scale == 3u;
    var sh = 0u;
    if (scale == 1u) { sh = 1u; } else if (scale == 2u) { sh = 2u; }
    var lerp = (a << 8u) + (b - a) * (c + (c >> 7u));
    if (!div2) {
        if (sh != 0u) { lerp = lerp << sh; }
        lerp = lerp + select(128, 127, op == 1u);
    }
    lerp = lerp >> 8u;
    var dterm = d;
    if (bias == 1u) { dterm = dterm + 128; } else if (bias == 2u) { dterm = dterm - 128; }
    if (!div2 && sh != 0u) { dterm = dterm << sh; }
    var res = dterm + lerp;
    if (op == 1u) { res = dterm - lerp; }
    if (div2) { res = res >> 1u; }
    return res;
}

// tev_compare's condition for the R8, GR16 and BGR24 modes.
fn tev_cond(a: vec4i, b: vec4i, comparison: u32, mode: u32) -> bool {
    var va = a.r;
    var vb = b.r;
    if (mode == 1u) { va = a.r + a.g * 256; vb = b.r + b.g * 256; }
    else if (mode == 2u) { va = a.r + a.g * 256 + a.b * 65536; vb = b.r + b.g * 256 + b.b * 65536; }
    if (comparison == 1u) { return va == vb; }
    return va > vb;
}

fn tev_compare3(a: vec4i, b: vec4i, c: vec4i, d: vec4i, comparison: u32, mode: u32) -> vec3i {
    if (mode == 3u) {
        if (comparison == 1u) {
            return d.rgb + (vec3i(1, 1, 1) - sign(abs(a.rgb - b.rgb))) * c.rgb;
        }
        return d.rgb + max(sign(a.rgb - b.rgb), vec3i(0, 0, 0)) * c.rgb;
    }
    return d.rgb + select(vec3i(0, 0, 0), c.rgb, tev_cond(a, b, comparison, mode));
}

fn tev_compare1(a: vec4i, b: vec4i, c: vec4i, d: vec4i, comparison: u32, mode: u32) -> i32 {
    var cond: bool;
    if (mode == 3u) {
        if (comparison == 1u) { cond = a.a == b.a; } else { cond = a.a > b.a; }
    } else {
        cond = tev_cond(a, b, comparison, mode);
    }
    return d.a + select(0, c.a, cond);
}

fn alpha_cond(comp: u32, a: i32, r: i32) -> bool {
    switch comp {
        case 0u: { return false; }
        case 1u: { return a < r; }
        case 2u: { return a == r; }
        case 3u: { return a <= r; }
        case 4u: { return a > r; }
        case 5u: { return a != r; }
        case 6u: { return a >= r; }
        default: { return true; }
    }
}

fn ind_wrap(coord: i32, wrap: u32) -> i32 {
    if (wrap == 0u) { return coord; }
    if (wrap >= 6u) { return 0; }
    let period = (256u >> (wrap - 1u)) << 7u;
    return coord & i32(period - 1u);
}

// The TEV (emit_tev_fragment): prev after the last stage, before the alpha test.
fn tev(in: VertexOut) -> vec4i {
    let ntg = min(kb(K_num_tex_gens), MAX_TEXGENS);
    let nind = kb(K_num_ind_stages);
    let indirect_enabled = nind != 0u;
    let textured = kb(K_textured) != 0u;
    let multi = psc.extra.x != 0u;
    var r = array<vec4i, 4>(psc.colors[0], psc.colors[1], psc.colors[2], psc.colors[3]);
    var rastemp = vec4i(0, 0, 0, 0);
    var rawtextemp = vec4i(0, 0, 0, 0);
    var textemp = vec4i(0, 0, 0, 0);
    var konsttemp = vec4i(0, 0, 0, 0);
    var alphabump = 0;
    var tevcoord = vec2i(0, 0);
    let col0i = vec4i(round(in.color0 * 255.0));
    let col1i = vec4i(round(in.color1 * 255.0));
    var fixpoint: array<vec2i, 5>;
    for (var i = 0u; i < ntg; i++) {
        let uv = uv_of(in, i);
        if (kb(K_tex_gens + i * TG_SIZE + TG_projection) != 0u) {
            fixpoint[i] = vec2i((uv.xy / max(uv.z, 1e-6)) * vec2f(psc.texdims[i].zw * 128));
        } else {
            fixpoint[i] = vec2i(uv.xy * vec2f(psc.texdims[i].zw * 128));
        }
    }
    let nstages = min(kb(K_num_tev_stages), MAX_TEV);
    var last_cc_dest = 0u;
    var last_ac_dest = 0u;
    for (var n = 0u; n < nstages; n++) {
        let s = K_tev_stages + n * TS_SIZE;
        let cc_a = kb(s + TS_cc_a);
        let cc_b = kb(s + TS_cc_b);
        let cc_c = kb(s + TS_cc_c);
        let cc_d = kb(s + TS_cc_d);
        let ac_a = kb(s + TS_ac_a);
        let ac_b = kb(s + TS_ac_b);
        let ac_c = kb(s + TS_ac_c);
        let ac_d = kb(s + TS_ac_d);
        let colorchan = kb(s + TS_tevorders_colorchan);
        let uses_ras = cc_a == 10u || cc_b == 10u || cc_c == 10u || cc_d == 10u ||
                       cc_a == 11u || cc_b == 11u || cc_c == 11u || cc_d == 11u ||
                       ac_a == 5u || ac_b == 5u || ac_c == 5u || ac_d == 5u;
        let uses_bump_ras = indirect_enabled && uses_ras && (colorchan == 5u || colorchan == 6u);
        if (uses_ras && !uses_bump_ras) {
            var src = vec4i(0, 0, 0, 0);
            if (colorchan == 0u) { src = col0i; } else if (colorchan == 1u) { src = col1i; }
            rastemp = swz(src, s + TS_ras_swap);
        }
        var texcoord = kb(s + TS_tevorders_texcoord);
        if (texcoord >= ntg) { texcoord = 0u; }
        let texunit = select(0u, kb(s + TS_tevorders_texmap) & 7u, multi);
        let direct_sample = kb(s + TS_tevorders_enable) != 0u && ntg > 0u && textured;
        var stage_dims = vec2i(0, 0);
        var base_coord = vec2i(0, 0);
        var stage_uv = vec2f(0.0, 0.0);
        if (direct_sample && indirect_enabled) {
            stage_dims = vec2i(gx_texdims(psc.texdims[texunit].xy, texmap_dims(texunit)));
            base_coord = fixpoint[texcoord];
            stage_uv = vec2f(base_coord) / (vec2f(stage_dims) * 128.0);
        }
        let ind_stage = kb(s + TS_ind_stage);
        let ind_matrix_index = kb(s + TS_ind_matrix_index);
        let ind_bump_alpha = kb(s + TS_ind_bump_alpha);
        let ind_format = kb(s + TS_ind_format) & 3u;
        let sample_indirect = ind_stage < nind && (ind_matrix_index != 0u || ind_bump_alpha != 0u);
        var ind_trans = vec2i(0, 0);
        if (sample_indirect) {
            let ind = K_ind_stages + ind_stage * IS_SIZE;
            var indcoord = kb(ind + IS_texcoord);
            if (indcoord >= ntg) { indcoord = 0u; }
            let indunit = select(0u, kb(ind + IS_texmap) & 7u, multi);
            let ind_coord_scaled = fixpoint[indcoord] >> vec2u(kb(ind + IS_scale_s), kb(ind + IS_scale_t));
            let ind_uv = vec2f(ind_coord_scaled) / (gx_texdims(psc.texdims[indunit].xy, texmap_dims(indunit)) * 128.0);
            let ind_raw = vec3i(round(texmap_sample(indunit, ind_uv).abg * 255.0));
            if (ind_bump_alpha != 0u) {
                var alpha_shift = array<u32, 4>(0u, 5u, 4u, 3u);
                alphabump = (ind_raw[(ind_bump_alpha - 1u) & 3u] << alpha_shift[ind_format]) & 248;
            }
            if (ind_matrix_index != 0u && direct_sample) {
                var format_shift = array<u32, 4>(0u, 3u, 4u, 5u);
                var ind_coord = ind_raw >> vec3u(format_shift[ind_format]);
                let bias = select(1, -128, ind_format == 0u);
                let ind_bias = kb(s + TS_ind_bias);
                if ((ind_bias & 1u) != 0u) { ind_coord.x += bias; }
                if ((ind_bias & 2u) != 0u) { ind_coord.y += bias; }
                if ((ind_bias & 4u) != 0u) { ind_coord.z += bias; }
                let matrix = min(2u * (ind_matrix_index - 1u), 4u);
                let id = kb(s + TS_ind_matrix_id);
                if (id == 0u) {
                    let m0 = psc.indtexmtx[matrix];
                    let m1 = psc.indtexmtx[matrix + 1u];
                    ind_trans = vec2i(m0.x * ind_coord.x + m0.y * ind_coord.y + m0.z * ind_coord.z,
                                      m1.x * ind_coord.x + m1.y * ind_coord.y + m1.z * ind_coord.z) >> vec2u(3u);
                } else if (id == 1u) {
                    ind_trans = (base_coord * vec2i(ind_coord.x)) >> vec2u(8u);
                } else if (id == 2u) {
                    ind_trans = (base_coord * vec2i(ind_coord.y)) >> vec2u(8u);
                } else {
                    ind_trans = vec2i(0, 0);
                }
                if (psc.indtexmtx[matrix].w >= 0) { ind_trans = ind_trans >> vec2u(u32(psc.indtexmtx[matrix].w)); }
                else { ind_trans = ind_trans << vec2u(u32(-psc.indtexmtx[matrix].w)); }
            }
        }
        if (direct_sample && indirect_enabled) {
            let wrapped = vec2i(ind_wrap(base_coord.x, kb(s + TS_ind_wrap_s)), ind_wrap(base_coord.y, kb(s + TS_ind_wrap_t)));
            var translation = vec2i(0, 0);
            if (sample_indirect && ind_matrix_index != 0u) { translation = ind_trans; }
            if (kb(s + TS_ind_add_prev) != 0u) { tevcoord += wrapped + translation; }
            else { tevcoord = wrapped + translation; }
            tevcoord = (tevcoord << vec2u(8u)) >> vec2u(8u);
            stage_uv = vec2f(tevcoord) / (vec2f(stage_dims) * 128.0);
        }
        if (direct_sample) {
            if (indirect_enabled) {
                rawtextemp = vec4i(round(texmap_sample(texunit, stage_uv) * 255.0));
            } else {
                let uv = vec2f(fixpoint[texcoord]) / (gx_texdims(psc.texdims[texunit].xy, texmap_dims(texunit)) * 128.0);
                rawtextemp = vec4i(round(texmap_sample(texunit, uv) * 255.0));
            }
            textemp = swz(rawtextemp, s + TS_tex_swap);
        } else if (ntg == 0u) {
            textemp = vec4i(0, 0, 0, 0);
        } else {
            textemp = vec4i(255, 255, 255, 255);
        }
        if (uses_bump_ras) {
            var bump = alphabump;
            if (colorchan != 5u) { bump = alphabump | (alphabump >> 5u); }
            rastemp = swz(vec4i(bump, bump, bump, bump), s + TS_ras_swap);
        }
        if (cc_a == 14u || cc_b == 14u || cc_c == 14u || cc_d == 14u ||
            ac_a == 6u || ac_b == 6u || ac_c == 6u || ac_d == 6u) {
            konsttemp = vec4i(konst_c(kb(s + TS_ksel_kc)), konst_a(kb(s + TS_ksel_ka)));
        }
        let tevin_a = vec4i(cin(cc_a, r, textemp, rastemp, konsttemp), ain(ac_a, r, textemp, rastemp, konsttemp)) & vec4i(255, 255, 255, 255);
        let tevin_b = vec4i(cin(cc_b, r, textemp, rastemp, konsttemp), ain(ac_b, r, textemp, rastemp, konsttemp)) & vec4i(255, 255, 255, 255);
        let tevin_c = vec4i(cin(cc_c, r, textemp, rastemp, konsttemp), ain(ac_c, r, textemp, rastemp, konsttemp)) & vec4i(255, 255, 255, 255);
        let tevin_d = vec4i(cin(cc_d, r, textemp, rastemp, konsttemp), ain(ac_d, r, textemp, rastemp, konsttemp));
        // Color combine.
        var crgb: vec3i;
        if (kb(s + TS_cc_bias) == 3u) {
            crgb = tev_compare3(tevin_a, tevin_b, tevin_c, tevin_d, kb(s + TS_cc_op), kb(s + TS_cc_scale));
        } else {
            crgb = tev_regular3(tevin_a.rgb, tevin_b.rgb, tevin_c.rgb, tevin_d.rgb, kb(s + TS_cc_bias),
                                kb(s + TS_cc_op), kb(s + TS_cc_scale));
        }
        let cdest = kb(s + TS_cc_dest) & 3u;
        if (kb(s + TS_cc_clamp) != 0u) { crgb = clamp(crgb, vec3<i32>(0), vec3<i32>(255)); }
        else { crgb = clamp(crgb, vec3<i32>(-1024), vec3<i32>(1023)); }
        r[cdest] = vec4i(crgb, r[cdest].a);
        // Alpha combine.
        var aexp: i32;
        if (kb(s + TS_ac_bias) == 3u) {
            aexp = tev_compare1(tevin_a, tevin_b, tevin_c, tevin_d, kb(s + TS_ac_op), kb(s + TS_ac_scale));
        } else {
            aexp = tev_regular1(tevin_a.a, tevin_b.a, tevin_c.a, tevin_d.a, kb(s + TS_ac_bias), kb(s + TS_ac_op),
                                kb(s + TS_ac_scale));
        }
        let adest = kb(s + TS_ac_dest) & 3u;
        if (kb(s + TS_ac_clamp) != 0u) { aexp = clamp(aexp, i32(0), i32(255)); }
        else { aexp = clamp(aexp, i32(-1024), i32(1023)); }
        r[adest] = vec4i(r[adest].rgb, aexp);
        last_cc_dest = cdest;
        last_ac_dest = adest;
    }
    var prev = r[0];
    if (last_cc_dest != 0u) { prev = vec4i(r[last_cc_dest].rgb, prev.a); }
    if (last_ac_dest != 0u) { prev = vec4i(prev.rgb, r[last_ac_dest].a); }
    return prev;
}

// emit_fog, on prev.rgb.
fn fog(prev: vec4i, in: VertexOut) -> vec4i {
    let fsel = kb(K_fog_fsel);
    if (fsel == 0u) { return prev; }
    var zCoord = i32((1.0 - in.pos.z) * 16777216.0);
    zCoord = clamp(zCoord, i32(0), i32(16777215));
    var ze: f32;
    if (kb(K_fog_proj) == 0u) {
        ze = (psc.fogf.x * 16777216.0) / f32(psc.fogi.y - (zCoord >> u32(psc.fogi.w)));
    } else {
        ze = psc.fogf.x * f32(zCoord) / 16777216.0;
    }
    if (kb(K_fog_range) != 0u) {
        let offset = (2.0 * (in.pos.x / psc.fogf.w)) - 1.0 - psc.fogf.z;
        let floatindex = clamp(9.0 - abs(offset) * 9.0, 0.0, 9.0);
        let indexlower = u32(floatindex);
        let indexupper = indexlower + 1u;
        let klower = psc.fogrange[indexlower >> 2u][indexlower & 3u];
        let kupper = psc.fogrange[indexupper >> 2u][indexupper & 3u];
        let k = mix(klower, kupper, fract(floatindex));
        let x_adjust = sqrt(offset * offset + k * k) / k;
        ze = ze * x_adjust;
    }
    var fogv = clamp(ze - psc.fogf.y, 0.0, 1.0);
    if (fsel == 4u) { fogv = 1.0 - exp2(-8.0 * fogv); }
    else if (fsel == 5u) { fogv = 1.0 - exp2(-8.0 * fogv * fogv); }
    else if (fsel == 6u) { fogv = exp2(-8.0 * (1.0 - fogv)); }
    else if (fsel == 7u) { fogv = 1.0 - fogv; fogv = exp2(-8.0 * fogv * fogv); }
    let ifog = i32(round(fogv * 256.0));
    return vec4i((prev.rgb * (256 - ifog) + psc.fogcolor.rgb * ifog) >> vec3u(8u), prev.a);
}

// The fragment's colour, in 0..1: the TEV, the alpha test and fog; or the
// passthrough of a key without TEV (vertex colour, or the first texture).
fn shade(in: VertexOut) -> vec4f {
    if (kb(K_tev_valid) == 0u) {
        var prev = in.color0;
        if (kb(K_textured) != 0u) {
            var uv = in.uv0.xy;
            if (kb(K_num_tex_gens) > 0u && kb(K_tex_gens + TG_projection) != 0u) { uv = in.uv0.xy / max(in.uv0.z, 1e-6); }
            prev = textureSample(tex0, samp0, uv);
        }
        return prev;
    }
    var prev = tev(in);
    let ok0 = alpha_cond(kb(K_alpha_comp0), prev.a, psc.alpha_ref.x);
    let ok1 = alpha_cond(kb(K_alpha_comp1), prev.a, psc.alpha_ref.y);
    let logic = kb(K_alpha_logic);
    var passed: bool;
    if (logic == 0u) { passed = ok0 && ok1; }
    else if (logic == 1u) { passed = ok0 || ok1; }
    else if (logic == 2u) { passed = ok0 != ok1; }
    else { passed = ok0 == ok1; }
    if (!passed) { discard; }
    prev = fog(prev, in);
    return vec4f(prev) / 255.0;
}
)WGSL";

constexpr const char* kPlainOut = R"WGSL(
@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4f {
    return shade(in);
}
)WGSL";

// Destination alpha (dual-source blending): the constant alpha written
// instead, and the colour's alpha as the second source.
constexpr const char* kDualOut = R"WGSL(
struct FragmentOut {
    @location(0) @blend_src(0) color: vec4f,
    @location(0) @blend_src(1) blend: vec4f,
};
@fragment
fn fs_main(in: VertexOut) -> FragmentOut {
    let c = shade(in);
    return FragmentOut(vec4f(c.rgb, f32(kb(K_dst_alpha)) / 255.0), vec4f(0.0, 0.0, 0.0, c.a));
}
)WGSL";

} // namespace

std::string generate_uber_wgsl(bool dual_source) {
  std::string out;
  out.reserve(32768);
  if (dual_source)
    out += "enable dual_source_blending;\n\n";
  out += "// gxcore ubershader (any key, read from the pixel uniform)\n";
  emit_key_layout(out);
  out += kBody;
  out += dual_source ? kDualOut : kPlainOut;
  return out;
}

} // namespace gxruntime::gxcore
