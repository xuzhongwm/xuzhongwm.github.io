/* ==========================================================================
   Background scene — a model you scroll through
   Raw WebGL, no library. A transformer stack (token row, layer sheets,
   residual rails, attention arcs, a mixture-of-experts router, an output
   distribution) with forward (cool) and backward (warm) passes running
   through it. Around it: a retrieval memory feeding the context (in-context
   knowledge editing), a loss landscape with three descent paths and an
   unlearning ascent, the accelerator die underneath, and an embedding cloud.
   Native scroll position picks a point between per-section viewpoints and
   the camera is damped toward it. No scroll hijacking.
   ========================================================================== */

(function () {
  "use strict";

  const canvas = document.getElementById("scene");
  if (!canvas || !canvas.getContext) return;

  const GL_OPTS = {
    alpha: true,
    antialias: true,
    premultipliedAlpha: true,
    depth: false,
    stencil: false,
    powerPreference: "low-power",
  };
  let gl = null;
  try {
    gl = canvas.getContext("webgl", GL_OPTS) || canvas.getContext("experimental-webgl", GL_OPTS);
  } catch (e) {
    gl = null;
  }
  // No WebGL: the static dotted hero texture stays as the fallback.
  if (!gl) return;

  const root = document.documentElement;
  const reduceMQ = window.matchMedia("(prefers-reduced-motion: reduce)");
  const fineMQ = window.matchMedia("(hover: hover) and (pointer: fine)");
  const darkMQ = window.matchMedia("(prefers-color-scheme: dark)");

  root.classList.add("has-scene");

  /* Helpers ------------------------------------------------------------ */

  const TAU = Math.PI * 2;
  const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
  const lerp = (a, b, t) => a + (b - a) * t;

  function smooth(e0, e1, x) {
    const t = clamp((x - e0) / (e1 - e0), 0, 1);
    return t * t * (3 - 2 * t);
  }

  function easeInOut(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }

  // Seeded so the scene has the same shape on every visit.
  let seed = 3193;
  function rand() {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  function gauss() {
    return (rand() + rand() + rand() + rand() - 2) / 2;
  }

  function bezier(p0, p1, p2, p3, s) {
    const u = 1 - s;
    const a = u * u * u;
    const b = 3 * u * u * s;
    const c = 3 * u * s * s;
    const d = s * s * s;
    return [
      a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
      a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
      a * p0[2] + b * p1[2] + c * p2[2] + d * p3[2],
    ];
  }

  const dist2 = (p, q) => (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2;

  /* Geometry -----------------------------------------------------------
     Every vertex: x y z | colour index | w ph group alpha | rate offset size.
     w is a position along whatever moves through that element (layer
     height for the forward/backward passes, distance along a path for
     pulses); the vertex shader does all the animation. */

  const compact = Math.min(window.innerWidth, window.innerHeight) < 620;
  const STRIDE = 11;

  const G = {
    NEURON: 0, FORGET: 1, ATTN: 2, MOE: 3, OUT: 4, LAND: 5, DESCENT: 6,
    ASCENT: 7, MEMORY: 8, BEAM: 9, CHIP: 10, CLOUD: 11, CLOUD_EDGE: 12,
  };
  const GROUPS = 13;
  const C = {
    NEURON: 0, RAIL: 1, ATTN: 2, MOE: 3, OUT: 4, LAND: 5, ACC: 6, COST: 7,
    FORMAT: 8, ASCENT: 9, MEMORY: 10, BEAM: 11, CHIP: 12, EDGE: 13, CLOUD: 14,
  };
  const COLORS = 18;

  const P = [];
  const L = [];

  function point(p, ci, g, a, size, w, rate, off) {
    P.push(p[0], p[1], p[2], ci, w == null ? -1 : w, rand(), g, a, rate || 0, off || 0, size);
  }

  function line(p, q, ci, g, a, w0, w1, rate, off) {
    const ph = rand();
    const ws = w0 == null ? -1 : w0;
    const we = w1 == null ? ws : w1;
    L.push(p[0], p[1], p[2], ci, ws, ph, g, a, rate || 0, off || 0, 0);
    L.push(q[0], q[1], q[2], ci, we, ph, g, a, rate || 0, off || 0, 0);
  }

  function path(points, ci, g, a, size, rate, off, pathAlpha) {
    for (let k = 0; k < points.length; k += 1) {
      point(points[k], ci, g, a, size, k / (points.length - 1), rate, off);
      if (k > 0 && pathAlpha) line(points[k - 1], points[k], ci, g, pathAlpha);
    }
  }

  function box(c, s, ci, g, a) {
    const v = [];
    for (let i = 0; i < 8; i += 1) {
      v.push([c[0] + (i & 1 ? s : -s), c[1] + (i & 2 ? s : -s), c[2] + (i & 4 ? s : -s)]);
    }
    [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]].forEach((e) =>
      line(v[e[0]], v[e[1]], ci, g, a)
    );
  }

  /* Transformer stack: layers are horizontal sheets (token × hidden dim),
     input tokens at the bottom, output distribution on top. */

  const TL = compact ? 5 : 6;
  const TT = compact ? 8 : 12;
  const TD = compact ? 6 : 8;
  const SX = 34;
  const SZ = 28;
  const SY = 72;
  const Y0 = -60;
  const xOf = (t) => (t - (TT - 1) / 2) * SX;
  const zOf = (d) => (d - (TD - 1) / 2) * SZ;
  const yOf = (l) => Y0 + l * SY;
  const wOf = (l) => l / (TL - 1);
  const halfX = ((TT - 1) / 2) * SX + 22;
  const halfZ = ((TD - 1) / 2) * SZ + 18;
  const yTop = yOf(TL - 1);
  const yIn = Y0 - 60;
  const target = Math.floor(TT * 0.3); // the token the retrieved fact lands on

  for (let l = 0; l < TL; l += 1) {
    const y = yOf(l);
    const w = wOf(l);
    for (let t = 0; t < TT; t += 1) {
      for (let d = 0; d < TD; d += 1) {
        const forget = l > 0 && l < TL - 1 && rand() < 0.12;
        point([xOf(t), y, zOf(d)], C.NEURON, forget ? G.FORGET : G.NEURON, 0.8, 4.4, w);
        if (l < TL - 1) {
          for (let k = 0; k < 2; k += 1) {
            const d2 = clamp(d + Math.round(gauss() * 2.4), 0, TD - 1);
            line([xOf(t), y, zOf(d)], [xOf(t), yOf(l + 1), zOf(d2)], C.NEURON, G.NEURON, 0.16, w, wOf(l + 1));
          }
        }
      }
    }
    // Layer plate.
    const c = [[-halfX, -halfZ], [halfX, -halfZ], [halfX, halfZ], [-halfX, halfZ]];
    for (let k = 0; k < 4; k += 1) {
      const a = c[k];
      const b = c[(k + 1) % 4];
      line([a[0], y, a[1]], [b[0], y, b[1]], C.RAIL, G.NEURON, 0.3, w);
    }
    // Causal attention arcs along the back edge.
    const arcs = Math.round(TT * 1.3);
    for (let k = 0; k < arcs; k += 1) {
      const i = 1 + Math.floor(rand() * (TT - 1));
      const j = Math.floor(rand() * i);
      const weight = rand() * rand();
      const h = 14 + (i - j) * 9;
      let prev = null;
      for (let s = 0; s <= 10; s += 1) {
        const f = s / 10;
        const p = [lerp(xOf(j), xOf(i), f), y + Math.sin(Math.PI * f) * h, -halfZ];
        if (prev) line(prev, p, C.ATTN, G.ATTN, 0.1 + 0.55 * weight, w);
        prev = p;
      }
    }
  }

  // Input tokens and residual rails up the front edge.
  for (let t = 0; t < TT; t += 1) {
    const hit = t === target ? 1 : 0;
    point([xOf(t), yIn, halfZ], C.RAIL, G.NEURON, 0.85, 6, -0.25, hit);
    let prev = [xOf(t), yIn, halfZ];
    let pw = -0.25;
    for (let l = 0; l < TL; l += 1) {
      const p = [xOf(t), yOf(l), halfZ];
      line(prev, p, C.RAIL, G.NEURON, 0.42, pw, wOf(l), hit);
      prev = p;
      pw = wOf(l);
    }
  }

  // Output distribution: bars whose heights the shader animates; one token
  // wins after each forward pass.
  const bars = compact ? 10 : 16;
  const yOut = yTop + 46;
  const winner = Math.floor(bars * 0.62);
  line([-halfX, yOut, 0], [halfX, yOut, 0], C.OUT, G.OUT, 0.3, 0);
  for (let i = 0; i < bars; i += 1) {
    const x = lerp(-halfX + 8, halfX - 8, i / (bars - 1));
    const rate = 0.4 + rand() * 0.8;
    const off = (i === winner ? -1 : 1) * (0.5 + rand() * 6);
    line([x, yOut, 0], [x, yOut, 0], C.OUT, G.OUT, 0.7, 0, 1, rate, off);
    point([x, yOut, 0], C.OUT, G.OUT, 0.8, 3.2, 1, rate, off);
  }

  // Mixture-of-experts router: small, medium, large experts. Most traffic
  // goes to the small one.
  const yM = yOf(Math.floor(TL / 2));
  const router = [halfX + 95, yM, 0];
  point(router, C.MOE, G.MOE, 0.95, 7);
  for (let d = 0; d < TD; d += 2) line([halfX, yM, zOf(d)], router, C.MOE, G.MOE, 0.22);
  const experts = [
    { s: 16, y: yM + 82, rate: 0.62, streams: 3 },
    { s: 26, y: yM, rate: 0.34, streams: 2 },
    { s: 40, y: yM - 96, rate: 0.17, streams: 1 },
  ];
  const expertX = halfX + 250;
  experts.forEach((e) => {
    box([expertX + e.s, e.y, 0], e.s, C.MOE, G.MOE, 0.32);
    const face = [expertX, e.y, 0];
    const pts = [];
    for (let s = 0; s <= 15; s += 1) {
      pts.push(bezier(router, [router[0] + 60, router[1], 0], [face[0] - 70, face[1], 0], face, s / 15));
    }
    for (let k = 0; k < e.streams; k += 1) path(pts, C.MOE, G.MOE, 0.1, 4.2, e.rate, k / e.streams, k ? 0 : 0.22);
  });

  // Retrieval memory: a panel of documents left of the stack. One is
  // retrieved each cycle and travels into the context row.
  const memRows = compact ? 4 : 5;
  const memCols = compact ? 4 : 6;
  const memX = -halfX - 250;
  const memZ = halfZ;
  let picked = null;
  for (let r = 0; r < memRows; r += 1) {
    for (let c = 0; c < memCols; c += 1) {
      const cy = Y0 - 40 + r * 46;
      const cx = memX + (c - (memCols - 1) / 2) * 40;
      const hit = r === 1 && c === memCols - 1 ? 1 : 0;
      if (hit) picked = [cx, cy, memZ];
      const sx = 15;
      const sy = 18;
      const k = [[-sx, -sy], [sx, -sy], [sx, sy], [-sx, sy]];
      for (let i = 0; i < 4; i += 1) {
        const a = k[i];
        const b = k[(i + 1) % 4];
        line([cx + a[0], cy + a[1], memZ], [cx + b[0], cy + b[1], memZ], C.MEMORY, G.MEMORY, 0.45, -1, -1, hit);
      }
      for (let i = 0; i < 3; i += 1) {
        const ly = cy + 9 - i * 8;
        line([cx - 9, ly, memZ], [cx + 9 - (i === 2 ? 7 : 0), ly, memZ], C.MEMORY, G.MEMORY, 0.28, -1, -1, hit);
      }
    }
  }
  const landing = [xOf(target), yIn, halfZ];
  const beam = [];
  for (let s = 0; s <= 43; s += 1) {
    beam.push(
      bezier(picked, [picked[0] + 60, picked[1] + 150, picked[2] + 40], [landing[0] - 80, landing[1] + 70, landing[2] + 60], landing, s / 43)
    );
  }
  path(beam, C.BEAM, G.BEAM, 0.1, 4, 0, 0, 0.18);

  // Loss landscape off to the side and below, with three descent paths
  // (one per objective) and an unlearning path climbing back out of a basin.
  const LX = 780;
  const LZ = 140;
  const LY = -380;
  const LS = 300;
  const LH = 90;
  const GN = compact ? 24 : 36;

  function gb(u, v, a, b, s) {
    return Math.exp(-((u - a) * (u - a) + (v - b) * (v - b)) / (2 * s * s));
  }

  function loss(u, v) {
    return (
      0.45 * (u * u + v * v) -
      0.95 * gb(u, v, -0.42, -0.3, 0.2) -
      0.8 * gb(u, v, 0.48, -0.22, 0.17) -
      0.85 * gb(u, v, 0.05, 0.52, 0.19) +
      0.12 * Math.sin(3.1 * u + 0.4) * Math.cos(2.7 * v - 0.3)
    );
  }

  const ground = (u, v, lift) => [LX + u * LS, LY + LH * loss(u, v) + (lift || 0), LZ + v * LS];

  function slope(u, v) {
    const e = 1e-3;
    return [(loss(u + e, v) - loss(u - e, v)) / (2 * e), (loss(u, v + e) - loss(u, v - e)) / (2 * e)];
  }

  const grid = [];
  for (let i = 0; i < GN; i += 1) {
    grid.push([]);
    for (let j = 0; j < GN; j += 1) {
      const u = -1 + (2 * i) / (GN - 1);
      const v = -1 + (2 * j) / (GN - 1);
      const h = loss(u, v);
      const edge = 1 - smooth(0.72, 1, Math.max(Math.abs(u), Math.abs(v)));
      grid[i].push({ p: ground(u, v), a: (0.08 + 0.26 * (1 - clamp((h + 0.7) / 1.5, 0, 1))) * edge });
    }
  }
  for (let i = 0; i < GN; i += 1) {
    for (let j = 0; j < GN; j += 1) {
      const n = grid[i][j];
      if (i + 1 < GN) line(n.p, grid[i + 1][j].p, C.LAND, G.LAND, Math.min(n.a, grid[i + 1][j].a));
      if (j + 1 < GN) line(n.p, grid[i][j + 1].p, C.LAND, G.LAND, Math.min(n.a, grid[i][j + 1].a));
    }
  }

  // Evenly spaced points along a (u, v) path, lifted off the surface.
  function resample(uv, n, lift) {
    const pts = uv.map((q) => ground(q[0], q[1], lift));
    const acc = [0];
    for (let k = 1; k < pts.length; k += 1) acc.push(acc[k - 1] + Math.sqrt(dist2(pts[k - 1], pts[k])));
    const total = acc[acc.length - 1] || 1;
    const out = [];
    let k = 1;
    for (let i = 0; i < n; i += 1) {
      const d = (total * i) / (n - 1);
      while (k < acc.length - 1 && acc[k] < d) k += 1;
      const f = clamp((d - acc[k - 1]) / (acc[k] - acc[k - 1] || 1), 0, 1);
      const a = uv[k - 1];
      const b = uv[k];
      out.push(ground(lerp(a[0], b[0], f), lerp(a[1], b[1], f), lift));
    }
    return out;
  }

  // Momentum gradient descent from three starts lands in three basins.
  const minima = [];
  const trailN = compact ? 50 : 90;
  [[-0.85, -0.8], [0.92, -0.65], [-0.35, 0.95]].forEach((start, i) => {
    let u = start[0];
    let v = start[1];
    let mu = 0;
    let mv = 0;
    const uv = [[u, v]];
    for (let k = 0; k < 800; k += 1) {
      const s = slope(u, v);
      mu = 0.86 * mu - 0.02 * s[0];
      mv = 0.86 * mv - 0.02 * s[1];
      u += mu;
      v += mv;
      uv.push([u, v]);
      if (k > 20 && Math.hypot(mu, mv) < 2e-4) break;
    }
    minima.push([u, v]);
    path(resample(uv, trailN, 8), C.ACC + i, G.DESCENT, 0.8, 5.5, 1 / 7, i * 0.31, 0.3);
    let prev = null;
    for (let k = 0; k <= 12; k += 1) {
      const a = (k / 12) * TAU;
      const p = ground(u + Math.cos(a) * 0.03, v + Math.sin(a) * 0.03, 3);
      if (prev) line(prev, p, C.ACC + i, G.DESCENT, 0.6);
      prev = p;
    }
  });
  // Unlearning: gradient ascent back out of the first basin.
  (function () {
    let u = minima[0][0];
    let v = minima[0][1];
    const uv = [[u, v]];
    for (let k = 0; k < 90; k += 1) {
      const s = slope(u, v);
      const m = Math.hypot(s[0], s[1]) || 1;
      u += (s[0] / m) * 0.006 + 0.0015;
      v += (s[1] / m) * 0.006;
      uv.push([u, v]);
    }
    path(resample(uv, Math.round(trailN * 0.6), 8), C.ASCENT, G.ASCENT, 0.8, 5.5, 1 / 5, 0.5, 0.3);
  })();

  // Accelerator die under the stack: a systolic grid with data pulsing
  // along rows and columns.
  const CY = -215;
  const CN = compact ? 8 : 12;
  const CW = 300;
  const chipAt = (i, j) => [-CW / 2 + (i * CW) / (CN - 1), CY, -CW / 2 + (j * CW) / (CN - 1)];
  for (let i = 0; i < CN; i += 1) {
    for (let j = 0; j < CN; j += 1) {
      point(chipAt(i, j), C.CHIP, G.CHIP, 0.5, 3);
      if (i + 1 < CN) line(chipAt(i, j), chipAt(i + 1, j), C.CHIP, G.CHIP, 0.22, i / (CN - 1), (i + 1) / (CN - 1), 0.32, j * 0.137);
      if (j + 1 < CN) line(chipAt(i, j), chipAt(i, j + 1), C.CHIP, G.CHIP, 0.22, j / (CN - 1), (j + 1) / (CN - 1), 0.27, i * 0.173 + 0.5);
    }
  }
  (function () {
    const e = CW / 2 + 22;
    const c = [[-e, -e], [e, -e], [e, e], [-e, e]];
    for (let k = 0; k < 4; k += 1) {
      const a = c[k];
      const b = c[(k + 1) % 4];
      line([a[0], CY, a[1]], [b[0], CY, b[1]], C.CHIP, G.CHIP, 0.4);
      const nx = b[1] - a[1];
      const nz = a[0] - b[0];
      const m = Math.hypot(nx, nz);
      for (let s = 1; s < 10; s += 1) {
        const x = lerp(a[0], b[0], s / 10);
        const z = lerp(a[1], b[1], s / 10);
        line([x, CY, z], [x + (nx / m) * 14, CY, z + (nz / m) * 14], C.CHIP, G.CHIP, 0.3);
      }
    }
  })();

  // Embedding cloud: clusters spread around everything, joined into a
  // 2-degenerate k-NN graph (each point links to at most two older points),
  // so greedy colouring always works and, with four colours, any proper
  // colouring can reach any other one vertex at a time (k >= degeneracy + 2).
  // The Graph Theory tag shows that off.
  const cloud = [];
  const cloudAdj = [];
  const cloudColor = [];
  (function () {
    const K = compact ? 6 : 10;
    const per = compact ? 22 : 34;
    const pos = [];
    const back = [];
    const hubs = [];
    function link(a, b) {
      if (a === b || back[a] >= 2 || cloudAdj[a].indexOf(b) !== -1) return;
      back[a] += 1;
      cloudAdj[a].push(b);
      cloudAdj[b].push(a);
      line(pos[a], pos[b], C.EDGE, G.CLOUD_EDGE, 0.16);
    }
    for (let k = 0; k < K; k += 1) {
      const u = rand() * TAU;
      const v = Math.acos(2 * rand() - 1);
      const r = 900 + rand() * 500;
      const c = [Math.sin(v) * Math.cos(u) * r, Math.cos(v) * r * 0.5 + 40, Math.sin(v) * Math.sin(u) * r];
      const start = pos.length;
      for (let n = 0; n < per; n += 1) {
        const i = pos.length;
        pos.push(n === 0 ? c : [c[0] + gauss() * 70, c[1] + gauss() * 55, c[2] + gauss() * 70]);
        back.push(0);
        cloudAdj.push([]);
        if (n === 0) {
          // Each cluster's hub joins the nearest earlier hub.
          if (hubs.length) {
            let best = hubs[0];
            hubs.forEach((h) => {
              if (dist2(pos[i], pos[h]) < dist2(pos[i], pos[best])) best = h;
            });
            link(i, best);
          }
          hubs.push(i);
        } else {
          const near = [];
          for (let j = start; j < i; j += 1) near.push([dist2(pos[i], pos[j]), j]);
          near.sort((a, b) => a[0] - b[0]);
          link(i, near[0][1]);
          if (near[1] && rand() < 0.45) link(i, near[1][1]);
        }
      }
    }
    // Greedy colouring in insertion order, randomised over four colours.
    for (let i = 0; i < pos.length; i += 1) {
      const used = cloudAdj[i].filter((j) => j < i).map((j) => cloudColor[j]);
      const free = [0, 1, 2, 3].filter((c) => used.indexOf(c) === -1);
      cloudColor.push(free[Math.floor(rand() * free.length)]);
      cloud.push(P.length / STRIDE);
      point(pos[i], C.CLOUD + cloudColor[i], G.CLOUD, 0.5, 3.2, 0);
    }
  })();

  const Pf = new Float32Array(P);
  const Lf = new Float32Array(L);
  const nP = Pf.length / STRIDE;
  const nL = Lf.length / STRIDE;

  // Centroid of each group, for camera nudges toward highlighted things.
  const centre = [];
  (function () {
    const sum = [];
    for (let g = 0; g < GROUPS; g += 1) sum.push([0, 0, 0, 0]);
    [Pf, Lf].forEach((f) => {
      for (let i = 0; i < f.length; i += STRIDE) {
        const s = sum[f[i + 6]];
        s[0] += f[i];
        s[1] += f[i + 1];
        s[2] += f[i + 2];
        s[3] += 1;
      }
    });
    sum.forEach((s) => centre.push(s[3] ? [s[0] / s[3], s[1] / s[3], s[2] / s[3]] : [0, 0, 0]));
  })();

  function mean(list) {
    const c = [0, 0, 0];
    list.forEach((p) => {
      c[0] += p[0] / list.length;
      c[1] += p[1] / list.length;
      c[2] += p[2] / list.length;
    });
    return c;
  }

  /* Viewpoints ---------------------------------------------------------
     t: look-at target, d: distance, yaw/pitch in radians, shift: where the
     target sits horizontally (-1 left margin … 1 right margin, wide screens
     only), hue: palette rotation in degrees. */

  const VIEWS = {
    // Wide establishing shot: the whole model over its loss landscape.
    hero: { t: [0, 40, 0], d: 1180, yaw: 0.55, pitch: 0.16, shift: 1, hue: 0 },
    // Down onto the landscape: three objectives, three descents, and the
    // unlearning climb.
    research: { t: centre[G.DESCENT], d: 980, yaw: 1.4, pitch: 0.85, shift: 1.1, hue: 10 },
    // Retrieval into the context row: in-context knowledge editing.
    publications: { t: mean([centre[G.MEMORY], landing]), d: 760, yaw: -0.3, pitch: 0.12, shift: -0.6, hue: -12 },
    // Pan from the router across to its experts.
    experienceA: { t: [router[0] + 60, yM, 0], d: 600, yaw: 1.05, pitch: 0.1, shift: 1, hue: -4 },
    experienceB: { t: [expertX + 30, yM - 10, 0], d: 580, yaw: 1.35, pitch: 0.04, shift: 1, hue: -6 },
    // Orbit the stack and the die it runs on.
    projectsA: { t: [0, -20, 0], d: 980, yaw: 2.3, pitch: 0.3, shift: 1.05, hue: 6 },
    projectsB: { t: [0, -20, 0], d: 980, yaw: 3.8, pitch: 0.26, shift: 1.05, hue: 8 },
    // Pull back above everything.
    contact: { t: [120, -60, 40], d: 1700, yaw: 4.4, pitch: 0.85, shift: 1, hue: 0 },
  };

  const newView = () => ({ t: [0, 0, 0], d: 0, yaw: 0, pitch: 0, shift: 0, hue: 0 });

  function copyView(src, out) {
    out.t[0] = src.t[0];
    out.t[1] = src.t[1];
    out.t[2] = src.t[2];
    out.d = src.d;
    out.yaw = src.yaw;
    out.pitch = src.pitch;
    out.shift = src.shift;
    out.hue = src.hue;
    return out;
  }

  function blendView(a, b, t, out) {
    out.t[0] = lerp(a.t[0], b.t[0], t);
    out.t[1] = lerp(a.t[1], b.t[1], t);
    out.t[2] = lerp(a.t[2], b.t[2], t);
    // Distance blends in log space so zooms feel even.
    out.d = Math.exp(lerp(Math.log(a.d), Math.log(b.d), t));
    out.yaw = lerp(a.yaw, b.yaw, t);
    out.pitch = lerp(a.pitch, b.pitch, t);
    out.shift = lerp(a.shift, b.shift, t);
    out.hue = lerp(a.hue, b.hue, t);
    return out;
  }

  /* Scroll → viewpoint ------------------------------------------------- */

  let keys = [];
  let vh = window.innerHeight;

  function measure() {
    vh = window.innerHeight;
    const y = window.scrollY;
    const max = Math.max(1, root.scrollHeight - vh);
    const box = (id) => {
      const el = document.getElementById(id);
      if (!el) return null;
      const r = el.getBoundingClientRect();
      return { top: r.top + y, h: r.height };
    };
    // Each section holds its viewpoint (or pans A → B) while it fills the
    // viewport; the camera only travels while a section boundary crosses
    // the middle of the screen (a band 0.3 viewport tall).
    const plan = [
      ["hero", VIEWS.hero],
      ["research", VIEWS.research],
      ["publications", VIEWS.publications],
      ["experience", VIEWS.experienceA, VIEWS.experienceB],
      ["projects", VIEWS.projectsA, VIEWS.projectsB],
    ];
    const k = [];
    plan.forEach((step, i) => {
      const b = i === 0 ? { top: 0, h: (box("research") || { top: vh }).top } : box(step[0]);
      if (!b) return;
      let sIn = i === 0 ? 0 : b.top - vh * 0.35;
      // Short sections keep their view until their top nears the header.
      let sOut = Math.max(b.top + b.h - vh * 0.65, b.top - vh * 0.05);
            // The last section orbits until the footer arrives; the pull-back
      // takes only the final stretch of scroll.
      if (i === plan.length - 1) sOut = Math.max(sOut, max - vh * 0.12);
      if (sOut < sIn) sIn = sOut = (sIn + sOut) / 2;
      k.push({ s: sIn, v: step[1], sv: step[1] });
      k.push({ s: sOut, v: step[2] || step[1], sv: step[1], pan: !!step[2] });
    });
    k.push({ s: max, v: VIEWS.contact, sv: VIEWS.contact });
    for (let i = 1; i < k.length; i += 1) {
      k[i].s = clamp(Math.max(k[i].s, k[i - 1].s), 0, max);
    }
    keys = k;
  }

  function segment(s) {
    for (let i = 0; i < keys.length - 1; i += 1) {
      if (s < keys[i + 1].s) {
        const a = keys[i];
        const b = keys[i + 1];
        return { a, b, t: b.s > a.s ? clamp((s - a.s) / (b.s - a.s), 0, 1) : 1 };
      }
    }
    return null;
  }

  function viewAt(s, out) {
    if (s <= keys[0].s) return copyView(keys[0].v, out);
    const g = segment(s);
    if (!g) return copyView(keys[keys.length - 1].v, out);
    const t = g.b.pan ? g.t : easeInOut(g.t);
    return blendView(g.a.v, g.b.v, t, out);
  }

  // Reduced motion / static fallback: one viewpoint per section.
  function staticView(s) {
    if (s <= keys[0].s) return keys[0].sv;
    const g = segment(s);
    if (!g) return keys[keys.length - 1].sv;
    return g.t < 0.5 || g.b.pan ? g.a.sv : g.b.sv;
  }

  /* Palette ------------------------------------------------------------
     Dark: additive glow. Light: ink on paper with normal blending. */

  const THEMES = {
    dark: {
      pal: [
        [0.45, 0.72, 0.95], [0.55, 0.8, 1.0], [0.62, 0.6, 1.0], [0.4, 0.9, 0.8], [0.7, 0.85, 1.0],
        [0.3, 0.5, 0.85], [0.4, 0.85, 1.0], [0.45, 0.95, 0.65], [0.8, 0.62, 1.0], [1.0, 0.5, 0.45],
        [0.75, 0.78, 1.0], [0.95, 0.85, 0.55], [0.35, 0.8, 0.7], [0.45, 0.55, 0.75],
        [0.55, 0.75, 0.95], [0.45, 0.85, 0.8], [0.75, 0.65, 0.95], [0.95, 0.75, 0.5],
      ],
      hot: [0.85, 0.97, 1.0],
      grad: [1.0, 0.68, 0.3],
      alpha: 1.45,
      scrim: 0.2,
      glow: 1,
    },
    light: {
      pal: [
        [0.08, 0.27, 0.5], [0.08, 0.3, 0.55], [0.3, 0.25, 0.6], [0.05, 0.42, 0.4], [0.1, 0.3, 0.55],
        [0.2, 0.32, 0.5], [0.0, 0.42, 0.7], [0.05, 0.5, 0.32], [0.4, 0.25, 0.65], [0.75, 0.22, 0.15],
        [0.25, 0.28, 0.55], [0.75, 0.45, 0.0], [0.05, 0.4, 0.35], [0.35, 0.42, 0.55],
        [0.1, 0.35, 0.6], [0.05, 0.45, 0.42], [0.38, 0.28, 0.62], [0.7, 0.42, 0.1],
      ],
      hot: [0.0, 0.5, 0.9],
      grad: [0.85, 0.4, 0.0],
      alpha: 1.15,
      scrim: 0.26,
      glow: 0,
    },
  };
  let theme = THEMES.light;
  let palFlat = new Float32Array(COLORS * 3);

  function isDark() {
    const attr = root.getAttribute("data-theme");
    if (attr === "dark") return true;
    if (attr === "light") return false;
    return darkMQ.matches;
  }

  function syncTheme() {
    theme = isDark() ? THEMES.dark : THEMES.light;
    palFlat = new Float32Array([].concat.apply([], theme.pal));
    kick();
  }

  /* Shaders ------------------------------------------------------------ */

  const VS = [
    "precision highp float;",
    "attribute vec3 a_pos; attribute float a_ci; attribute vec4 a_p; attribute vec3 a_q;",
    "uniform vec3 u_eye; uniform vec3 u_r; uniform vec3 u_u; uniform vec3 u_f;",
    "uniform vec2 u_res; uniform vec2 u_center; uniform vec2 u_ptr;",
    "uniform float u_focal; uniform float u_d; uniform float u_time;",
    "uniform float u_fwd; uniform float u_bwd; uniform float u_beam; uniform float u_arrive; uniform float u_sharp;",
    "uniform float u_ptrS; uniform float u_focusOn; uniform float u_fade; uniform float u_scrimMin; uniform float u_col;",
    "uniform float u_dpr; uniform float u_alpha; uniform float u_hue;",
    "uniform float u_focus[" + GROUPS + "];",
    "uniform vec3 u_pal[" + COLORS + "];",
    "uniform vec3 u_hot; uniform vec3 u_grad; uniform float u_size;",
    "varying vec4 v_col;",
    "bool is(float g, float v) { return abs(g - v) < 0.5; }",
    "vec3 hueShift(vec3 c, float a) {",
    "  vec3 k = vec3(0.57735); float ca = cos(a);",
    "  return c * ca + cross(k, c) * sin(a) + k * dot(k, c) * (1.0 - ca);",
    "}",
    "void main() {",
    "  vec3 pos = a_pos;",
    "  float w = a_p.x; float ph = a_p.y; float g = a_p.z;",
    "  float rate = a_q.x; float off = a_q.y; float t = u_time;",
    "  vec3 col = u_pal[0];",
    "  for (int k = 0; k < " + COLORS + "; k++) { if (float(k) == a_ci) col = u_pal[k]; }",
    "  float act = 0.72 + 0.28 * sin(t * (0.6 + fract(ph * 7.31) * 1.7) + ph * 6.2832);",
    "  float boost = 0.0; float grow = 0.0;",
    // Transformer: forward pass up (cool), backward pass down (warm).
    "  if (g < 2.5) {",
    "    float fw = exp(-pow((w - u_fwd) * 5.0, 2.0));",
    "    float bw = exp(-pow((w - u_bwd) * 5.0, 2.0));",
    "    col = mix(col, u_hot, fw * 0.8); col = mix(col, u_grad, bw * 0.85);",
    "    boost += fw * 3.0 + bw * 2.6; grow += fw * 1.0 + bw * 0.9;",
    "    if (rate > 0.5) { boost += u_arrive * 3.0; col = mix(col, u_hot, u_arrive * 0.6); }",
    "    if (is(g, 1.0)) { float c = 0.5 + 0.5 * cos(t * 0.35 + ph * 6.2832); act *= 1.0 - 0.92 * c * c * c; }",
    // Output distribution sharpens onto one token after a forward pass.
    "  } else if (is(g, 4.0)) {",
    "    float h = pow(0.5 + 0.5 * sin(t * rate + abs(off)), 3.0);",
    "    h = mix(0.06 + 0.55 * h, off < 0.0 ? 1.0 : 0.06, u_sharp);",
    "    pos.y += w * (6.0 + 90.0 * h);",
    "    boost += off < 0.0 ? u_sharp * 1.5 : 0.0;",
    // Optimiser paths: a comet with a fading trail.
    "  } else if (is(g, 6.0) || is(g, 7.0)) {",
    "    if (w >= 0.0) {",
    "      float dd = fract(t * rate + off) - w;",
    "      float trail = dd >= 0.0 ? exp(-dd * 4.0) : 0.0;",
    "      float tip = exp(-pow(dd * 40.0, 2.0));",
    "      act = 0.15 + 0.85 * trail; boost += tip * 4.0; grow += tip * 1.8;",
    "    }",
    // Router traffic and systolic data flow: pulses along w.
    "  } else if (is(g, 3.0) || is(g, 10.0)) {",
    "    if (w >= 0.0) {",
    "      float p = exp(-pow((w - fract(t * rate + off)) * 12.0, 2.0));",
    "      boost += p * (is(g, 3.0) ? 8.0 : 5.0); grow += p * 1.4; col = mix(col, u_hot, p * 0.5);",
    "    }",
    // The retrieved document glows as it leaves; the beam carries it in.
    "  } else if (is(g, 8.0)) {",
    "    if (rate > 0.5 && u_beam > -1.0) boost += (1.0 - smoothstep(0.0, 0.4, u_beam)) * 4.0;",
    "  } else if (is(g, 9.0)) {",
    "    if (w >= 0.0 && u_beam > -1.0) {",
    "      float dd = u_beam - w;",
    "      float p = dd >= 0.0 ? exp(-dd * 7.0) : exp(-pow(dd * 30.0, 2.0));",
    "      boost += p * 6.0; grow += p * 1.3;",
    "    }",
    // Embedding points flash when recoloured.
    "  } else if (is(g, 11.0)) {",
    "    boost += w * 3.0; grow += w * 1.5;",
    "  }",
    "  float f = 0.0;",
    "  for (int k = 0; k < " + GROUPS + "; k++) { if (float(k) == floor(g + 0.5)) f = u_focus[k]; }",
    "  float lit = f * u_focusOn;",
    "  float fm = 1.0 + 1.4 * lit - 0.55 * u_focusOn * (1.0 - f);",
    "  vec3 d = pos - u_eye;",
    "  float z = dot(d, u_f);",
    "  float zc = max(z, 10.0);",
    "  vec2 s = vec2(dot(d, u_r), dot(d, u_u)) * (u_focal / zc);",
    "  vec2 px = u_center + vec2(s.x, -s.y);",
    "  float prox = 0.0;",
    "  if (u_ptrS > 0.01) {",
    "    vec2 o = px - u_ptr; float dl = length(o);",
    "    if (dl < 160.0) {",
    "      prox = 1.0 - dl / 160.0;",
    "      if (dl > 0.5) px += o / dl * prox * prox * 20.0 * u_ptrS;",
    "      prox *= u_ptrS;",
    "    }",
    "  }",
    "  float zr = z / u_d;",
    "  float fog = smoothstep(0.04, 0.3, zr) * (1.0 - 0.85 * smoothstep(1.0, 2.6, zr));",
    // Scrim: everything dims behind the text column.
    "  float sc = mix(u_scrimMin, 1.0, smoothstep(u_col - 30.0, u_col + 170.0, abs(px.x - u_res.x * 0.5)));",
    "  float a = a_p.w * act * (1.0 + boost) * fog * fm * (1.0 + 1.2 * prox) * u_alpha;",
    "  a = min(a * sc, mix(0.3, 1.0, (sc - u_scrimMin) / (1.0 - u_scrimMin))) * u_fade;",
    "  v_col = vec4(clamp(hueShift(col, u_hue), 0.0, 1.0), clamp(a, 0.0, 1.0));",
    "  vec2 clip = px / u_res * 2.0 - 1.0;",
    "  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);",
    "#ifdef POINTS",
    "  gl_PointSize = clamp(a_q.z * u_focal / zc * u_size, 1.5, 22.0) * (1.0 + grow) * (1.0 + 0.35 * lit) * u_dpr;",
    "#endif",
    "}",
  ].join("\n");

  const FS_POINTS = [
    "precision mediump float;",
    "varying vec4 v_col; uniform float u_glow;",
    "void main() {",
    "  float r = length(gl_PointCoord - 0.5) * 2.0;",
    "  if (r > 1.0) discard;",
    "  float disc = 1.0 - smoothstep(0.5, 1.0, r);",
    "  float glow = (1.0 - smoothstep(0.0, 0.42, r)) * 0.85 + pow(1.0 - r, 2.5) * 0.6;",
    "  float a = v_col.a * mix(disc, glow, u_glow);",
    "  gl_FragColor = vec4(v_col.rgb * a, a);",
    "}",
  ].join("\n");

  const FS_LINES = [
    "precision mediump float;",
    "varying vec4 v_col; uniform float u_glow;",
    "void main() { gl_FragColor = vec4(v_col.rgb * v_col.a, v_col.a); }",
  ].join("\n");

  const UNIFORMS = [
    "u_eye", "u_r", "u_u", "u_f", "u_res", "u_center", "u_ptr", "u_focal", "u_d", "u_time",
    "u_fwd", "u_bwd", "u_beam", "u_arrive", "u_sharp", "u_ptrS", "u_focusOn", "u_fade",
    "u_scrimMin", "u_col", "u_dpr", "u_alpha", "u_hue", "u_focus[0]", "u_pal[0]", "u_hot",
    "u_grad", "u_glow", "u_size",
  ];

  let progP = null;
  let progL = null;
  let bufP = null;
  let bufL = null;

  function compile(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }

  function program(points) {
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, (points ? "#define POINTS\n" : "") + VS));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, points ? FS_POINTS : FS_LINES));
    ["a_pos", "a_ci", "a_p", "a_q"].forEach((n, i) => gl.bindAttribLocation(p, i, n));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {};
    UNIFORMS.forEach((n) => (u[n.replace("[0]", "")] = gl.getUniformLocation(p, n)));
    return { p, u };
  }

  function setupGL() {
    progP = program(true);
    progL = program(false);
    bufP = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, bufP);
    gl.bufferData(gl.ARRAY_BUFFER, Pf, gl.DYNAMIC_DRAW);
    bufL = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, bufL);
    gl.bufferData(gl.ARRAY_BUFFER, Lf, gl.STATIC_DRAW);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.clearColor(0, 0, 0, 0);
  }

  function bindAttribs(buf) {
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    const b = STRIDE * 4;
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, b, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, b, 12);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 4, gl.FLOAT, false, b, 16);
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 3, gl.FLOAT, false, b, 32);
  }

  /* State -------------------------------------------------------------- */

  let W = 0;
  let H = 0;
  let dpr = 1;
  let dprCap = compact ? 1.25 : 1.5;
  let focal = 1;
  let shiftMax = 0;
  let lost = false;

  let staticMode = reduceMQ.matches;
  let quality = 0; // 0 full, 1 reduced; static mode is the last step.

  const cam = newView();
  const goal = newView();
  const view = newView();
  const fromView = newView();
  let shown = null;
  let fadeT = 1;

  const ptr = { x: 0, y: 0, tx: 0, ty: 0, s: 0, on: false, moved: 0 };
  const focus = { groups: new Float32Array(GROUPS), amt: 0, target: 0, c: [0, 0, 0] };
  const U = { fwd: -9, bwd: -9, beam: -9, arrive: 0, sharp: 0 };

  let recolorOrder = null;
  let recolorAt = 0;
  let recolorAcc = 0;
  let recolorSteps = 0;
  const flashing = new Set();

  let time = 0;
  let raf = 0;
  let last = 0;
  let lastDraw = 0;
  let started = false;

  /* Sizing ------------------------------------------------------------- */

  function resize() {
    const rect = canvas.getBoundingClientRect();
    W = Math.max(1, rect.width);
    H = Math.max(1, rect.height);
    dpr = Math.min(window.devicePixelRatio || 1, dprCap);
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    if (!lost) gl.viewport(0, 0, canvas.width, canvas.height);
    focal = Math.max(W, H) * 0.78;
    shiftMax = smooth(980, 1280, W) * Math.min(W * 0.3, (W - 760) / 2 + 120);
    measure();
    kick();
  }

  /* Drawing ------------------------------------------------------------ */

  function uniforms(prog, v, fade, moving) {
    const u = prog.u;
    const cp = Math.cos(v.pitch);
    const sp = Math.sin(v.pitch);
    const cy = Math.cos(v.yaw);
    const sy = Math.sin(v.yaw);
    gl.uniform3f(u.u_eye, v.t[0] + v.d * cp * sy, v.t[1] + v.d * sp, v.t[2] + v.d * cp * cy);
    gl.uniform3f(u.u_f, -cp * sy, -sp, -cp * cy);
    gl.uniform3f(u.u_r, cy, 0, -sy);
    gl.uniform3f(u.u_u, -sy * sp, cp, -cy * sp);
    gl.uniform2f(u.u_res, W, H);
    gl.uniform2f(u.u_center, W / 2 + v.shift * shiftMax, H / 2);
    gl.uniform2f(u.u_ptr, ptr.x, ptr.y);
    gl.uniform1f(u.u_focal, focal);
    gl.uniform1f(u.u_d, v.d);
    gl.uniform1f(u.u_time, moving ? time : 9.5);
    gl.uniform1f(u.u_fwd, moving ? U.fwd : -9);
    gl.uniform1f(u.u_bwd, moving ? U.bwd : -9);
    gl.uniform1f(u.u_beam, moving ? U.beam : -9);
    gl.uniform1f(u.u_arrive, moving ? U.arrive : 0);
    gl.uniform1f(u.u_sharp, moving ? U.sharp : 0);
    gl.uniform1f(u.u_ptrS, moving ? ptr.s : 0);
    gl.uniform1f(u.u_focusOn, focus.amt);
    gl.uniform1fv(u.u_focus, focus.groups);
    gl.uniform1f(u.u_fade, fade);
    gl.uniform1f(u.u_scrimMin, compact ? theme.scrim + 0.12 : theme.scrim);
    gl.uniform1f(u.u_col, Math.min(W, 760) / 2);
    gl.uniform1f(u.u_dpr, dpr);
    gl.uniform1f(u.u_alpha, theme.alpha);
    gl.uniform1f(u.u_hue, (v.hue * Math.PI) / 180);
    gl.uniform3fv(u.u_pal, palFlat);
    gl.uniform3fv(u.u_hot, theme.hot);
    gl.uniform3fv(u.u_grad, theme.grad);
    gl.uniform1f(u.u_glow, theme.glow);
    gl.uniform1f(u.u_size, 1 + 0.7 * theme.glow);
  }

  function draw(v, fade) {
    const moving = !staticMode;
    if (theme.glow) gl.blendFunc(gl.ONE, gl.ONE);
    else gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(progL.p);
    uniforms(progL, v, fade, moving);
    bindAttribs(bufL);
    gl.drawArrays(gl.LINES, 0, nL);
    gl.useProgram(progP.p);
    uniforms(progP, v, fade, moving);
    bindAttribs(bufP);
    gl.drawArrays(gl.POINTS, 0, nP);
  }

  /* Simulation --------------------------------------------------------- */

  // One cycle: retrieve a document into the context, forward pass up the
  // stack, the output sharpens onto a token, then the backward pass.
  const CYCLE = 7.6;
  function passes() {
    const s = time % CYCLE;
    U.beam = s < 1.4 ? s / 1.4 : -9;
    U.fwd = s >= 1.25 && s < 3.85 ? -0.3 + ((s - 1.25) / 2.6) * 1.6 : -9;
    U.bwd = s >= 4.3 && s < 6.9 ? 1.3 - ((s - 4.3) / 2.6) * 1.6 : -9;
    U.arrive = s >= 1.3 ? Math.max(0, 1 - (s - 1.3) / 1.6) : 0;
    U.sharp = smooth(3.3, 3.9, s) * (1 - smooth(5.6, 6.6, s));
  }

  function startRecolor() {
    const order = [0];
    const seen = new Uint8Array(cloud.length);
    seen[0] = 1;
    for (let i = 0; i < order.length; i += 1) {
      cloudAdj[order[i]].forEach((j) => {
        if (!seen[j]) {
          seen[j] = 1;
          order.push(j);
        }
      });
    }
    for (let i = 0; i < cloud.length; i += 1) if (!seen[i]) order.push(i);
    recolorOrder = order;
    recolorAt = 0;
    recolorAcc = 0;
    recolorSteps = 0;
    announce("Recoloring the embedding graph, one vertex at a time.");
    kick();
  }

  function stepRecolor(dt) {
    let dirty = false;
    if (recolorOrder) {
      recolorAcc += dt;
      const every = 4.2 / recolorOrder.length;
      while (recolorAcc >= every && recolorOrder) {
        recolorAcc -= every;
        const v = recolorOrder[recolorAt];
        const used = cloudAdj[v].map((j) => cloudColor[j]);
        const free = [0, 1, 2, 3].filter((c) => c !== cloudColor[v] && used.indexOf(c) === -1);
        if (free.length) {
          cloudColor[v] = free[Math.floor(rand() * free.length)];
          const base = cloud[v] * STRIDE;
          Pf[base + 3] = C.CLOUD + cloudColor[v];
          if (!staticMode) {
            Pf[base + 4] = 1;
            flashing.add(v);
          }
          recolorSteps += 1;
          dirty = true;
        }
        recolorAt += 1;
        if (recolorAt >= recolorOrder.length) {
          recolorOrder = null;
          announce(recolorSteps + " single-vertex recolorings, and the coloring stayed proper after every one.");
        }
      }
    }
    flashing.forEach((v) => {
      const i = cloud[v] * STRIDE + 4;
      Pf[i] = Math.max(0, Pf[i] - dt * 1.6);
      if (Pf[i] === 0) flashing.delete(v);
      dirty = true;
    });
    if (dirty && !lost) {
      gl.bindBuffer(gl.ARRAY_BUFFER, bufP);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, Pf);
    }
  }

  /* Frame loop --------------------------------------------------------- */

  const perf = { frames: 0, slow: 0, warm: 0 };

  function degrade() {
    quality += 1;
    if (quality === 1) {
      dprCap = 1;
      resize();
    } else {
      setStatic(true);
    }
  }

  // GPU work is asynchronous, so watch frame spacing rather than JS cost.
  function watchPerf(dt) {
    if (staticMode || document.hidden) return;
    perf.warm += dt;
    if (perf.warm < 1.5 || dt > 0.25) return;
    perf.frames += 1;
    if (dt > 0.026) perf.slow += 1;
    if (perf.frames < 90) return;
    if (perf.slow / perf.frames > 0.35) degrade();
    perf.frames = perf.slow = 0;
  }

  function frame(now) {
    raf = 0;
    if (lost || !W) return;
    const dt = Math.min(0.1, last ? (now - last) / 1000 : 0.016);
    last = now;

    if (staticMode) {
      const want = staticView(window.scrollY);
      if (!shown) {
        shown = want;
        fadeT = 1;
      } else if (want !== shown) {
        copyView(shown, fromView);
        shown = want;
        fadeT = 0;
      }
      focus.amt = focus.target;
      stepRecolor(dt);
      fadeT = Math.min(1, fadeT + dt / 0.7);
      gl.clear(gl.COLOR_BUFFER_BIT);
      if (fadeT < 1) {
        const t = easeInOut(fadeT);
        draw(fromView, 1 - t);
        draw(shown, t);
        raf = requestAnimationFrame(frame);
      } else {
        draw(shown, 1);
        if (recolorOrder) raf = requestAnimationFrame(frame);
      }
      reveal();
      return;
    }

    // Camera: damped toward the scroll-driven goal.
    viewAt(window.scrollY, goal);
    blendView(cam, goal, 1 - Math.exp(-dt * 4.2), cam);

    const kp = 1 - Math.exp(-dt * 7);
    ptr.x = lerp(ptr.x, ptr.tx, kp);
    ptr.y = lerp(ptr.y, ptr.ty, kp);
    ptr.s = lerp(ptr.s, ptr.on && fineMQ.matches ? 1 : 0, 1 - Math.exp(-dt * 3));
    focus.amt = lerp(focus.amt, focus.target, 1 - Math.exp(-dt * 6));
    if (!focus.target && focus.amt < 0.002) focus.amt = 0;

    copyView(cam, view);
    if (ptr.s > 0.001) {
      view.yaw += (ptr.x / W - 0.5) * 0.08 * ptr.s;
      view.pitch += (ptr.y / H - 0.5) * 0.05 * ptr.s;
    }
    if (focus.amt > 0.001) {
      const pull = 0.2 * focus.amt;
      view.t[0] = lerp(view.t[0], focus.c[0], pull);
      view.t[1] = lerp(view.t[1], focus.c[1], pull);
      view.t[2] = lerp(view.t[2], focus.c[2], pull);
    }

    time += dt;
    passes();
    stepRecolor(dt);

    // Once the camera has settled and nothing is happening, ambient motion
    // runs at half rate to save battery.
    const settled =
      Math.abs(cam.t[0] - goal.t[0]) + Math.abs(cam.t[1] - goal.t[1]) + Math.abs(cam.t[2] - goal.t[2]) < 0.6 &&
      Math.abs(cam.d - goal.d) < 0.6 &&
      Math.abs(cam.yaw - goal.yaw) < 0.0015 &&
      Math.abs(focus.amt - focus.target) < 0.01 &&
      now - ptr.moved > 1200 &&
      !recolorOrder &&
      !flashing.size;
    if (!settled || now - lastDraw > 31) {
      gl.clear(gl.COLOR_BUFFER_BIT);
      draw(view, 1);
      lastDraw = now;
      if (!settled) watchPerf(dt);
      reveal();
    }

    raf = requestAnimationFrame(frame);
  }

  function reveal() {
    if (started) return;
    started = true;
    canvas.classList.add("is-ready");
  }

  function kick() {
    if (!raf && !document.hidden && !lost) {
      last = 0;
      raf = requestAnimationFrame(frame);
    }
  }

  function setStatic(on) {
    staticMode = on;
    shown = null;
    if (!on) viewAt(window.scrollY, cam);
    perf.frames = perf.slow = perf.warm = 0;
    kick();
  }

  /* Foreground ↔ background links ---------------------------------------
     Hover (mouse), keyboard focus or tap (touch) on anything with a
     data-graph attribute lights up the matching part of the scene. */

  const KEY_GROUPS = {
    llm: [G.NEURON, G.FORGET, G.ATTN, G.OUT],
    agents: [G.MOE],
    rl: [G.LAND, G.DESCENT, G.ASCENT],
    graph: [G.CLOUD, G.CLOUD_EDGE],
    moa: [G.LAND, G.DESCENT],
    unlearn: [G.ASCENT, G.FORGET],
    edit: [G.MEMORY, G.BEAM],
    anytime: [G.MOE],
    bestlink: [G.MEMORY],
    moike: [G.MEMORY, G.BEAM],
    akita: [G.CHIP],
    recolor: [G.CLOUD, G.CLOUD_EDGE],
  };

  let hoverEl = null;
  let focusEl = null;
  let tapEl = null;
  let tapY = 0;
  let shownEl = null;

  function setFocus(spec) {
    const groups = [];
    (spec || "").split(/\s+/).forEach((k) => {
      (KEY_GROUPS[k] || []).forEach((g) => {
        if (groups.indexOf(g) === -1) groups.push(g);
      });
    });
    if (!groups.length) {
      focus.target = 0;
    } else {
      focus.groups.fill(0);
      groups.forEach((g) => (focus.groups[g] = 1));
      // The landscape is huge; aim the nudge at the paths on it instead.
      const aim = groups.filter((g) => g !== G.LAND);
      focus.c = mean((aim.length ? aim : groups).map((g) => centre[g]));
      focus.target = 1;
    }
    kick();
  }

  function syncLinked() {
    // Keyboard focus wins over a resting mouse, which wins over a tap.
    const el = focusEl || hoverEl || tapEl;
    if (el === shownEl) return;
    if (shownEl) shownEl.classList.remove("is-linked");
    shownEl = el;
    if (el) el.classList.add("is-linked");
    setFocus(el ? el.getAttribute("data-graph") : "");
  }

  const linkedFrom = (node) => (node && node.closest ? node.closest("[data-graph]") : null);

  const keyboardFocus = (el) => {
    try {
      return el.matches(":focus-visible");
    } catch (err) {
      return true;
    }
  };

  document.addEventListener("pointerover", (e) => {
    if (e.pointerType !== "mouse") return;
    hoverEl = linkedFrom(e.target);
    syncLinked();
  });

  document.addEventListener("pointerout", (e) => {
    if (e.pointerType !== "mouse" || e.relatedTarget) return;
    hoverEl = null;
    syncLinked();
  });

  document.addEventListener("focusin", (e) => {
    focusEl = keyboardFocus(e.target) ? linkedFrom(e.target) : null;
    syncLinked();
  });

  document.addEventListener("focusout", (e) => {
    if (!e.relatedTarget) {
      focusEl = null;
      syncLinked();
    }
  });

  document.addEventListener("click", (e) => {
    if (fineMQ.matches) return;
    const el = linkedFrom(e.target);
    tapEl = el && el !== tapEl ? el : null;
    tapY = window.scrollY;
    syncLinked();
  });

  /* Pointer layer (desktop) ------------------------------------------- */

  window.addEventListener(
    "pointermove",
    (e) => {
      if (e.pointerType !== "mouse") return;
      ptr.tx = e.clientX;
      ptr.ty = e.clientY;
      if (!ptr.on) {
        ptr.x = ptr.tx;
        ptr.y = ptr.ty;
      }
      ptr.on = true;
      ptr.moved = performance.now();
    },
    { passive: true }
  );

  document.documentElement.addEventListener("pointerleave", () => {
    ptr.on = false;
    ptr.moved = performance.now();
  });

  /* Graph Theory tag: recolor the embedding graph ---------------------- */

  let note = null;
  let noteTimer = 0;

  function announce(text) {
    if (!note) {
      note = document.createElement("p");
      note.className = "scene-note";
      note.setAttribute("role", "status");
      document.body.appendChild(note);
    }
    note.textContent = text;
    note.classList.add("is-shown");
    window.clearTimeout(noteTimer);
    noteTimer = window.setTimeout(() => note.classList.remove("is-shown"), 5200);
  }

  /* Wiring ------------------------------------------------------------- */

  try {
    setupGL();
  } catch (e) {
    root.classList.remove("has-scene");
    return;
  }

  (function () {
    const li = document.querySelector('.tag[data-graph~="graph"]');
    if (!li) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "tag tag-btn";
    btn.setAttribute("data-graph", li.getAttribute("data-graph"));
    btn.title = "Recolor the background graph";
    btn.innerHTML = li.innerHTML + '<span class="sr-only">: recolor the background graph</span>';
    li.removeAttribute("data-graph");
    li.className = "tag-item";
    li.textContent = "";
    li.appendChild(btn);
    btn.addEventListener("click", startRecolor);
  })();

  let measureQueued = false;
  function queueMeasure() {
    if (measureQueued) return;
    measureQueued = true;
    requestAnimationFrame(() => {
      measureQueued = false;
      measure();
      kick();
    });
  }

  window.addEventListener(
    "scroll",
    () => {
      if (tapEl && Math.abs(window.scrollY - tapY) > vh * 0.6) {
        tapEl = null;
        syncLinked();
      }
      kick();
    },
    { passive: true }
  );

  window.addEventListener("resize", queueMeasure, { passive: true });

  if ("ResizeObserver" in window) {
    new ResizeObserver(resize).observe(canvas);
    new ResizeObserver(queueMeasure).observe(document.body);
  } else {
    window.addEventListener("resize", resize, { passive: true });
  }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      cancelAnimationFrame(raf);
      raf = 0;
    } else {
      kick();
    }
  });

  canvas.addEventListener("webglcontextlost", (e) => {
    e.preventDefault();
    lost = true;
    cancelAnimationFrame(raf);
    raf = 0;
  });

  canvas.addEventListener("webglcontextrestored", () => {
    lost = false;
    setupGL();
    resize();
  });

  new MutationObserver(syncTheme).observe(root, { attributes: true, attributeFilter: ["data-theme"] });
  darkMQ.addEventListener("change", syncTheme);
  reduceMQ.addEventListener("change", () => setStatic(reduceMQ.matches || quality >= 2));

  syncTheme();
  resize();
  viewAt(window.scrollY, cam);
  if (!staticMode) cam.d *= 1.2; // a short settle on arrival
  kick();
})();
