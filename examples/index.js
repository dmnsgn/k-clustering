import canvasContext from "canvas-context";
import { create, fromHex, fromOklch, toCSSRGB, toOklch } from "pex-color";
import random from "pex-random";
import { Pane } from "tweakpane";

import { KMedians, KCenter, FuzzyCMeans } from "k-clustering";

import { ALGORITHMS } from "./algorithms.js";

const CONFIG = {
  distribution: "noisy blobs",
  count: 2000,
  k: 8,
  seed: 1,
  algorithm: "hamerly",
};

const TAU = Math.PI * 2;
const DIMENSION = 2;
// Background Voronoi cells resolution, in CSS pixels.
const CELL_SIZE = 4;

// Golden angle hue steps, in turns, from the accent hue: consecutive clusters
// stay apart for any k. Pale opaque cells keep the darker points readable.
// Lightness and chroma are in sRGB gamut for every hue.
const GOLDEN_ANGLE = (3 - Math.sqrt(5)) / 2;
const accentHue = toOklch(
  fromHex(
    create(),
    getComputedStyle(document.documentElement)
      .getPropertyValue("--color-accent")
      .trim(),
  ),
)[2];
let palette = [];
const createPalette = (k) =>
  Array.from({ length: k }, (_, c) => {
    const hue = (accentHue + c * GOLDEN_ANGLE) % 1;
    const color = fromOklch(create(), 0.55, 0.09, hue);
    return {
      color,
      fill: toCSSRGB(color),
      cell: toCSSRGB(fromOklch(create(), 0.9, 0.04, hue)),
    };
  });

const gaussian = (rng) =>
  Math.sqrt(-2 * Math.log(1 - rng.float())) * Math.cos(TAU * rng.float());

// Points in [-1, 1]², centered on the origin for spherical k-means.
const DISTRIBUTIONS = {
  "noisy blobs": (rng, i, centers) => {
    if (rng.float() < 0.04) return [rng.float(-1, 1), rng.float(-1, 1)];
    const c = Math.floor(rng.float() ** 2 * centers.length);
    const [x, y] = centers[c];
    const spread = 0.04 + 0.03 * c;
    return [x + gaussian(rng) * spread, y + gaussian(rng) * spread];
  },
  blobs: (rng, i, centers) => {
    const [x, y] = centers[i % centers.length];
    return [x + gaussian(rng) * 0.12, y + gaussian(rng) * 0.12];
  },
  stretched: (rng, i, centers) => {
    const [x, y] = centers[i % centers.length];
    const u = gaussian(rng) * 0.3;
    const v = gaussian(rng) * 0.05;
    return [x + u * 0.8 - v * 0.6, y + u * 0.6 + v * 0.8];
  },
  rings: (rng, i) => {
    const angle = rng.float(TAU);
    const radius = (i % 2 ? 0.8 : 0.35) + gaussian(rng) * 0.04;
    return [Math.cos(angle) * radius, Math.sin(angle) * radius];
  },
  uniform: (rng) => [rng.float(-1, 1), rng.float(-1, 1)],
};

function createPoints() {
  const rng = random.create(CONFIG.seed);
  const centers = Array.from({ length: 6 }, () => [
    rng.float(-0.7, 0.7),
    rng.float(-0.7, 0.7),
  ]);
  const generate = DISTRIBUTIONS[CONFIG.distribution];
  const points = new Float64Array(CONFIG.count * DIMENSION);
  for (let i = 0; i < CONFIG.count; i++) {
    points.set(generate(rng, i, centers), i * DIMENSION);
  }
  return points;
}

const main = document.querySelector("main");
const grid = document.createElement("div");
grid.className = "Panels";
main.append(grid);

const panels = Object.keys(ALGORITHMS).map((name) => {
  const figure = document.createElement("figure");
  const { canvas, context } = canvasContext("2d");
  figure.className = "Panel";
  canvas.className = "Panel-canvas";
  const caption = document.createElement("figcaption");
  caption.className = "Panel-caption";
  figure.append(canvas, caption);
  grid.append(figure);
  const panel = { name, canvas, context, caption, model: null };
  panel.worker = new Worker(new URL("worker.js", import.meta.url));
  panel.worker.addEventListener("message", ({ data }) => show(panel, data));
  return panel;
});

// Fuzzy points blend the colors of their clusters by membership.
const blend = create();
function pointFill(model, i) {
  if (!(model instanceof FuzzyCMeans)) return palette[model.labels[i]].fill;
  const { k, memberships } = model;
  blend.fill(0, 0, 3);
  for (let c = 0; c < k; c++) {
    for (let a = 0; a < 3; a++) {
      blend[a] += memberships[i * k + c] * palette[c].color[a];
    }
  }
  return toCSSRGB(blend);
}

// Part of a convex polygon where side() is positive.
function clip(polygon, side) {
  const clipped = [];
  for (let i = 0; i < polygon.length; i++) {
    const p = polygon[i];
    const q = polygon[(i + 1) % polygon.length];
    const sp = side(p);
    const sq = side(q);
    if (sp >= 0) clipped.push(p);
    if (sp >= 0 !== sq >= 0) {
      const t = sp / (sp - sq);
      clipped.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
    }
  }
  return clipped;
}

// Voronoi cell of a centroid within [-1.1, 1.1]²: the square clipped by the
// half-plane closer to the centroid than to each other one (Sutherland–Hodgman).
// Bisectors are lines for Euclidean distances, and for cosine distance with
// normalized centroids: there, the offset is 0.
function voronoiCell({ centroids, k }, c) {
  const ax = centroids[c * DIMENSION];
  const ay = centroids[c * DIMENSION + 1];
  let polygon = [
    [-1.1, -1.1],
    [1.1, -1.1],
    [1.1, 1.1],
    [-1.1, 1.1],
  ];
  for (let o = 0; o < k && polygon.length; o++) {
    if (o === c) continue;
    const bx = centroids[o * DIMENSION];
    const by = centroids[o * DIMENSION + 1];
    const nx = bx - ax;
    const ny = by - ay;
    // Coinciding centroids: ties go to the lowest index.
    if (!nx && !ny) {
      if (o < c) return [];
      continue;
    }
    const offset = (bx * bx + by * by - ax * ax - ay * ay) / 2;
    polygon = clip(polygon, ([x, y]) => offset - x * nx - y * ny);
  }
  return polygon;
}

// Manhattan bisectors are polylines: k-medians cells are sampled instead.
function drawSampledCells(context, model, size, fromScreen) {
  const step = CELL_SIZE * devicePixelRatio;
  const position = new Float64Array(DIMENSION);
  for (let y = 0; y < size; y += step) {
    for (let x = 0; x < size; x += step) {
      position[0] = fromScreen(x + step / 2);
      position[1] = fromScreen(y + step / 2);
      context.fillStyle = palette[model.predict(position)].cell;
      context.fillRect(x, y, step, step);
    }
  }
}

function drawCells(context, model, size, toScreen, fromScreen) {
  if (model instanceof KMedians) {
    drawSampledCells(context, model, size, fromScreen);
    return;
  }
  context.lineWidth = devicePixelRatio;
  context.strokeStyle = getComputedStyle(document.body).color;
  for (let c = 0; c < model.k; c++) {
    context.beginPath();
    for (const [x, y] of voronoiCell(model, c)) {
      context.lineTo(toScreen(x), toScreen(y));
    }
    context.closePath();
    context.fillStyle = palette[c].cell;
    context.fill();
    context.globalAlpha = 0.3;
    context.stroke();
    context.globalAlpha = 1;
  }
}

function draw({ canvas, context, model, points }) {
  const size = canvas.clientWidth * devicePixelRatio;
  canvas.width = canvas.height = size;
  if (!model) return;
  const toScreen = (value) => ((value + 1.1) / 2.2) * size;
  const fromScreen = (value) => (value / size) * 2.2 - 1.1;

  drawCells(context, model, size, toScreen, fromScreen);

  const { k } = model;
  // Smaller marks as points and clusters get denser.
  const radius =
    Math.max(0.75, 2.5 * Math.sqrt(800 / model.count)) * devicePixelRatio;
  const centroidRadius = Math.max(2, 14 / Math.sqrt(k)) * devicePixelRatio;
  for (let i = 0; i < model.count; i++) {
    context.fillStyle = pointFill(model, i);
    context.beginPath();
    context.arc(
      toScreen(points[i * DIMENSION]),
      toScreen(points[i * DIMENSION + 1]),
      radius,
      0,
      TAU,
    );
    context.fill();
  }

  context.lineWidth = 2 * devicePixelRatio;
  context.strokeStyle = getComputedStyle(document.body).color;
  for (let c = 0; c < k; c++) {
    const x = toScreen(model.centroids[c * DIMENSION]);
    const y = toScreen(model.centroids[c * DIMENSION + 1]);
    context.fillStyle = palette[c].fill;
    context.beginPath();
    context.arc(x, y, centroidRadius, 0, TAU);
    context.fill();
    context.stroke();
    // k-center covers every point within its radius.
    if (model instanceof KCenter) {
      context.beginPath();
      context.arc(x, y, (model.cost / 2.2) * size, 0, TAU);
      context.stroke();
    }
  }
}

let points = createPoints();
let runId = 0;

function show(panel, { id, state, time, daviesBouldin }) {
  if (id !== runId) return;
  const model = ALGORITHMS[panel.name].create(
    { dimension: DIMENSION, k: CONFIG.k },
    CONFIG,
  );
  panel.model = Object.assign(model, state);
  panel.caption.innerHTML = `<strong>${panel.name}</strong>
    <span class="Panel-stats">Davies–Bouldin ${daviesBouldin.toFixed(3)} · ${model.iterations} iterations · ${time.toFixed(1)} ms</span>`;
  draw({ ...panel, points });
}

function run() {
  const id = ++runId;
  points = createPoints();
  palette = createPalette(CONFIG.k);
  for (const panel of panels) {
    panel.model = null;
    draw({ ...panel, points });
    if (ALGORITHMS[panel.name].skip?.(CONFIG)) {
      panel.caption.innerHTML = `<strong>${panel.name}</strong>
        <span class="Panel-stats">skipped: too slow for ${CONFIG.count} points and k=${CONFIG.k}</span>`;
      continue;
    }
    panel.caption.innerHTML = `<strong>${panel.name}</strong><span class="Panel-stats">…</span>`;
    panel.worker.postMessage({
      id,
      name: panel.name,
      points,
      config: { ...CONFIG },
      dimension: DIMENSION,
    });
  }
}

function render() {
  for (const panel of panels) draw({ ...panel, points });
}

const pane = new Pane({ title: "k-clustering" });
pane.addBinding(CONFIG, "distribution", {
  options: Object.fromEntries(Object.keys(DISTRIBUTIONS).map((d) => [d, d])),
});
pane.addBinding(CONFIG, "count", { min: 100, max: 20_000, step: 100 });
pane.addBinding(CONFIG, "k", { min: 1, max: 256, step: 1 });
pane.addBinding(CONFIG, "seed", { min: 1, max: 100, step: 1 });
pane.addBinding(CONFIG, "algorithm", {
  label: "k-means",
  options: {
    Hamerly: "hamerly",
    Elkan: "elkan",
    Lloyd: "lloyd",
    "Lloyd + k-d tree": "kdTree",
  },
});
pane.on("change", run);

addEventListener("resize", render);
run();

if (import.meta.hot) {
  const { data } = import.meta.hot;
  data.dispose?.();
  data.dispose = () => {
    runId++;
    for (const panel of panels) panel.worker.terminate();
    removeEventListener("resize", render);
    grid.remove();
    pane.dispose();
  };
  import.meta.hot.accept(() => {});
}
