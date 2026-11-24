/** @module metrics */

import { euclidean, squaredEuclidean } from "./distance.js";
import { weightOf } from "./utils.js";

/**
 * @typedef {object} MetricOptions
 * @property {number} [dimension=3] Number of components per point.
 */

/**
 * @typedef {object} InertiaOptions
 * @property {ArrayLike<number>} [weights] Weight of each point. All 1 when omitted.
 */

/**
 * @typedef {object} SilhouetteOptions
 * @property {import("./distance.js").Distance} [distance=euclidean]
 * @property {Float64Array} [out] Receives the coefficient of each point.
 */

// Number of clusters: labels are expected in [0, k).
const clusterCount = (labels) => {
  let max = -1;
  for (let i = 0; i < labels.length; i++) max = Math.max(max, labels[i]);
  return max + 1;
};

// Point count and mean of each cluster.
function clusterMeans(points, labels, dimension) {
  const k = clusterCount(labels);
  const sizes = new Float64Array(k);
  const means = new Float64Array(k * dimension);
  for (let i = 0; i < labels.length; i++) {
    sizes[labels[i]]++;
    for (let a = 0; a < dimension; a++) {
      means[labels[i] * dimension + a] += points[i * dimension + a];
    }
  }
  for (let c = 0; c < k; c++) {
    for (let a = 0; a < dimension; a++) means[c * dimension + a] /= sizes[c];
  }
  return { k, sizes, means };
}

// Silhouette of a point of cluster `own`, from its sums of distances to the
// points of each cluster.
function pointSilhouette(sums, sizes, own) {
  if (sizes[own] <= 1) return 0;
  const a = sums[own] / (sizes[own] - 1);
  let b = Infinity;
  for (let c = 0; c < sizes.length; c++) {
    if (c !== own && sizes[c]) b = Math.min(b, sums[c] / sizes[c]);
  }
  return b === Infinity ? 0 : (b - a) / Math.max(a, b) || 0;
}

/**
 * Weighted sum of squared distances of the points to their centroid: the
 * k-means objective.
 *
 * @param {ArrayLike<number>} points Flat array of point components.
 * @param {ArrayLike<number>} centroids Flat array of centroid components.
 * @param {ArrayLike<number>} labels Cluster of each point.
 * @param {MetricOptions & InertiaOptions} [options]
 * @returns {number}
 */
export function inertia(
  points,
  centroids,
  labels,
  { dimension = 3, weights } = {},
) {
  let sum = 0;
  for (let i = 0; i < labels.length; i++) {
    sum +=
      weightOf(weights, i) *
      squaredEuclidean(
        points,
        i * dimension,
        centroids,
        labels[i] * dimension,
        dimension,
      );
  }
  return sum;
}

/**
 * Mean silhouette coefficient, between -1 and 1: how much closer each point is
 * to its own cluster than to the nearest other one. Higher is better. O(n²).
 *
 * Points alone in their cluster score 0.
 *
 * @param {ArrayLike<number>} points Flat array of point components.
 * @param {ArrayLike<number>} labels Cluster of each point.
 * @param {MetricOptions & SilhouetteOptions} [options]
 * @returns {number}
 * @see Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics, 20, 53–65. https://doi.org/10.1016/0377-0427(87)90125-7
 */
export function silhouette(
  points,
  labels,
  { dimension = 3, distance = euclidean, out } = {},
) {
  const count = labels.length;
  const k = clusterCount(labels);
  const sizes = new Float64Array(k);
  for (let i = 0; i < count; i++) sizes[labels[i]]++;
  const sums = new Float64Array(k);

  let total = 0;
  for (let i = 0; i < count; i++) {
    sums.fill(0);
    for (let j = 0; j < count; j++) {
      if (j !== i) {
        sums[labels[j]] += distance(
          points,
          i * dimension,
          points,
          j * dimension,
          dimension,
        );
      }
    }
    const score = pointSilhouette(sums, sizes, labels[i]);
    if (out) out[i] = score;
    total += score;
  }
  return total / count;
}

/**
 * Davies–Bouldin index: mean similarity of each cluster with its most similar
 * one, comparing their spread to their distance. Lower is better, 0 at best.
 *
 * @param {ArrayLike<number>} points Flat array of point components.
 * @param {ArrayLike<number>} labels Cluster of each point.
 * @param {MetricOptions} [options]
 * @returns {number}
 * @see Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-1(2), 224–227. https://doi.org/10.1109/TPAMI.1979.4766909
 */
export function daviesBouldin(points, labels, { dimension = 3 } = {}) {
  const { k, sizes, means } = clusterMeans(points, labels, dimension);
  const spreads = new Float64Array(k);
  for (let i = 0; i < labels.length; i++) {
    spreads[labels[i]] += euclidean(
      points,
      i * dimension,
      means,
      labels[i] * dimension,
      dimension,
    );
  }

  let total = 0;
  let clusters = 0;
  for (let c = 0; c < k; c++) {
    if (!sizes[c]) continue;
    spreads[c] /= sizes[c];
  }
  for (let c = 0; c < k; c++) {
    if (!sizes[c]) continue;
    clusters++;
    let max = 0;
    for (let j = 0; j < k; j++) {
      if (j === c || !sizes[j]) continue;
      const d = euclidean(
        means,
        c * dimension,
        means,
        j * dimension,
        dimension,
      );
      // Coinciding centroids are ignored, as in scikit-learn.
      if (d) max = Math.max(max, (spreads[c] + spreads[j]) / d);
    }
    total += max;
  }
  return total / clusters;
}

/**
 * Calinski–Harabasz index (variance ratio criterion): dispersion between
 * clusters over dispersion within clusters, scaled by their degrees of
 * freedom. Higher is better.
 *
 * @param {ArrayLike<number>} points Flat array of point components.
 * @param {ArrayLike<number>} labels Cluster of each point.
 * @param {MetricOptions} [options]
 * @returns {number}
 * @see Caliński, T., & Harabasz, J. (1974). A dendrite method for cluster analysis. Communications in Statistics, 3(1), 1–27. https://doi.org/10.1080/03610927408827101
 */
export function calinskiHarabasz(points, labels, { dimension = 3 } = {}) {
  const count = labels.length;
  const { k, sizes, means } = clusterMeans(points, labels, dimension);
  const mean = new Float64Array(dimension);
  for (let i = 0; i < count; i++) {
    for (let a = 0; a < dimension; a++)
      mean[a] += points[i * dimension + a] / count;
  }

  let between = 0;
  let within = 0;
  let clusters = 0;
  for (let c = 0; c < k; c++) {
    if (!sizes[c]) continue;
    clusters++;
    between +=
      sizes[c] * squaredEuclidean(means, c * dimension, mean, 0, dimension);
  }
  for (let i = 0; i < count; i++) {
    within += squaredEuclidean(
      points,
      i * dimension,
      means,
      labels[i] * dimension,
      dimension,
    );
  }
  // A perfect clustering, as in scikit-learn.
  return within
    ? (between * (count - clusters)) / (within * (clusters - 1))
    : 1;
}
