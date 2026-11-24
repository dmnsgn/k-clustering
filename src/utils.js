/** @module utils */

import reserve from "typed-array-reserve";

/**
 * Typed array of exactly `length` elements, reusing the storage of `array` when
 * large enough. A new view is only created when the length changes.
 *
 * @template {Float64Array | Int32Array | Uint32Array} T
 * @param {T} array
 * @param {number} length
 * @returns {T}
 */
export function resize(array, length) {
  if (array.length === length) return array;
  const Constructor = /** @type {function(new: T, ArrayBufferLike)} */ (
    array.constructor
  );
  const storage = new Constructor(array.buffer);
  return reserve(storage, length).subarray(0, length);
}

/**
 * Copy the points at `indices` into `out`, contiguously.
 *
 * @param {ArrayLike<number>} points
 * @param {ArrayLike<number>} indices
 * @param {number} count Number of indices to copy.
 * @param {number} dimension
 * @param {Float64Array} out Receives `count * dimension` components.
 * @returns {Float64Array}
 */
export function gather(points, indices, count, dimension, out) {
  for (let s = 0; s < count; s++) {
    for (let a = 0; a < dimension; a++) {
      out[s * dimension + a] = points[indices[s] * dimension + a];
    }
  }
  return out;
}

/**
 * Index of the nearest of `k` centroids. Ties go to the lowest index.
 *
 * @param {ArrayLike<number>} points
 * @param {number} offset Index of the first component of the point.
 * @param {Float64Array} centroids
 * @param {number} k
 * @param {number} dimension
 * @param {import("./distance.js").Distance} distance
 * @param {Float64Array} [out] Receives the distance to the nearest centroid at
 *   index 0.
 * @returns {number}
 */
export function nearestCentroid(
  points,
  offset,
  centroids,
  k,
  dimension,
  distance,
  out,
) {
  let min = Infinity;
  let label = 0;
  for (let c = 0; c < k; c++) {
    const d = distance(points, offset, centroids, c * dimension, dimension);
    if (d < min) {
      min = d;
      label = c;
    }
  }
  if (out) out[0] = min;
  return label;
}

/**
 * Mean of the variances of each axis.
 *
 * @param {ArrayLike<number>} points
 * @param {number} count
 * @param {number} dimension
 * @returns {number}
 */
export function meanVariance(points, count, dimension) {
  let total = 0;
  for (let a = 0; a < dimension; a++) {
    let mean = 0;
    for (let i = 0; i < count; i++) mean += points[i * dimension + a];
    mean /= count;
    let variance = 0;
    for (let i = 0; i < count; i++) {
      variance += (points[i * dimension + a] - mean) ** 2;
    }
    total += variance / count;
  }
  return total / dimension;
}

/**
 * Absolute tolerance on the sum of squared centroid shifts: `tolerance` times
 * the mean variance of the axes, as in scikit-learn, so that it doesn't depend
 * on the scale of the points. 0 when `tolerance` is 0.
 *
 * @param {ArrayLike<number>} points
 * @param {number} count
 * @param {number} dimension
 * @param {number} tolerance
 * @returns {number}
 */
export function scaledTolerance(points, count, dimension, tolerance) {
  return tolerance ? tolerance * meanVariance(points, count, dimension) : 0;
}

/**
 * Normalize points to unit length. Zero vectors are kept.
 *
 * @param {Float64Array} points
 * @param {number} count
 * @param {number} dimension
 */
export function normalize(points, count, dimension) {
  for (let i = 0; i < count; i++) {
    const offset = i * dimension;
    let norm = 0;
    for (let a = 0; a < dimension; a++) norm += points[offset + a] ** 2;
    if (!norm) continue;
    norm = Math.sqrt(norm);
    for (let a = 0; a < dimension; a++) points[offset + a] /= norm;
  }
}

/**
 * Move each empty cluster to the point farthest from its centroid, as in
 * scikit-learn. Points are taken once, farthest first.
 *
 * @param {Float64Array} sizes Cluster weights: 0 when empty.
 * @param {Float64Array} distances Distance of each point to its centroid. Moved
 *   points are set to 0.
 * @param {number} count
 * @param {function(number, number): void} move Called with a point index and
 *   the empty cluster.
 */
export function relocateEmptyClusters(sizes, distances, count, move) {
  for (let c = 0; c < sizes.length; c++) {
    if (sizes[c] > 0) continue;
    let farthest = 0;
    for (let i = 1; i < count; i++) {
      if (distances[i] > distances[farthest]) farthest = i;
    }
    // Every point sits on a centroid: there are fewer distinct points than k.
    if (distances[farthest] <= 0) return;
    distances[farthest] = 0;
    move(farthest, c);
  }
}

/**
 * Weight of a point: 1 without weights.
 *
 * @param {ArrayLike<number> | undefined} weights
 * @param {number} i
 * @returns {number}
 */
export const weightOf = (weights, i) => (weights ? weights[i] : 1);
