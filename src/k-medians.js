/** @module kMedians */

import { Clustering } from "./clustering.js";
import { manhattan } from "./distance.js";
import {
  relocateEmptyClusters,
  resize,
  scaledTolerance,
  weightOf,
} from "./utils.js";

/**
 * @typedef {object} KMediansOptions
 * @property {number} [maxIterations=300] As `KMeans`.
 * @property {number} [tolerance=0] Stop when the sum of squared centroid shifts
 *   falls under `tolerance` times the mean variance of the axes. 0 runs until
 *   no point changes cluster: medians move between point components, without
 *   the tail of vanishing shifts of means.
 * @property {import("./clustering.js").EmptyClusters} [emptyClusters="farthest"]
 */

/**
 * K-medians clustering: minimize the sum of Manhattan distances of the points
 * to their cluster median, taken on each axis.
 *
 * Less sensitive to outliers than k-means.
 *
 * @augments {Clustering}
 * @see Jain, A. K., & Dubes, R. C. (1988). Algorithms for clustering data, §3.3. Prentice Hall.
 * @see Bradley, P. S., Mangasarian, O. L., & Street, W. N. (1997). Clustering via concave minimization. Advances in Neural Information Processing Systems 9, 368–374.
 */
export class KMedians extends Clustering {
  /** @type {number} */
  maxIterations;
  /** @type {number} */
  tolerance;
  /** @type {import("./clustering.js").EmptyClusters} */
  emptyClusters;

  // Points sorted by cluster, from `#offsets[c]` to `#offsets[c + 1]`.
  #order = new Uint32Array(0);
  #offsets = new Uint32Array(0);

  /** @type {import("./clustering.js").Weights | undefined} */
  #weights;

  /**
   * @param {import("./clustering.js").ClusteringOptions &
   *   import("./clustering.js").InitializedOptions &
   *   KMediansOptions} [options]
   */
  constructor({
    maxIterations = 300,
    tolerance = 0,
    emptyClusters = "farthest",
    ...options
  } = {}) {
    super({ ...options, distance: manhattan });
    this.maxIterations = maxIterations;
    this.tolerance = tolerance;
    this.emptyClusters = emptyClusters;
  }

  /**
   * @param {import("./clustering.js").Points} points Flat array of point
   *   components, `dimension` per point.
   * @param {import("./clustering.js").FitOptions} [options]
   * @returns {this}
   */
  fit(points, { weights, centroids } = {}) {
    this._prepare(points);
    const { k, dimension, count } = this;
    this.#order = resize(this.#order, count);
    this.#offsets = resize(this.#offsets, k + 1);
    this.#weights = weights;

    this._initCentroids(points, { weights, centroids }, manhattan);
    this.labels.fill(-1);
    this._iterate(
      this.maxIterations,
      scaledTolerance(points, count, dimension, this.tolerance),
      () => this.#assign(points),
      () => {
        if (this.emptyClusters === "farthest" && this.sizes.includes(0)) {
          relocateEmptyClusters(this.sizes, this._distances, count, this.#move);
        }
        return this.#update(points);
      },
    );

    this.cost = this._tally(points, weights);
    return this;
  }

  #assign(points) {
    const { k, dimension, count, centroids, labels, sizes } = this;
    const distances = this._distances;
    sizes.fill(0);
    let changed = false;
    for (let i = 0; i < count; i++) {
      let label = 0;
      let min = Infinity;
      // Inlined for speed. See KMeans.
      for (let c = 0; c < k; c++) {
        const d = manhattan(
          points,
          i * dimension,
          centroids,
          c * dimension,
          dimension,
        );
        if (d < min) {
          min = d;
          label = c;
        }
      }
      if (labels[i] !== label) {
        labels[i] = label;
        changed = true;
      }
      distances[i] = min;
      sizes[label] += weightOf(this.#weights, i);
    }
    return changed;
  }

  #move = (i, cluster) => {
    const w = weightOf(this.#weights, i);
    this.sizes[this.labels[i]] -= w;
    this.sizes[cluster] = w;
    this.labels[i] = cluster;
  };

  // Move the centroids to their cluster median on each axis.
  #update(points) {
    const { k, dimension, count, centroids, labels, sizes } = this;
    const order = this.#order;
    const offsets = this.#offsets;

    // Counting sort keeps points in index order within a cluster.
    offsets.fill(0);
    for (let i = 0; i < count; i++) offsets[labels[i] + 1]++;
    for (let c = 0; c < k; c++) offsets[c + 1] += offsets[c];
    for (let i = 0; i < count; i++) order[offsets[labels[i]]++] = i;
    for (let c = k; c > 0; c--) offsets[c] = offsets[c - 1];
    offsets[0] = 0;

    let shiftTotal = 0;
    for (let c = 0; c < k; c++) {
      if (sizes[c] <= 0) continue;
      const members = order.subarray(offsets[c], offsets[c + 1]);
      let shift = 0;
      for (let a = 0; a < dimension; a++) {
        const median = this.#median(points, members, a, sizes[c]);
        shift += (median - centroids[c * dimension + a]) ** 2;
        centroids[c * dimension + a] = median;
      }
      shiftTotal += shift;
    }
    return shiftTotal;
  }

  // Weighted median: halfway between the two middle values when they split
  // the weight evenly, as numpy.median without weights.
  #median(points, members, axis, total) {
    const { dimension } = this;
    const value = (i) => points[i * dimension + axis];
    members.sort((i, j) => value(i) - value(j) || i - j);
    const half = total / 2;
    let cumulative = 0;
    for (let m = 0; m < members.length; m++) {
      cumulative += weightOf(this.#weights, members[m]);
      if (cumulative > half) return value(members[m]);
      if (cumulative === half) {
        return (value(members[m]) + value(members[m + 1] ?? members[m])) / 2;
      }
    }
    return value(members.at(-1));
  }
}
