/** @module miniBatchKMeans */

import { Clustering } from "./clustering.js";
import { squaredEuclidean } from "./distance.js";
import {
  gather,
  nearestCentroid,
  resize,
  scaledTolerance,
  weightOf,
} from "./utils.js";

/**
 * @typedef {object} MiniBatchKMeansOptions
 * @property {number} [batchSize=1024] Points sampled, with replacement, per
 *   batch. scikit-learn's default.
 * @property {number} [maxIterations=100] Passes over the points: at most
 *   `maxIterations * count / batchSize` batches. scikit-learn's default.
 * @property {number} [maxNoImprovement=10] Stop after this many batches without
 *   improving the smoothed batch cost. scikit-learn's default. 0 disables it.
 * @property {number} [initSize] Points sampled, with replacement, to initialize
 *   the centroids on. Defaults to `3 * batchSize`, or `3 * k` when that is
 *   under `k`, as in scikit-learn: on all points, the initialization costs at
 *   least O(k n).
 * @property {number} [tolerance=0] Stop when the sum of squared centroid shifts
 *   of a batch falls under `tolerance` times the mean variance of the axes. 0
 *   disables it, as scikit-learn's default: a single batch shift is noisy.
 */

/**
 * Mini-batch k-means clustering: k-means updated from small random batches,
 * each centroid moving towards its points with a decreasing learning rate.
 *
 * Orders of magnitude faster than k-means on large inputs, for a slightly
 * higher cost.
 *
 * @augments {Clustering}
 * @see Sculley, D. (2010). Web-scale k-means clustering. Proceedings of the 19th International Conference on World Wide Web, 1177–1178. https://doi.org/10.1145/1772690.1772862
 * @see https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/cluster/_kmeans.py (passes, early stopping, initialization subset)
 */
export class MiniBatchKMeans extends Clustering {
  /** @type {number} */
  batchSize;
  /** @type {number} */
  maxIterations;
  /** @type {number} */
  maxNoImprovement;
  /** @type {number | undefined} */
  initSize;
  /** @type {number} */
  tolerance;

  /**
   * Batches processed by the last fit. `iterations` counts passes over the
   * points.
   *
   * @type {number}
   */
  steps = 0;

  // Weight of the points that moved each centroid so far.
  #counts = new Float64Array(0);
  #batch = new Uint32Array(0);
  #batchLabels = new Int32Array(0);
  #previous = new Float64Array(0);
  #nearestDistance = new Float64Array(1);
  #initMembers = new Uint32Array(0);
  #initPoints = new Float64Array(0);
  #initWeights = new Float64Array(0);

  /**
   * @param {import("./clustering.js").ClusteringOptions &
   *   import("./clustering.js").InitializedOptions &
   *   MiniBatchKMeansOptions} [options]
   */
  constructor({
    batchSize = 1024,
    maxIterations = 100,
    maxNoImprovement = 10,
    initSize,
    tolerance = 0,
    ...options
  } = {}) {
    super({ ...options, distance: squaredEuclidean });
    this.batchSize = batchSize;
    this.maxIterations = maxIterations;
    this.maxNoImprovement = maxNoImprovement;
    this.initSize = initSize;
    this.tolerance = tolerance;
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
    const batchSize = Math.min(this.batchSize, count);
    const maxSteps = Math.floor((this.maxIterations * count) / batchSize);
    this.#counts = resize(this.#counts, k);
    this.#batch = resize(this.#batch, batchSize);
    this.#batchLabels = resize(this.#batchLabels, batchSize);
    this.#previous = resize(this.#previous, k * dimension);

    this.#initCentroids(points, weights, centroids);
    const tolerance = scaledTolerance(points, count, dimension, this.tolerance);
    // Smoothing over about half a pass, as scikit-learn.
    const alpha = Math.min((2 * batchSize) / (count + 1), 1);
    let smoothedCost = 0;
    let bestCost = Infinity;
    let noImprovement = 0;
    this.#counts.fill(0);

    let step = 0;
    while (step < maxSteps) {
      this.#previous.set(this.centroids);
      const cost = this.#assignBatch(points, weights, batchSize);
      this.#updateBatch(points, weights, batchSize);
      step++;

      if (tolerance && this.#batchShift() <= tolerance) {
        this.converged = true;
        break;
      }

      // The first batch measures the initialization, as in scikit-learn.
      if (step === 1) continue;
      smoothedCost =
        step === 2 ? cost : smoothedCost * (1 - alpha) + cost * alpha;
      if (smoothedCost < bestCost) {
        bestCost = smoothedCost;
        noImprovement = 0;
      } else if (
        this.maxNoImprovement &&
        ++noImprovement >= this.maxNoImprovement
      ) {
        this.converged = true;
        break;
      }
    }
    this.steps = step;
    this.iterations = Math.ceil((step * batchSize) / count);

    this.assign(points, this.labels);
    this.cost = this._tally(points, weights);
    return this;
  }

  // k-means++ on all points costs more than the batches on large inputs.
  #initCentroids(points, weights, centroids) {
    const { k, dimension, count, random } = this;
    let size = this.initSize ?? 3 * this.batchSize;
    if (this.initSize === undefined && size < k) size = 3 * k;
    if (size < k) {
      throw new RangeError(`initSize must be at least k (${k}).`);
    }
    if (centroids || size >= count) {
      this._initCentroids(points, { weights, centroids }, squaredEuclidean);
      return;
    }

    const members = (this.#initMembers = resize(this.#initMembers, size));
    for (let s = 0; s < size; s++) members[s] = Math.floor(random() * count);
    this.#initPoints = resize(this.#initPoints, size * dimension);
    const subset = gather(points, members, size, dimension, this.#initPoints);
    let subsetWeights;
    if (weights) {
      subsetWeights = this.#initWeights = resize(this.#initWeights, size);
      for (let s = 0; s < size; s++) subsetWeights[s] = weights[members[s]];
    }

    const indices = this._indices;
    this._initIndices(subset, indices, {
      weights: subsetWeights,
      distance: squaredEuclidean,
    });
    for (let c = 0; c < k; c++) indices[c] = members[indices[c]];
    this._setCentroids(points, indices);
  }

  // Sample a batch and label it. Labels are cached before any update: the
  // paper's algorithm 1. Returns the batch cost.
  #assignBatch(points, weights, batchSize) {
    const { k, dimension, count, random, centroids } = this;
    const batch = this.#batch;
    const batchLabels = this.#batchLabels;
    const nearestDistance = this.#nearestDistance;
    let cost = 0;
    for (let b = 0; b < batchSize; b++) {
      batch[b] = Math.floor(random() * count);
      batchLabels[b] = nearestCentroid(
        points,
        batch[b] * dimension,
        centroids,
        k,
        dimension,
        squaredEuclidean,
        nearestDistance,
      );
      cost += weightOf(weights, batch[b]) * nearestDistance[0];
    }
    return cost;
  }

  // Move each centroid towards its batch points, with a learning rate of the
  // inverse of its total weight so far.
  #updateBatch(points, weights, batchSize) {
    const { dimension, centroids } = this;
    const counts = this.#counts;
    for (let b = 0; b < batchSize; b++) {
      const i = this.#batch[b];
      const c = this.#batchLabels[b];
      const w = weightOf(weights, i);
      if (w <= 0) continue;
      counts[c] += w;
      const rate = w / counts[c];
      for (let a = 0; a < dimension; a++) {
        centroids[c * dimension + a] +=
          rate * (points[i * dimension + a] - centroids[c * dimension + a]);
      }
    }
  }

  // Sum of squared centroid shifts of the last batch.
  #batchShift() {
    const { k, dimension, centroids } = this;
    let shift = 0;
    for (let c = 0; c < k; c++) {
      shift += squaredEuclidean(
        this.#previous,
        c * dimension,
        centroids,
        c * dimension,
        dimension,
      );
    }
    return shift;
  }
}
