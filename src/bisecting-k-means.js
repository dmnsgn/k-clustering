/** @module bisectingKMeans */

import { Clustering } from "./clustering.js";
import { squaredEuclidean } from "./distance.js";
import { KMeans } from "./k-means.js";
import { gather, resize, weightOf } from "./utils.js";

/**
 * @typedef {object} BisectingKMeansOptions
 * @property {"cost" | "size"} [strategy="cost"] Split the cluster of highest
 *   cost or of largest size. scikit-learn's default.
 * @property {number} [trials=1] Bisections tried per split, keeping the one of
 *   lowest cost. scikit-learn's default.
 * @property {number} [maxIterations=300] Iterations limit of each bisection.
 *   See `KMeans`.
 * @property {number} [tolerance=1e-4] Tolerance of each bisection. See
 *   `KMeans`.
 */

/**
 * @typedef {object} BisectingKMeansFitOptions
 * @property {import("./clustering.js").Weights} [weights] Weight of each point.
 *   All 1 when omitted.
 */

/**
 * Bisecting k-means clustering: start from one cluster and split one cluster in
 * two with k-means until there are `k`.
 *
 * Deterministic cluster hierarchy, less sensitive to initialization and faster
 * than k-means with many clusters. Splits aren't refined afterwards: pass
 * `centroids` to `KMeans` for that.
 *
 * Stops early with clusters of size 0 when points hold fewer than `k` distinct
 * values.
 *
 * @augments {Clustering}
 * @see Steinbach, M., Karypis, G., & Kumar, V. (2000). A comparison of document clustering techniques. KDD Workshop on Text Mining.
 */
export class BisectingKMeans extends Clustering {
  /** @type {"cost" | "size"} */
  strategy;
  /** @type {number} */
  trials;

  #kMeans;
  #costs = new Float64Array(0);
  #subset = new Float64Array(0);
  #subsetWeights = new Float64Array(0);
  // Point index of each subset point.
  #members = new Uint32Array(0);
  #bestLabels = new Int32Array(0);
  #bestCentroids = new Float64Array(0);

  /**
   * @param {import("./clustering.js").ClusteringOptions &
   *   import("./clustering.js").InitializedOptions &
   *   BisectingKMeansOptions} [options]
   */
  constructor({
    strategy = "cost",
    trials = 1,
    maxIterations = 300,
    tolerance = 1e-4,
    ...options
  } = {}) {
    super({ ...options, distance: squaredEuclidean });
    this.strategy = strategy;
    this.trials = trials;
    this.#kMeans = new KMeans({
      ...options,
      k: 2,
      maxIterations,
      tolerance,
    });
  }

  /**
   * @param {import("./clustering.js").Points} points Flat array of point
   *   components, `dimension` per point.
   * @param {BisectingKMeansFitOptions} [options]
   * @returns {this}
   */
  fit(points, { weights } = {}) {
    this._prepare(points);
    const { k, dimension, count, labels, centroids, sizes } = this;
    const kMeans = this.#kMeans;
    kMeans.dimension = dimension;
    kMeans.random = this.random;
    kMeans.init = this.init;

    const costs = (this.#costs = resize(this.#costs, k));
    this.#subset = resize(this.#subset, count * dimension);
    this.#subsetWeights = resize(this.#subsetWeights, count);
    this.#members = resize(this.#members, count);
    this.#bestLabels = resize(this.#bestLabels, count);
    this.#bestCentroids = resize(this.#bestCentroids, 2 * dimension);

    labels.fill(0);
    centroids.fill(0);
    sizes.fill(0);
    costs.fill(0);
    for (let i = 0; i < count; i++) {
      const w = weightOf(weights, i);
      sizes[0] += w;
      for (let a = 0; a < dimension; a++) {
        centroids[a] += w * points[i * dimension + a];
      }
    }
    for (let a = 0; a < dimension; a++) centroids[a] /= sizes[0];
    costs[0] = this._tally(points, weights);

    let clusters = 1;
    for (; clusters < k; clusters++) {
      const target = this.#selectCluster(clusters);
      if (target === -1) break;
      this.#bisect(points, weights, target, clusters);
    }
    // Remaining clusters are empty, on the first centroid.
    for (let c = clusters; c < k; c++) {
      centroids.copyWithin(c * dimension, 0, dimension);
    }

    this.cost = this._tally(points, weights);
    this.converged = true;
    return this;
  }

  // Splittable cluster of highest cost or size, -1 if none: a cluster of
  // identical points has a cost of 0.
  #selectCluster(clusters) {
    const costs = this.#costs;
    const values = this.strategy === "size" ? this.sizes : costs;
    let target = -1;
    for (let c = 0; c < clusters; c++) {
      if (costs[c] <= 0) continue;
      if (target === -1 || values[c] > values[target]) target = c;
    }
    return target;
  }

  #bisect(points, weights, target, added) {
    const { dimension, count, labels, centroids, sizes } = this;
    const kMeans = this.#kMeans;
    const subset = this.#subset;
    const subsetWeights = this.#subsetWeights;
    const members = this.#members;
    const bestLabels = this.#bestLabels;
    const bestCentroids = this.#bestCentroids;

    let size = 0;
    for (let i = 0; i < count; i++) {
      if (labels[i] !== target) continue;
      members[size] = i;
      subsetWeights[size++] = weightOf(weights, i);
    }
    gather(points, members, size, dimension, subset);

    let bestCost = Infinity;
    for (let trial = 0; trial < this.trials; trial++) {
      kMeans.fit(subset.subarray(0, size * dimension), {
        weights: weights ? subsetWeights.subarray(0, size) : undefined,
      });
      this.iterations += kMeans.iterations;
      if (kMeans.cost < bestCost) {
        bestCost = kMeans.cost;
        bestLabels.set(kMeans.labels);
        bestCentroids.set(kMeans.centroids);
      }
    }

    const costs = this.#costs;
    costs[target] = costs[added] = 0;
    sizes[target] = sizes[added] = 0;
    centroids.set(bestCentroids.subarray(0, dimension), target * dimension);
    centroids.set(bestCentroids.subarray(dimension), added * dimension);
    for (let s = 0; s < size; s++) {
      const cluster = bestLabels[s] ? added : target;
      const i = members[s];
      labels[i] = cluster;
      sizes[cluster] += subsetWeights[s];
      costs[cluster] +=
        subsetWeights[s] *
        squaredEuclidean(
          points,
          i * dimension,
          centroids,
          cluster * dimension,
          dimension,
        );
    }
  }
}
