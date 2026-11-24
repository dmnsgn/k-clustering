/** @module kMeans */

import { Clustering } from "./clustering.js";
import { cosine, euclidean, squaredEuclidean } from "./distance.js";
import {
  normalize,
  relocateEmptyClusters,
  resize,
  scaledTolerance,
  weightOf,
} from "./utils.js";

/**
 * Nearest neighbour structure over the centroids, such as a spatial-partition
 * `KdTree`.
 *
 * @typedef {object} CentroidIndex
 * @property {function(Float64Array): void} build Index the centroids.
 * @property {function(ArrayLike<number>, number, number, number[]): number[]} queryNearest Called with a position, `k`, a maximum distance and an output array. Returns the indices of the `k` nearest centroids.
 */

/**
 * @typedef {object} KMeansOptions
 * @property {"lloyd" | "elkan" | "hamerly"} [algorithm="hamerly"] Same clusters, different speed. Hamerly stores 2 bounds per point: fastest in low dimension. Elkan stores `k` bounds per point: fastest in high dimension with many clusters. Lloyd stores none and is the only one using `index`.
 * @property {number} [maxIterations=300] scikit-learn's default.
 * @property {number} [tolerance=1e-4] Stop when the sum of squared centroid shifts falls under `tolerance` times the mean variance of the axes. scikit-learn's default: cuts the tail of vanishing shifts before labels settle. 0 runs until no point changes cluster.
 * @property {import("./clustering.js").EmptyClusters} [emptyClusters="farthest"]
 * @property {boolean} [spherical=false] Cluster directions: points and centroids are normalized and compared by cosine distance.
 * @property {CentroidIndex} [index] Lloyd only. Rebuilt over the centroids at each iteration, with the same `dimension`: faster with hundreds of clusters. Equidistant centroids may break ties differently.
 */

/**
 * k-means clustering: minimize the sum of squared distances of the points to
 * their cluster mean.
 *
 * Best for compact, similarly sized clusters, color quantization and
 * centroidal Voronoi tessellations (Lloyd relaxation).
 *
 * @augments {Clustering}
 * @see Lloyd, S. P. (1982). Least squares quantization in PCM. IEEE Transactions on Information Theory, 28(2), 129–137. https://doi.org/10.1109/TIT.1982.1056489
 * @see MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, 1, 281–297.
 * @see Elkan, C. (2003). Using the triangle inequality to accelerate k-means. Proceedings of the Twentieth International Conference on Machine Learning, 147–153. https://dl.acm.org/doi/10.5555/3041838.3041857
 * @see Hamerly, G. (2010). Making k-means even faster. Proceedings of the 2010 SIAM International Conference on Data Mining, 130–140. https://doi.org/10.1137/1.9781611972801.12
 * @see Dhillon, I. S., & Modha, D. S. (2001). Concept decompositions for large sparse text data using clustering. Machine Learning, 42(1), 143–175. https://doi.org/10.1023/A:1007612920971 (spherical k-means)
 * @see https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/cluster/_kmeans.py (convergence, empty clusters)
 */
export class KMeans extends Clustering {
  /** @type {"lloyd" | "elkan" | "hamerly"} */
  algorithm;
  /** @type {number} */
  maxIterations;
  /** @type {number} */
  tolerance;
  /** @type {import("./clustering.js").EmptyClusters} */
  emptyClusters;
  /** @type {boolean} */
  spherical;
  /** @type {CentroidIndex | undefined} */
  index;

  // Weighted sums of the points of each cluster: the next centroids.
  #sums = new Float64Array(0);
  #shifts = new Float64Array(0);
  #normalized = new Float64Array(0);
  // Triangle inequality bounds: distance to the assigned centroid (upper), and
  // to each other centroid (Elkan) or the second nearest (Hamerly).
  #upper = new Float64Array(0);
  #lower = new Float64Array(0);
  #halfDistances = new Float64Array(0);
  #nearestHalf = new Float64Array(0);
  #position = new Float64Array(0);
  /** @type {number[]} */
  #nearest = [];

  /** @type {import("./clustering.js").Points} */
  #points = [];
  /** @type {import("./clustering.js").Weights | undefined} */
  #weights;

  /**
   * @param {import("./clustering.js").ClusteringOptions & import("./clustering.js").InitializedOptions & KMeansOptions} [options]
   */
  constructor({
    algorithm = "hamerly",
    maxIterations = 300,
    tolerance = 1e-4,
    emptyClusters = "farthest",
    spherical = false,
    index,
    ...options
  } = {}) {
    super({ ...options, distance: spherical ? cosine : squaredEuclidean });
    this.algorithm = algorithm;
    this.maxIterations = maxIterations;
    this.tolerance = tolerance;
    this.emptyClusters = emptyClusters;
    this.spherical = spherical;
    this.index = index;
  }

  /**
   * @param {import("./clustering.js").Points} points Flat array of point
   *   components, `dimension` per point.
   * @param {import("./clustering.js").FitOptions} [options]
   * @returns {this}
   */
  fit(points, { weights, centroids } = {}) {
    this._prepare(points);
    const { k, dimension, count, algorithm } = this;
    this.#sums = resize(this.#sums, k * dimension);
    this.#shifts = resize(this.#shifts, k);
    this.#position = resize(this.#position, dimension);

    if (this.spherical) {
      this.#normalized = resize(this.#normalized, count * dimension);
      this.#normalized.set(points);
      normalize(this.#normalized, count, dimension);
      points = this.#normalized;
    }
    this.#points = points;
    this.#weights = weights;

    this._initCentroids(points, { weights, centroids }, squaredEuclidean);
    if (this.spherical) normalize(this.centroids, k, dimension);

    this.labels.fill(-1);

    if (algorithm !== "lloyd") {
      this.#halfDistances = resize(this.#halfDistances, k * k);
      this.#nearestHalf = resize(this.#nearestHalf, k);
      this.#upper = resize(this.#upper, count);
      this.#lower = resize(
        this.#lower,
        algorithm === "elkan" ? count * k : count,
      );
      this.#updateCentroidDistances();
      if (algorithm === "elkan") this.#initElkan();
      else this.#initHamerly();
    }

    this._iterate(
      this.maxIterations,
      scaledTolerance(points, count, dimension, this.tolerance),
      (accumulate) => this.#assign(accumulate),
      () => this.#update(),
    );
    this.cost = this._tally(points, weights);
    return this;
  }

  // Accumulate sums the points of each cluster. Returns whether a label changed.
  #assign(accumulate) {
    if (accumulate) {
      this.#sums.fill(0);
      this.sizes.fill(0);
    }
    if (this.algorithm === "elkan") return this.#assignElkan(accumulate);
    if (this.algorithm === "hamerly") return this.#assignHamerly(accumulate);
    return this.#assignLloyd(accumulate);
  }

  #accumulate(i, label) {
    const { dimension } = this;
    const points = this.#points;
    const sums = this.#sums;
    const w = weightOf(this.#weights, i);
    this.sizes[label] += w;
    for (let a = 0; a < dimension; a++) {
      sums[label * dimension + a] += w * points[i * dimension + a];
    }
  }

  #assignLloyd(accumulate) {
    const { count, labels, centroids, index } = this;
    index?.build(centroids);
    let changed = false;
    for (let i = 0; i < count; i++) {
      const label = this.#nearestLloyd(i);
      if (labels[i] !== label) {
        labels[i] = label;
        changed = true;
      }
      if (accumulate) this.#accumulate(i, label);
    }
    return changed;
  }

  // Nearest centroid of point i, its squared distance into `_distances`.
  // Not utils.nearestCentroid: its distance call is shared by all clusterings,
  // which makes it about 3 times slower.
  #nearestLloyd(i) {
    const { k, dimension, centroids, index } = this;
    const points = this.#points;
    const offset = i * dimension;
    let label = 0;
    let min = Infinity;
    if (index) {
      const position = this.#position;
      for (let a = 0; a < dimension; a++) position[a] = points[offset + a];
      index.queryNearest(position, 1, Infinity, this.#nearest);
      label = this.#nearest[0];
      min = squaredEuclidean(
        points,
        offset,
        centroids,
        label * dimension,
        dimension,
      );
    } else {
      for (let c = 0; c < k; c++) {
        const d = squaredEuclidean(
          points,
          offset,
          centroids,
          c * dimension,
          dimension,
        );
        if (d < min) {
          min = d;
          label = c;
        }
      }
    }
    this._distances[i] = min;
    return label;
  }

  #updateCentroidDistances() {
    const { k, dimension, centroids } = this;
    const half = this.#halfDistances;
    const nearestHalf = this.#nearestHalf;
    for (let c = 0; c < k; c++) {
      half[c * k + c] = 0;
      for (let j = c + 1; j < k; j++) {
        const d =
          euclidean(
            centroids,
            c * dimension,
            centroids,
            j * dimension,
            dimension,
          ) / 2;
        half[c * k + j] = d;
        half[j * k + c] = d;
      }
    }
    for (let c = 0; c < k; c++) {
      let min = Infinity;
      for (let j = 0; j < k; j++) {
        if (j !== c && half[c * k + j] < min) min = half[c * k + j];
      }
      nearestHalf[c] = min;
    }
  }

  // A centroid is skipped when the point's distance to its own centroid is
  // under half of their distance (Elkan lemma 1).
  #initElkan() {
    const { k, dimension, count, centroids, labels } = this;
    const points = this.#points;
    const upper = this.#upper;
    const lower = this.#lower;
    const half = this.#halfDistances;
    lower.fill(0);
    for (let i = 0; i < count; i++) {
      const offset = i * dimension;
      let label = 0;
      let min = euclidean(points, offset, centroids, 0, dimension);
      lower[i * k] = min;
      for (let j = 1; j < k; j++) {
        if (min > half[label * k + j]) {
          const d = euclidean(
            points,
            offset,
            centroids,
            j * dimension,
            dimension,
          );
          lower[i * k + j] = d;
          if (d < min) {
            min = d;
            label = j;
          }
        }
      }
      labels[i] = label;
      upper[i] = min;
    }
  }

  #assignElkan(accumulate) {
    const { k, dimension, count, centroids, labels } = this;
    const points = this.#points;
    const upperBounds = this.#upper;
    const lower = this.#lower;
    const half = this.#halfDistances;

    let changed = false;
    for (let i = 0; i < count; i++) {
      const offset = i * dimension;
      const previous = labels[i];
      let label = previous;
      let upper = upperBounds[i];
      let tight = false;
      if (upper > this.#nearestHalf[label]) {
        for (let j = 0; j < k; j++) {
          if (
            j === label ||
            upper <= lower[i * k + j] ||
            upper <= half[label * k + j]
          ) {
            continue;
          }
          if (!tight) {
            upper = euclidean(
              points,
              offset,
              centroids,
              label * dimension,
              dimension,
            );
            lower[i * k + label] = upper;
            tight = true;
            if (upper <= lower[i * k + j] || upper <= half[label * k + j]) {
              continue;
            }
          }
          const d = euclidean(
            points,
            offset,
            centroids,
            j * dimension,
            dimension,
          );
          lower[i * k + j] = d;
          if (d < upper) {
            label = j;
            upper = d;
          }
        }
      }
      if (label !== previous) {
        labels[i] = label;
        changed = true;
      }
      upperBounds[i] = upper;
      if (accumulate) this.#accumulate(i, label);
    }
    return changed;
  }

  #initHamerly() {
    const { count, labels } = this;
    for (let i = 0; i < count; i++) labels[i] = this.#nearestTwo(i);
  }

  // Exact nearest and second nearest distances, into the bounds.
  #nearestTwo(i) {
    const { k, dimension, centroids } = this;
    const points = this.#points;
    let label = 0;
    let first = Infinity;
    let second = Infinity;
    for (let c = 0; c < k; c++) {
      const d = euclidean(
        points,
        i * dimension,
        centroids,
        c * dimension,
        dimension,
      );
      if (d < first) {
        second = first;
        first = d;
        label = c;
      } else if (d < second) {
        second = d;
      }
    }
    this.#upper[i] = first;
    this.#lower[i] = second;
    return label;
  }

  #assignHamerly(accumulate) {
    const { dimension, count, centroids, labels } = this;
    const points = this.#points;
    const upper = this.#upper;
    const lower = this.#lower;

    let changed = false;
    for (let i = 0; i < count; i++) {
      let label = labels[i];
      const bound = Math.max(this.#nearestHalf[label], lower[i]);
      if (upper[i] > bound) {
        upper[i] = euclidean(
          points,
          i * dimension,
          centroids,
          label * dimension,
          dimension,
        );
        if (upper[i] > bound) label = this.#nearestTwo(i);
      }
      if (label !== labels[i]) {
        labels[i] = label;
        changed = true;
      }
      if (accumulate) this.#accumulate(i, label);
    }
    return changed;
  }

  // scikit-learn moves the point's weight to the empty cluster but keeps its
  // label until the next assignment.
  #move = (i, cluster) => {
    const { dimension } = this;
    const points = this.#points;
    const sums = this.#sums;
    const previous = this.labels[i];
    const w = weightOf(this.#weights, i);
    this.sizes[previous] -= w;
    this.sizes[cluster] = w;
    for (let a = 0; a < dimension; a++) {
      sums[previous * dimension + a] -= w * points[i * dimension + a];
      sums[cluster * dimension + a] = w * points[i * dimension + a];
    }
  };

  // Move the centroids to their cluster mean. Returns the sum of squared
  // shifts.
  #update() {
    const { k, dimension, centroids, sizes, algorithm } = this;
    const sums = this.#sums;
    const shifts = this.#shifts;

    if (this.emptyClusters === "farthest" && sizes.includes(0)) {
      this.#relocateEmptyClusters();
    }

    let shiftTotal = 0;
    for (let c = 0; c < k; c++) {
      shifts[c] = 0;
      if (sizes[c] <= 0) continue;
      const offset = c * dimension;
      for (let a = 0; a < dimension; a++) sums[offset + a] /= sizes[c];
      if (this.spherical)
        normalize(sums.subarray(offset, offset + dimension), 1, dimension);
      shifts[c] = euclidean(centroids, offset, sums, offset, dimension);
      shiftTotal += shifts[c] ** 2;
      for (let a = 0; a < dimension; a++)
        centroids[offset + a] = sums[offset + a];
    }

    if (algorithm === "elkan") this.#updateElkanBounds();
    else if (algorithm === "hamerly") this.#updateHamerlyBounds();
    return shiftTotal;
  }

  // Lloyd assignment already measured the distances.
  #relocateEmptyClusters() {
    const { dimension, count, centroids, labels, algorithm } = this;
    const distances = this._distances;
    if (algorithm !== "lloyd") {
      for (let i = 0; i < count; i++) {
        distances[i] = squaredEuclidean(
          this.#points,
          i * dimension,
          centroids,
          labels[i] * dimension,
          dimension,
        );
      }
    }
    relocateEmptyClusters(this.sizes, distances, count, this.#move);
  }

  #updateElkanBounds() {
    const { k, count, labels } = this;
    const shifts = this.#shifts;
    const lower = this.#lower;
    for (let i = 0; i < count; i++) {
      this.#upper[i] += shifts[labels[i]];
      for (let j = 0; j < k; j++) {
        lower[i * k + j] = Math.max(0, lower[i * k + j] - shifts[j]);
      }
    }
    this.#updateCentroidDistances();
  }

  // The second nearest centroid gets at most as close as the farthest moving
  // other centroid.
  #updateHamerlyBounds() {
    const { k, count, labels } = this;
    const shifts = this.#shifts;
    let farthest = 0;
    let second = 0;
    for (let c = 1; c < k; c++) {
      if (shifts[c] > shifts[farthest]) {
        second = farthest;
        farthest = c;
      } else if (second === farthest || shifts[c] > shifts[second]) {
        second = c;
      }
    }
    for (let i = 0; i < count; i++) {
      const label = labels[i];
      this.#upper[i] += shifts[label];
      this.#lower[i] -= shifts[label === farthest ? second : farthest];
    }
    this.#updateCentroidDistances();
  }
}
