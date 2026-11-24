/** @module clustering */

import { squaredEuclidean } from "./distance.js";
import { kMeansPlusPlus } from "./init.js";
import { gather, nearestCentroid, resize, weightOf } from "./utils.js";

/** @typedef {Float32Array | Float64Array | number[]} Points */
/** @typedef {ArrayLike<number>} Position */
/** @typedef {Float32Array | Float64Array | number[]} Weights */

/**
 * Empty cluster handling: move it to the point farthest from its centroid, as
 * in scikit-learn, or keep its centroid with a size of 0.
 *
 * @typedef {"farthest" | "keep"} EmptyClusters
 */

/**
 * @typedef {object} ClusteringOptions
 * @property {number} [dimension=3] Number of components per point.
 * @property {number} [k=8] Number of clusters. scikit-learn's default.
 * @property {function(): number} [random=Math.random] Uniform random number in
 *   [0, 1). Pass a seeded generator for reproducible results.
 */

/**
 * @typedef {object} InitializedOptions
 * @property {import("./init.js").Init} [init=kMeansPlusPlus] Centroids
 *   initialization when `fit` isn't given centroids, from the `init` namespace
 *   or custom.
 */

/**
 * @typedef {object} BaseOptions
 * @property {import("./distance.js").Distance} [distance=squaredEuclidean]
 *   Distance of the points to the centroids.
 */

/**
 * @typedef {object} FitOptions
 * @property {Weights} [weights] Weight of each point. All 1 when omitted.
 * @property {ArrayLike<number>} [centroids] Initial centroids, `k * dimension`
 *   components. Skips `init`.
 */

/**
 * Shared state and nearest centroid queries of all clusterings. Subclasses
 * implement `fit`.
 */
export class Clustering {
  /**
   * Number of components per point.
   *
   * @type {number}
   */
  dimension;

  /**
   * Number of clusters.
   *
   * @type {number}
   */
  k;

  /**
   * Uniform random number in [0, 1).
   *
   * @type {function(): number}
   */
  random;

  /**
   * Distance assigning points to centroids.
   *
   * @type {import("./distance.js").Distance}
   */
  distance;

  /**
   * Centroids initialization.
   *
   * @type {import("./init.js").Init}
   */
  init;

  /**
   * Flat cluster centers of the last fit, `k * dimension` components.
   *
   * @type {Float64Array}
   */
  centroids = new Float64Array(0);

  /**
   * Cluster of each point of the last fit.
   *
   * @type {Int32Array}
   */
  labels = new Int32Array(0);

  /**
   * Sum of the point weights of each cluster: point counts without weights. 0
   * for an empty cluster.
   *
   * @type {Float64Array}
   */
  sizes = new Float64Array(0);

  /**
   * Number of points of the last fit.
   *
   * @type {number}
   */
  count = 0;

  /**
   * Objective minimized by the last fit. Lower is better.
   *
   * @type {number}
   */
  cost = 0;

  /**
   * Iterations run by the last fit.
   *
   * @type {number}
   */
  iterations = 0;

  /**
   * Whether the last fit converged before its iterations limit.
   *
   * @type {boolean}
   */
  converged = false;

  /**
   * Scratch distances, one per point.
   *
   * @type {Float64Array}
   * @protected
   */
  _distances = new Float64Array(0);

  /**
   * Selected point indices of the initialization.
   *
   * @type {Uint32Array}
   * @protected
   */
  _indices = new Uint32Array(0);

  /** @param {ClusteringOptions & InitializedOptions & BaseOptions} [options] */
  constructor({
    dimension = 3,
    k = 8,
    random = Math.random,
    distance = squaredEuclidean,
    init = kMeansPlusPlus,
  } = {}) {
    this.dimension = dimension;
    this.k = k;
    this.random = random;
    this.distance = distance;
    this.init = init;
  }

  /**
   * Cluster points. Buffers are reused across fits.
   *
   * @abstract
   * @param {Points} points Flat array of point components, `dimension` per
   *   point.
   * @param {object} [options]
   * @returns {this}
   */
  // eslint-disable-next-line no-unused-vars
  fit(points, options) {
    throw new Error("Not implemented");
  }

  /**
   * Index of the nearest centroid of a single position, such as a new point:
   * the Voronoi cell containing it.
   *
   * @param {Position} position
   * @returns {number}
   */
  predict(position) {
    const { centroids, k, dimension, distance } = this;
    return nearestCentroid(position, 0, centroids, k, dimension, distance);
  }

  /**
   * Index of the nearest centroid of each point, such as labelling a full
   * dataset after fitting a sample. scikit-learn's `predict`.
   *
   * @param {Points} points Flat array of point components.
   * @param {Int32Array} [out] Receives the indices. Reuse it to avoid
   *   allocating.
   * @returns {Int32Array}
   */
  assign(points, out) {
    const { centroids, k, dimension, distance } = this;
    const count = Math.floor(points.length / dimension);
    out ??= new Int32Array(count);
    for (let i = 0; i < count; i++) {
      out[i] = nearestCentroid(
        points,
        i * dimension,
        centroids,
        k,
        dimension,
        distance,
      );
    }
    return out;
  }

  /**
   * Size the buffers for `points`.
   *
   * @param {Points} points
   * @protected
   */
  _prepare(points) {
    const { k, dimension } = this;
    const count = Math.floor(points.length / dimension);
    if (!Number.isInteger(k) || k < 1 || k > count) {
      throw new RangeError(`k must be an integer between 1 and ${count}.`);
    }
    this.count = count;
    this.centroids = resize(this.centroids, k * dimension);
    this.labels = resize(this.labels, count);
    this.sizes = resize(this.sizes, k);
    this._distances = resize(this._distances, count);
    this._indices = resize(this._indices, k);
    this.iterations = 0;
    this.converged = false;
  }

  /**
   * Select `k` points with `init` into `indices`.
   *
   * @param {Points} points
   * @param {Uint32Array} indices
   * @param {import("./init.js").InitOptions} options Distance of the objective
   *   and other options of the clustering.
   * @protected
   */
  _initIndices(points, indices, options) {
    const { k, dimension, random } = this;
    const selected = this.init(points, k, {
      dimension,
      random,
      indices,
      distances: this._distances,
      ...options,
    });
    // A custom init may return its own array instead of filling `indices`.
    if (selected !== indices) indices.set(selected);
  }

  /**
   * Copy the points at `indices` into `centroids`.
   *
   * @param {Points} points
   * @param {ArrayLike<number>} indices
   * @protected
   */
  _setCentroids(points, indices) {
    gather(points, indices, this.k, this.dimension, this.centroids);
  }

  /**
   * Set the initial centroids: the given ones, or selected points.
   *
   * @param {Points} points
   * @param {FitOptions} options
   * @param {import("./distance.js").Distance} distance Distance of the
   *   objective, for the initialization.
   * @protected
   */
  _initCentroids(points, { weights, centroids }, distance) {
    const { k, dimension } = this;
    if (centroids) {
      if (centroids.length !== k * dimension) {
        throw new RangeError(
          `centroids must have ${k * dimension} components.`,
        );
      }
      this.centroids.set(centroids);
      return;
    }

    this._initIndices(points, this._indices, { weights, distance });
    this._setCentroids(points, this._indices);
  }

  /**
   * Alternate assignments and centroid updates (Lloyd iterations) until no
   * label changes or the centroids shift by `tolerance` at most.
   *
   * @param {number} maxIterations
   * @param {number} tolerance Absolute tolerance on the sum of squared centroid
   *   shifts. See `utils.scaledTolerance`.
   * @param {function(boolean): boolean} assign Called with whether an update
   *   follows. Returns whether a label changed.
   * @param {function(): number} update Returns the sum of squared centroid
   *   shifts.
   * @protected
   */
  _iterate(maxIterations, tolerance, assign, update) {
    let strict = false;
    let shift = 0;
    for (let i = 0; i < maxIterations; i++) {
      // The initialization may have set the labels already.
      const changed = assign(true) || i === 0;
      shift = update();
      this.iterations = i + 1;
      if (!changed) {
        strict = true;
        break;
      }
      if (shift <= tolerance) break;
    }
    // Labels match the centroids before their last update otherwise.
    if (!strict) assign(false);
    this.converged = strict || shift <= tolerance;
  }

  /**
   * Fill `sizes` from `labels`.
   *
   * @param {Points} points
   * @param {Weights} [weights]
   * @returns {number} Weighted sum of the distances of the points to their
   *   centroid.
   * @protected
   */
  _tally(points, weights) {
    const { centroids, labels, sizes, count, dimension, distance } = this;
    sizes.fill(0);
    let sum = 0;
    for (let i = 0; i < count; i++) {
      const w = weightOf(weights, i);
      sizes[labels[i]] += w;
      sum +=
        w *
        distance(
          points,
          i * dimension,
          centroids,
          labels[i] * dimension,
          dimension,
        );
    }
    return sum;
  }
}
