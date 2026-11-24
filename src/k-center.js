/** @module kCenter */

import { Clustering } from "./clustering.js";
import { euclidean } from "./distance.js";
import { farthestFirst } from "./init.js";
import { resize } from "./utils.js";

/**
 * @typedef {object} KCenterOptions
 * @property {import("./distance.js").Distance} [distance=euclidean] A metric: the 2-approximation relies on the triangle inequality.
 */

/**
 * @typedef {object} KCenterFitOptions
 * @property {number} [first] Index of the first center. Random when omitted.
 */

/**
 * k-center clustering: minimize the largest distance of a point to its
 * center, a point of the input, by farthest-first traversal.
 *
 * Within twice the optimal radius, in O(k n). Covers outliers rather than
 * dense regions: suited to coverage, sampling and spreading colors apart.
 * Point weights don't change the radius and aren't supported.
 *
 * @augments {Clustering}
 * @see Gonzalez, T. F. (1985). Clustering to minimize the maximum intercluster distance. Theoretical Computer Science, 38, 293–306. https://doi.org/10.1016/0304-3975(85)90224-5
 * @see Hochbaum, D. S., & Shmoys, D. B. (1985). A best possible heuristic for the k-center problem. Mathematics of Operations Research, 10(2), 180–184. https://doi.org/10.1287/moor.10.2.180 (2 is the best ratio unless P = NP)
 */
export class KCenter extends Clustering {
  /**
   * Center point indices of the last fit.
   *
   * @type {Uint32Array}
   */
  centers = new Uint32Array(0);

  /**
   * @param {import("./clustering.js").ClusteringOptions & KCenterOptions} [options]
   */
  constructor({ distance = euclidean, ...options } = {}) {
    super({ ...options, distance });
  }

  /**
   * @param {import("./clustering.js").Points} points Flat array of point
   *   components, `dimension` per point.
   * @param {KCenterFitOptions} [options]
   * @returns {this}
   */
  fit(points, { first } = {}) {
    this._prepare(points);
    const { k, dimension, count, random, distance, labels } = this;
    const distances = this._distances;
    this.centers = resize(this.centers, k);

    farthestFirst(points, k, {
      dimension,
      random,
      distance,
      first,
      indices: this.centers,
      distances,
      labels,
    });

    this._setCentroids(points, this.centers);
    this._tally(points);
    this.cost = 0;
    for (let i = 0; i < count; i++) {
      this.cost = Math.max(this.cost, distances[i]);
    }
    this.iterations = k;
    this.converged = true;
    return this;
  }
}
