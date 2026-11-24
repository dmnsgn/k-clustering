/** @module fuzzyCMeans */

import { Clustering } from "./clustering.js";
import { euclidean, squaredEuclidean } from "./distance.js";
import { resize, weightOf } from "./utils.js";

/**
 * @typedef {object} FuzzyCMeansOptions
 * @property {number} [fuzziness=2] Exponent m above 1 weighting the memberships: higher blends clusters more. The usual choice, middle of the [1.5, 2.5] range of Pal and Bezdek (1995).
 * @property {number} [maxIterations=150] The fuzzy-c-means package's default.
 * @property {number} [tolerance=1e-5] Stop when the Frobenius norm of the memberships change falls under it, as in scikit-fuzzy. The fuzzy-c-means package's default.
 */

/**
 * Fuzzy c-means clustering: each point belongs to every cluster with a
 * membership between 0 and 1, minimizing the sum of squared distances weighted
 * by the memberships raised to `fuzziness`.
 *
 * Suited to overlapping clusters and soft boundaries, like color gradients.
 * `labels` hold the cluster of highest membership.
 *
 * @augments {Clustering}
 * @see Dunn, J. C. (1973). A fuzzy relative of the ISODATA process and its use in detecting compact well-separated clusters. Journal of Cybernetics, 3(3), 32–57. https://doi.org/10.1080/01969727308546046
 * @see Bezdek, J. C. (1981). Pattern recognition with fuzzy objective function algorithms. Plenum Press. https://doi.org/10.1007/978-1-4757-0450-1
 * @see Bezdek, J. C., Ehrlich, R., & Full, W. (1984). FCM: the fuzzy c-means clustering algorithm. Computers & Geosciences, 10(2–3), 191–203. https://doi.org/10.1016/0098-3004(84)90020-7
 * @see Pal, N. R., & Bezdek, J. C. (1995). On cluster validity for the fuzzy c-means model. IEEE Transactions on Fuzzy Systems, 3(3), 370–379. https://doi.org/10.1109/91.413225 (fuzziness)
 * @see https://github.com/scikit-fuzzy/scikit-fuzzy/blob/v0.5.0/skfuzzy/cluster/_cmeans.py (convergence, distances floor)
 * @see https://github.com/omadson/fuzzy-c-means (defaults)
 */
export class FuzzyCMeans extends Clustering {
  /** @type {number} */
  fuzziness;
  /** @type {number} */
  maxIterations;
  /** @type {number} */
  tolerance;

  /**
   * Membership of each point in each cluster of the last fit, `count * k`
   * values. Each point's memberships sum to 1.
   *
   * @type {Float64Array}
   */
  memberships = new Float64Array(0);

  #previous = new Float64Array(0);
  #row = new Float64Array(0);

  /**
   * @param {import("./clustering.js").ClusteringOptions & import("./clustering.js").InitializedOptions & FuzzyCMeansOptions} [options]
   */
  constructor({
    fuzziness = 2,
    maxIterations = 150,
    tolerance = 1e-5,
    ...options
  } = {}) {
    super({ ...options, distance: squaredEuclidean });
    this.fuzziness = fuzziness;
    this.maxIterations = maxIterations;
    this.tolerance = tolerance;
  }

  /**
   * @param {import("./clustering.js").Points} points Flat array of point
   *   components, `dimension` per point.
   * @param {import("./clustering.js").FitOptions} [options]
   * @returns {this}
   */
  fit(points, { weights, centroids } = {}) {
    if (Number.isNaN(this.fuzziness) || this.fuzziness <= 1) {
      throw new RangeError("fuzziness must be greater than 1.");
    }
    this._prepare(points);
    const { k, count } = this;
    this.memberships = resize(this.memberships, count * k);
    this.#previous = resize(this.#previous, count * k);
    this.#row = resize(this.#row, k);

    this._initCentroids(points, { weights, centroids }, squaredEuclidean);
    this.#updateMemberships(points);

    for (let i = 0; i < this.maxIterations; i++) {
      this.#previous.set(this.memberships);
      this.#updateCentroids(points, weights);
      this.#updateMemberships(points);
      this.iterations = i + 1;

      let change = 0;
      for (let j = 0; j < count * k; j++) {
        change += (this.memberships[j] - this.#previous[j]) ** 2;
      }
      if (Math.sqrt(change) < this.tolerance) {
        this.converged = true;
        break;
      }
    }

    this.#labelByMembership();
    this._tally(points, weights);
    this.cost = this.#objective(points, weights);
    return this;
  }

  // Memberships are floored to machine epsilon before weighting, as in
  // scikit-fuzzy.
  #updateCentroids(points, weights) {
    const { k, dimension, count, centroids, memberships, fuzziness } = this;
    const totals = this.#row;
    centroids.fill(0);
    totals.fill(0);
    for (let i = 0; i < count; i++) {
      const w = weightOf(weights, i);
      for (let c = 0; c < k; c++) {
        const u =
          w * Math.max(memberships[i * k + c], Number.EPSILON) ** fuzziness;
        totals[c] += u;
        for (let a = 0; a < dimension; a++) {
          centroids[c * dimension + a] += u * points[i * dimension + a];
        }
      }
    }
    for (let c = 0; c < k; c++) {
      for (let a = 0; a < dimension; a++) {
        centroids[c * dimension + a] /= totals[c];
      }
    }
  }

  // u_ic = 1 / Σ_j (d_ic / d_ij)^(2 / (m - 1)). Distances are floored to
  // machine epsilon, so a point on a centroid gets a membership of almost 1.
  #updateMemberships(points) {
    const { k, dimension, count, centroids, memberships, fuzziness } = this;
    const row = this.#row;
    const exponent = -2 / (fuzziness - 1);
    for (let i = 0; i < count; i++) {
      let min = Infinity;
      for (let c = 0; c < k; c++) {
        row[c] = Math.max(
          euclidean(points, i * dimension, centroids, c * dimension, dimension),
          Number.EPSILON,
        );
        min = Math.min(min, row[c]);
      }
      // Relative to the nearest centroid to avoid overflowing.
      let sum = 0;
      for (let c = 0; c < k; c++) {
        row[c] = (row[c] / min) ** exponent;
        sum += row[c];
      }
      for (let c = 0; c < k; c++) memberships[i * k + c] = row[c] / sum;
    }
  }

  #labelByMembership() {
    const { k, count, memberships, labels } = this;
    for (let i = 0; i < count; i++) {
      let label = 0;
      for (let c = 1; c < k; c++) {
        if (memberships[i * k + c] > memberships[i * k + label]) label = c;
      }
      labels[i] = label;
    }
  }

  #objective(points, weights) {
    const { k, dimension, count, centroids, memberships, fuzziness } = this;
    let sum = 0;
    for (let i = 0; i < count; i++) {
      const w = weightOf(weights, i);
      for (let c = 0; c < k; c++) {
        sum +=
          w *
          memberships[i * k + c] ** fuzziness *
          squaredEuclidean(
            points,
            i * dimension,
            centroids,
            c * dimension,
            dimension,
          );
      }
    }
    return sum;
  }
}
