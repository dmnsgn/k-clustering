/** @module kMedoids */

import { Clustering } from "./clustering.js";
import { euclidean } from "./distance.js";
import { build } from "./init.js";
import { resize } from "./utils.js";

/**
 * @typedef {object} KMedoidsOptions
 * @property {"fasterpam" | "pam" | "alternate"} [algorithm="fasterpam"] FasterPAM eagerly swaps medoids, as good as PAM in O(n²) per iteration. PAM makes the best swap per iteration, in O(k n²). Alternate updates medoids within fixed clusters, like k-means: faster but worse.
 * @property {import("./init.js").Init} [init=build] Medoids initialization when `fit` isn't given medoids. Reads the dissimilarity matrix with `init.build`.
 * @property {import("./distance.js").Distance} [distance=euclidean] Any dissimilarity, not necessarily a metric.
 * @property {number} [maxIterations=100] The kmedoids package's default, also bounding the swaps of `"pam"` and the partition updates of `"alternate"`.
 */

/**
 * @typedef {object} KMedoidsFitOptions
 * @property {ArrayLike<number>} [medoids] Initial medoid point indices. Skips `init`.
 * @property {Float64Array} [dissimilarities] Precomputed `count * count` dissimilarity matrix, row-major. Skips `distance`.
 */

/**
 * k-medoids clustering: minimize the sum of distances of the points to their
 * cluster medoid, a point of the cluster.
 *
 * Works with any dissimilarity and returns actual points. Stores a
 * `count * count` dissimilarity matrix: suited to a few thousand points.
 *
 * Point weights aren't supported.
 *
 * @augments {Clustering}
 * @see Kaufman, L., & Rousseeuw, P. J. (1990). Partitioning around medoids (program PAM). In Finding groups in data: an introduction to cluster analysis, 68–125. Wiley. https://doi.org/10.1002/9780470316801.ch2
 * @see Schubert, E., & Rousseeuw, P. J. (2021). Fast and eager k-medoids clustering: O(k) runtime improvement of the PAM, CLARA, and CLARANS algorithms. Information Systems, 101, 101804. https://doi.org/10.1016/j.is.2021.101804
 * @see Park, H.-S., & Jun, C.-H. (2009). A simple and fast algorithm for K-medoids clustering. Expert Systems with Applications, 36(2), 3336–3341. https://doi.org/10.1016/j.eswa.2008.01.039 (alternate)
 * @see https://github.com/kno10/rust-kmedoids (original implementation)
 */
export class KMedoids extends Clustering {
  /** @type {"fasterpam" | "pam" | "alternate"} */
  algorithm;
  /** @type {number} */
  maxIterations;

  /**
   * Medoid point indices of the last fit.
   *
   * @type {Uint32Array}
   */
  medoids = new Uint32Array(0);

  /**
   * Medoid swaps of the last fit.
   *
   * @type {number}
   */
  swaps = 0;

  #matrix = new Float64Array(0);
  /** @type {Float64Array} */
  #dissimilarities = this.#matrix;
  #count = 0;
  // Nearest and second nearest medoid of each point, as positions in
  // `medoids`. -1 when unset.
  #nearest = new Int32Array(0);
  #nearestDistance = new Float64Array(0);
  #second = new Int32Array(0);
  #secondDistance = new Float64Array(0);
  #removalLoss = new Float64Array(0);
  #swapLoss = new Float64Array(0);

  /**
   * @param {import("./clustering.js").ClusteringOptions & KMedoidsOptions} [options]
   */
  constructor({
    algorithm = "fasterpam",
    init = build,
    distance = euclidean,
    maxIterations = 100,
    ...options
  } = {}) {
    super({ ...options, init, distance });
    this.algorithm = algorithm;
    this.maxIterations = maxIterations;
  }

  /**
   * @param {import("./clustering.js").Points} points Flat array of point
   *   components, `dimension` per point.
   * @param {KMedoidsFitOptions} [options]
   * @returns {this}
   */
  fit(points, { medoids, dissimilarities } = {}) {
    this._prepare(points);
    const { k, count } = this;
    this.#count = count;
    this.medoids = resize(this.medoids, k);
    this.#nearest = resize(this.#nearest, count);
    this.#nearestDistance = resize(this.#nearestDistance, count);
    this.#second = resize(this.#second, count);
    this.#secondDistance = resize(this.#secondDistance, count);
    this.#removalLoss = resize(this.#removalLoss, k);
    this.#swapLoss = resize(this.#swapLoss, k);
    this.swaps = 0;

    if (dissimilarities) {
      if (dissimilarities.length !== count * count) {
        throw new RangeError(
          `dissimilarities must have ${count * count} elements.`,
        );
      }
      this.#dissimilarities = dissimilarities;
    } else {
      this.#dissimilarities = this.#matrix = resize(
        this.#matrix,
        count * count,
      );
      this.#computeDissimilarities(points);
    }

    if (medoids) {
      this.medoids.set(medoids);
    } else {
      this._initIndices(points, this.medoids, {
        distance: this.distance,
        dissimilarities: this.#dissimilarities,
      });
    }

    let loss;
    if (k === 1) {
      this.labels.fill(0);
      this.swaps = Number(this.#chooseMedoidWithinPartition(0));
      this.iterations = 1;
      this.converged = true;
    } else if (this.algorithm === "alternate") {
      loss = this.#alternate();
    } else {
      loss = this.#initialAssignment();
      loss =
        this.algorithm === "pam" ? this.#pamSwap(loss) : this.#fasterPam(loss);
      this.labels.set(this.#nearest);
    }

    this._setCentroids(points, this.medoids);
    this.cost = loss ?? this.#lossOf();
    this._tally(points);
    return this;
  }

  #get(i, j) {
    return this.#dissimilarities[i * this.#count + j];
  }

  #computeDissimilarities(points) {
    const { count, dimension, distance } = this;
    const matrix = this.#matrix;
    for (let i = 0; i < count; i++) {
      matrix[i * count + i] = 0;
      for (let j = i + 1; j < count; j++) {
        const d = distance(
          points,
          i * dimension,
          points,
          j * dimension,
          dimension,
        );
        matrix[i * count + j] = d;
        matrix[j * count + i] = d;
      }
    }
  }

  #lossOf() {
    let loss = 0;
    for (let i = 0; i < this.count; i++) {
      loss += this.#get(i, this.medoids[this.labels[i]]);
    }
    return loss;
  }

  #initialAssignment() {
    const { k, count, medoids } = this;
    const nearest = this.#nearest;
    const nearestDistance = this.#nearestDistance;
    const second = this.#second;
    const secondDistance = this.#secondDistance;

    let loss = 0;
    for (let i = 0; i < count; i++) {
      nearest[i] = 0;
      nearestDistance[i] = this.#get(i, medoids[0]);
      second[i] = -1;
      secondDistance[i] = 0;
      for (let m = 1; m < k; m++) {
        const d = this.#get(i, medoids[m]);
        if (d < nearestDistance[i] || i === medoids[m]) {
          second[i] = nearest[i];
          secondDistance[i] = nearestDistance[i];
          nearest[i] = m;
          nearestDistance[i] = d;
        } else if (second[i] === -1 || d < secondDistance[i]) {
          second[i] = m;
          secondDistance[i] = d;
        }
      }
      loss += nearestDistance[i];
    }
    return loss;
  }

  #updateRemovalLoss() {
    const removalLoss = this.#removalLoss;
    removalLoss.fill(0);
    for (let i = 0; i < this.count; i++) {
      removalLoss[this.#nearest[i]] +=
        this.#secondDistance[i] - this.#nearestDistance[i];
    }
  }

  // Loss change of the best swap of point j with a medoid, in O(n + k).
  #findBestSwap(j) {
    const ploss = this.#swapLoss;
    const nearest = this.#nearest;
    const nearestDistance = this.#nearestDistance;
    const secondDistance = this.#secondDistance;
    ploss.set(this.#removalLoss);
    let shared = 0;
    for (let o = 0; o < this.count; o++) {
      const doj = this.#get(o, j);
      if (doj < nearestDistance[o]) {
        shared += doj - nearestDistance[o];
        ploss[nearest[o]] += nearestDistance[o] - secondDistance[o];
      } else if (doj < secondDistance[o]) {
        ploss[nearest[o]] += doj - secondDistance[o];
      }
    }
    let best = 0;
    for (let m = 1; m < this.k; m++) if (ploss[m] < ploss[best]) best = m;
    return { change: ploss[best] + shared, medoid: best };
  }

  #fasterPam(loss) {
    const { count, medoids } = this;
    let lastSwap = count;
    let iteration = 0;
    this.#updateRemovalLoss();
    while (iteration < this.maxIterations) {
      iteration++;
      const swapsBefore = this.swaps;
      const lastLoss = loss;
      for (let j = 0; j < count; j++) {
        // A full pass without swap since the last one: no swap is left.
        if (j === lastSwap) break;
        if (j === medoids[this.#nearest[j]]) continue;
        const { change, medoid } = this.#findBestSwap(j);
        if (change >= 0) continue;
        this.swaps++;
        lastSwap = j;
        loss = this.#doSwap(medoid, j);
        this.#updateRemovalLoss();
      }
      if (this.swaps === swapsBefore || loss >= lastLoss) {
        this.converged = true;
        break;
      }
    }
    this.iterations = iteration;
    return loss;
  }

  #pamSwap(loss) {
    const { k, count, medoids } = this;
    let iteration = 0;
    while (iteration < this.maxIterations) {
      iteration++;
      let bestChange = 0;
      let bestMedoid = k;
      let bestPoint = -1;
      for (let j = 0; j < count; j++) {
        if (j === medoids[this.#nearest[j]]) continue;
        const { change, medoid } = this.#findBestSwapPam(j);
        if (change >= bestChange) continue;
        bestChange = change;
        bestMedoid = medoid;
        bestPoint = j;
      }
      if (bestChange >= 0) {
        this.converged = true;
        break;
      }
      this.swaps++;
      const newLoss = this.#doSwap(bestMedoid, bestPoint);
      // Swaps that don't decrease the loss are numerical noise.
      if (newLoss >= loss) {
        this.converged = true;
        break;
      }
      loss = newLoss;
    }
    this.iterations = iteration;
    return loss;
  }

  // Loss change of swapping point j with each medoid, in O(k n).
  #findBestSwapPam(j) {
    const nearest = this.#nearest;
    const nearestDistance = this.#nearestDistance;
    const secondDistance = this.#secondDistance;
    let bestChange = 0;
    let bestMedoid = -1;
    for (let m = 0; m < this.k; m++) {
      let change = -nearestDistance[j];
      for (let o = 0; o < this.count; o++) {
        if (o === j) continue;
        const doj = this.#get(o, j);
        if (nearest[o] === m) {
          change += Math.min(doj, secondDistance[o]) - nearestDistance[o];
        } else if (doj < nearestDistance[o]) {
          change += doj - nearestDistance[o];
        }
      }
      if (change < bestChange) {
        bestChange = change;
        bestMedoid = m;
      }
    }
    return { change: bestChange, medoid: bestMedoid };
  }

  // Replace medoid b by point j, and update the nearest medoids.
  #doSwap(b, j) {
    const nearest = this.#nearest;
    const nearestDistance = this.#nearestDistance;
    const second = this.#second;
    const secondDistance = this.#secondDistance;
    this.medoids[b] = j;

    let loss = 0;
    for (let o = 0; o < this.count; o++) {
      if (o === j) {
        if (nearest[o] !== b) {
          second[o] = nearest[o];
          secondDistance[o] = nearestDistance[o];
        }
        nearest[o] = b;
        nearestDistance[o] = 0;
        continue;
      }
      const doj = this.#get(o, j);
      if (nearest[o] === b) {
        if (doj < secondDistance[o]) {
          nearestDistance[o] = doj;
        } else {
          nearest[o] = second[o];
          nearestDistance[o] = secondDistance[o];
          this.#updateSecondNearest(o, b, doj);
        }
      } else if (doj < nearestDistance[o]) {
        second[o] = nearest[o];
        secondDistance[o] = nearestDistance[o];
        nearest[o] = b;
        nearestDistance[o] = doj;
      } else if (doj < secondDistance[o]) {
        second[o] = b;
        secondDistance[o] = doj;
      } else if (second[o] === b) {
        this.#updateSecondNearest(o, b, doj);
      }
      loss += nearestDistance[o];
    }
    return loss;
  }

  #updateSecondNearest(o, b, doj) {
    let index = b;
    let min = doj;
    for (let m = 0; m < this.k; m++) {
      if (m === this.#nearest[o] || m === b) continue;
      const d = this.#get(o, this.medoids[m]);
      if (d < min) {
        index = m;
        min = d;
      }
    }
    this.#second[o] = index;
    this.#secondDistance[o] = min;
  }

  #alternate() {
    let loss = this.#assignNearest();
    let iteration = 0;
    while (iteration < this.maxIterations) {
      iteration++;
      let changed = false;
      for (let m = 0; m < this.k; m++) {
        changed = this.#chooseMedoidWithinPartition(m) || changed;
      }
      if (!changed) {
        this.converged = true;
        break;
      }
      loss = this.#assignNearest();
    }
    this.iterations = iteration;
    return loss;
  }

  #assignNearest() {
    const { k, count, medoids, labels } = this;
    let loss = 0;
    for (let i = 0; i < count; i++) {
      let label = 0;
      let min = this.#get(i, medoids[0]);
      for (let m = 1; m < k; m++) {
        const d = this.#get(i, medoids[m]);
        if (d < min || i === medoids[m]) {
          label = m;
          min = d;
        }
      }
      labels[i] = label;
      loss += min;
    }
    return loss;
  }

  // Point of cluster m with the lowest sum of distances to the others.
  #chooseMedoidWithinPartition(m) {
    const { count, labels, medoids } = this;
    const first = medoids[m];
    let best = first;
    let bestSum = 0;
    for (let i = 0; i < count; i++) {
      if (i !== first && labels[i] === m) bestSum += this.#get(first, i);
    }
    for (let j = 0; j < count; j++) {
      if (j === first || labels[j] !== m) continue;
      let sum = 0;
      for (let i = 0; i < count; i++) {
        if (i !== j && labels[i] === m) sum += this.#get(j, i);
      }
      if (sum < bestSum) {
        best = j;
        bestSum = sum;
      }
    }
    medoids[m] = best;
    return best !== first;
  }
}
