/** @module init */

import { euclidean, squaredEuclidean } from "./distance.js";
import { resize, weightOf } from "./utils.js";

/**
 * Options given by the clusterings to their `init`. Each initialization reads
 * the ones it needs.
 *
 * @typedef {object} InitOptions
 * @property {number} [dimension=3] Number of components per point.
 * @property {function(): number} [random=Math.random] Uniform random number in [0, 1).
 * @property {Uint32Array} [indices] Receives the selected point indices. Reuse
 *   it to avoid allocating.
 * @property {ArrayLike<number>} [weights] Weight of each point.
 * @property {import("./distance.js").Distance} [distance] Distance of the clustering objective.
 * @property {Float64Array} [distances] Scratch buffer of at least one element per point.
 */

/**
 * Select `k` distinct points to initialize a clustering, such as
 * `kMeansPlusPlus`. Wrap one to set its own options:
 * `(points, k, options) => kMeansPlusPlus(points, k, { ...options, localTrials: 1 })`.
 *
 * @callback Init
 * @param {ArrayLike<number>} points Flat array of point components.
 * @param {number} k
 * @param {InitOptions} options
 * @returns {ArrayLike<number>} Indices of the selected points.
 */

/**
 * Pick `k` distinct points uniformly (Forgy initialization).
 *
 * @param {ArrayLike<number>} points Flat array of point components.
 * @param {number} k
 * @param {InitOptions} [options]
 * @returns {Uint32Array} Indices of the selected points.
 * @see Forgy, E. W. (1965). Cluster analysis of multivariate data: efficiency versus interpretability of classifications. Biometrics, 21, 768–769.
 * @see Bentley, J., & Floyd, R. W. (1987). Programming pearls: a sample of brilliance. Communications of the ACM, 30(9), 754–757. https://doi.org/10.1145/30401.315746 (sampling without replacement)
 */
export function randomPoints(
  points,
  k,
  { dimension = 3, random = Math.random, indices = new Uint32Array(k) } = {},
) {
  const count = Math.floor(points.length / dimension);
  // Floyd's algorithm: O(k²) checks but no O(n) shuffle buffer, as k << n.
  for (let s = 0, j = count - k; j < count; s++, j++) {
    const t = Math.floor(random() * (j + 1));
    let taken = false;
    for (let p = 0; p < s; p++) {
      if (indices[p] === t) {
        taken = true;
        break;
      }
    }
    indices[s] = taken ? j : t;
  }
  return indices;
}

/**
 * @typedef {object} KMeansPlusPlusOptions
 * @property {import("./distance.js").Distance} [distance=squaredEuclidean] Points are sampled proportionally to their distance to the nearest selected point: squared Euclidean for k-means, Manhattan for k-medians.
 * @property {number} [localTrials] Candidates sampled per point, keeping the one reducing the potential the most. Defaults to 2 + ln(k) like the authors' and scikit-learn's implementations. 1 is the paper's algorithm.
 */

/**
 * Pick `k` points spread out by D² sampling (k-means++).
 *
 * @param {ArrayLike<number>} points Flat array of point components.
 * @param {number} k
 * @param {InitOptions & KMeansPlusPlusOptions} [options]
 * @returns {Uint32Array} Indices of the selected points.
 * @see Arthur, D., & Vassilvitskii, S. (2007). k-means++: the advantages of careful seeding. Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms, 1027–1035. https://dl.acm.org/doi/10.5555/1283383.1283494
 * @see https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/cluster/_kmeans.py (greedy local trials)
 */
export function kMeansPlusPlus(
  points,
  k,
  {
    dimension = 3,
    random = Math.random,
    indices = new Uint32Array(k),
    weights,
    distance = squaredEuclidean,
    localTrials = 2 + Math.floor(Math.log(k)),
    distances,
  } = {},
) {
  const count = Math.floor(points.length / dimension);
  const closest =
    distances && distances.length >= count
      ? distances
      : new Float64Array(count);

  // Index of the first point whose cumulative weighted potential reaches
  // `target`, as numpy.searchsorted.
  const sample = (target, potentials) => {
    let cumulative = 0;
    for (let i = 0; i < count; i++) {
      cumulative += weightOf(weights, i) * (potentials ? potentials[i] : 1);
      if (cumulative >= target) return i;
    }
    return count - 1;
  };

  let total = 0;
  for (let i = 0; i < count; i++) total += weightOf(weights, i);
  const first = sample(random() * total);
  indices[0] = first;

  let potential = 0;
  for (let i = 0; i < count; i++) {
    closest[i] = distance(
      points,
      i * dimension,
      points,
      first * dimension,
      dimension,
    );
    potential += weightOf(weights, i) * closest[i];
  }

  for (let c = 1; c < k; c++) {
    let best = -1;
    let bestPotential = Infinity;
    for (let trial = 0; trial < localTrials; trial++) {
      const candidate = sample(random() * potential, closest);
      let candidatePotential = 0;
      for (let i = 0; i < count; i++) {
        const d = distance(
          points,
          i * dimension,
          points,
          candidate * dimension,
          dimension,
        );
        candidatePotential += weightOf(weights, i) * Math.min(closest[i], d);
      }
      if (candidatePotential < bestPotential) {
        bestPotential = candidatePotential;
        best = candidate;
      }
    }

    indices[c] = best;
    potential = 0;
    for (let i = 0; i < count; i++) {
      const d = distance(
        points,
        i * dimension,
        points,
        best * dimension,
        dimension,
      );
      if (d < closest[i]) closest[i] = d;
      potential += weightOf(weights, i) * closest[i];
    }
  }
  return indices;
}

/**
 * @typedef {object} FarthestFirstOptions
 * @property {import("./distance.js").Distance} [distance=squaredEuclidean]
 * @property {number} [first] Index of the first point. Random when omitted.
 * @property {Int32Array} [labels] Receives the position in `indices` of the nearest selected point of each point.
 */

/**
 * Pick `k` points, each the farthest from the ones already selected
 * (farthest-first traversal, maximin initialization).
 *
 * Also a 2-approximation of the k-center problem. `distances` receives the
 * distance of each point to its nearest selected point.
 *
 * @param {ArrayLike<number>} points Flat array of point components.
 * @param {number} k
 * @param {InitOptions & FarthestFirstOptions} [options]
 * @returns {Uint32Array} Indices of the selected points.
 * @see Gonzalez, T. F. (1985). Clustering to minimize the maximum intercluster distance. Theoretical Computer Science, 38, 293–306. https://doi.org/10.1016/0304-3975(85)90224-5
 * @see Katsavounidis, I., Kuo, C.-C. J., & Zhang, Z. (1994). A new initialization technique for generalized Lloyd iteration. IEEE Signal Processing Letters, 1(10), 144–146. https://doi.org/10.1109/97.329844
 */
export function farthestFirst(
  points,
  k,
  {
    dimension = 3,
    random = Math.random,
    indices = new Uint32Array(k),
    distance = squaredEuclidean,
    first,
    distances,
    labels,
  } = {},
) {
  const count = Math.floor(points.length / dimension);
  const nearest =
    distances && distances.length >= count
      ? distances
      : new Float64Array(count);
  nearest.fill(Infinity, 0, count);
  labels?.fill(0, 0, count);

  let next = first ?? Math.floor(random() * count);
  for (let c = 0; c < k; c++) {
    indices[c] = next;
    let farthest = 0;
    for (let i = 0; i < count; i++) {
      const d = distance(
        points,
        i * dimension,
        points,
        next * dimension,
        dimension,
      );
      if (d < nearest[i]) {
        nearest[i] = d;
        if (labels) labels[i] = c;
      }
      if (nearest[i] > nearest[farthest]) farthest = i;
    }
    next = farthest;
  }
  return indices;
}

let buildColumn = new Float64Array(0);

/**
 * @typedef {object} BuildOptions
 * @property {import("./distance.js").Distance} [distance=euclidean]
 * @property {ArrayLike<number>} [dissimilarities] Precomputed `count * count` dissimilarity matrix, row-major. Skips `distance`.
 */

/**
 * Pick `k` points greedily, each reducing the sum of distances of the points
 * to their nearest selected one the most (PAM BUILD), in O(k n²).
 *
 * @param {ArrayLike<number>} points Flat array of point components.
 * @param {number} k
 * @param {InitOptions & BuildOptions} [options]
 * @returns {Uint32Array} Indices of the selected points.
 * @see Kaufman, L., & Rousseeuw, P. J. (1990). Partitioning around medoids (program PAM). In Finding groups in data: an introduction to cluster analysis, 68–125. Wiley. https://doi.org/10.1002/9780470316801.ch2
 */
export function build(
  points,
  k,
  {
    dimension = 3,
    indices = new Uint32Array(k),
    distance = euclidean,
    dissimilarities,
    distances,
  } = {},
) {
  const count = Math.floor(points.length / dimension);
  const nearest =
    distances && distances.length >= count
      ? distances
      : new Float64Array(count);
  const column = (buildColumn = resize(buildColumn, count));
  // Distances of all points to point i. A lookup function in the O(n²) loops
  // makes k-medoids about 1.5× slower than reading this buffer.
  const fillColumn = (i) => {
    if (dissimilarities) {
      for (let j = 0; j < count; j++)
        column[j] = dissimilarities[j * count + i];
    } else {
      for (let j = 0; j < count; j++) {
        column[j] = distance(
          points,
          j * dimension,
          points,
          i * dimension,
          dimension,
        );
      }
    }
    column[i] = 0;
  };

  // Start from the most central point.
  let best = 0;
  let bestSum = Infinity;
  for (let i = 0; i < count; i++) {
    fillColumn(i);
    let sum = 0;
    for (let j = 0; j < count; j++) sum += column[j];
    if (i === 0 || sum < bestSum) {
      bestSum = sum;
      best = i;
    }
  }
  indices[0] = best;
  fillColumn(best);
  nearest.set(column);

  let added = 1;
  for (; added < k; added++) {
    best = 0;
    bestSum = 0;
    for (let i = 0; i < count; i++) {
      fillColumn(i);
      let sum = 0;
      for (let j = 0; j < count; j++) {
        if (column[j] < nearest[j]) sum += column[j] - nearest[j];
      }
      if (i === 0 || sum < bestSum) {
        bestSum = sum;
        best = i;
      }
    }
    // No point reduces the sum: the remaining ones duplicate selected points.
    if (bestSum >= 0) break;
    fillColumn(best);
    for (let j = 0; j < count; j++) {
      if (column[j] < nearest[j]) nearest[j] = column[j];
    }
    indices[added] = best;
  }
  // The original implementation returns fewer points. Duplicates keep k.
  for (let i = 0; i < count && added < k; i++) {
    if (!indices.subarray(0, added).includes(i)) indices[added++] = i;
  }
  return indices;
}
