/** @module distance */

/**
 * Distance between the points at `aOffset` in `a` and `bOffset` in `b`.
 *
 * @callback Distance
 * @param {ArrayLike<number>} a
 * @param {number} aOffset Index of the first component in `a`.
 * @param {ArrayLike<number>} b
 * @param {number} bOffset Index of the first component in `b`.
 * @param {number} dimension
 * @returns {number}
 */

/** @type {Distance} */
export function squaredEuclidean(a, aOffset, b, bOffset, dimension) {
  let sum = 0;
  for (let k = 0; k < dimension; k++) {
    const delta = a[aOffset + k] - b[bOffset + k];
    sum += delta * delta;
  }
  return sum;
}

/** @type {Distance} */
export function euclidean(a, aOffset, b, bOffset, dimension) {
  return Math.sqrt(squaredEuclidean(a, aOffset, b, bOffset, dimension));
}

/** @type {Distance} */
export function manhattan(a, aOffset, b, bOffset, dimension) {
  let sum = 0;
  for (let k = 0; k < dimension; k++) {
    sum += Math.abs(a[aOffset + k] - b[bOffset + k]);
  }
  return sum;
}

/** @type {Distance} */
export function chebyshev(a, aOffset, b, bOffset, dimension) {
  let max = 0;
  for (let k = 0; k < dimension; k++) {
    max = Math.max(max, Math.abs(a[aOffset + k] - b[bOffset + k]));
  }
  return max;
}

/**
 * One minus the cosine similarity, between 0 and 2.
 *
 * @type {Distance}
 */
export function cosine(a, aOffset, b, bOffset, dimension) {
  let dot = 0;
  let aNorm = 0;
  let bNorm = 0;
  for (let k = 0; k < dimension; k++) {
    const x = a[aOffset + k];
    const y = b[bOffset + k];
    dot += x * y;
    aNorm += x * x;
    bNorm += y * y;
  }
  // A zero vector has no direction: as dissimilar as an orthogonal one, as in
  // scikit-learn.
  return aNorm && bNorm ? 1 - dot / Math.sqrt(aNorm * bNorm) : 1;
}
