import test, { describe } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import random from "pex-random";
import { KdTree } from "spatial-partition";

import {
  KMeans,
  MiniBatchKMeans,
  BisectingKMeans,
  KMedians,
  KMedoids,
  KCenter,
  FuzzyCMeans,
  distance,
  init,
  metrics,
} from "../index.js";

const reference = JSON.parse(
  readFileSync(new URL("fixtures/reference.json", import.meta.url), "utf8"),
);
const data = {
  blobs2: { points: Float64Array.from(reference.data.blobs2), dimension: 2 },
  blobs3: { points: Float64Array.from(reference.data.blobs3), dimension: 3 },
};
const weights = Float64Array.from(reference.weights);

const seeded = (seed) => {
  const rng = random.create(seed);
  return () => rng.float();
};

// Values differing by more than a relative tolerance, empty when close.
const differences = (actual, expected, tolerance = 1e-9) => {
  if (actual.length !== expected.length) {
    return [`length: ${actual.length} != ${expected.length}`];
  }
  const result = [];
  for (let i = 0; i < expected.length; i++) {
    const scale = Math.max(1, Math.abs(expected[i]));
    const error = Math.abs(actual[i] - expected[i]);
    if (Number.isNaN(error) || error > tolerance * scale) {
      result.push(`[${i}]: ${actual[i]} != ${expected[i]}`);
    }
  }
  return result;
};

// Repeat each point by its integer weight.
const repeat = (points, dimension, counts) => {
  const repeated = [];
  for (let i = 0; i < counts.length; i++) {
    for (let r = 0; r < counts[i]; r++) {
      repeated.push(...points.subarray(i * dimension, (i + 1) * dimension));
    }
  }
  return Float64Array.from(repeated);
};

describe("KMeans", () => {
  const ALGORITHMS = {
    lloyd: () => ({ algorithm: "lloyd" }),
    elkan: () => ({ algorithm: "elkan" }),
    hamerly: () => ({ algorithm: "hamerly" }),
    kdTree: (dimension) => ({
      algorithm: "lloyd",
      index: new KdTree({ dimension }),
    }),
  };

  for (const [name, options] of Object.entries(ALGORITHMS)) {
    for (const expected of reference.kmeans) {
      const { points, dimension } = data[expected.data];
      const title = `${expected.data} k=${expected.k} weighted=${expected.weighted} tol=${expected.tolerance}`;
      test(`${name} matches scikit-learn from the same centroids (${title})`, () => {
        const kMeans = new KMeans({
          dimension,
          k: expected.k,
          tolerance: expected.tolerance,
          ...options(dimension),
        }).fit(points, {
          centroids: expected.init,
          weights: expected.weighted ? weights : undefined,
        });
        assert.deepEqual([...kMeans.labels], expected.labels);
        assert.deepEqual(differences(kMeans.centroids, expected.centroids), []);
        assert.deepEqual(differences([kMeans.cost], [expected.inertia]), []);
        assert.equal(kMeans.iterations, expected.iterations);
        assert.ok(kMeans.converged);
      });
    }
  }

  test("relocates an empty cluster to the farthest point", () => {
    const expected = reference.kmeans.at(-1);
    const kMeans = new KMeans({ dimension: 2, k: 4, tolerance: 0 }).fit(
      data.blobs2.points,
      {
        centroids: expected.init,
      },
    );
    assert.ok(kMeans.sizes.every((size) => size > 0));
    assert.deepEqual(differences(kMeans.centroids, expected.centroids), []);
  });

  test("keeps an empty cluster centroid", () => {
    const expected = reference.kmeans.at(-1);
    const kMeans = new KMeans({
      dimension: 2,
      k: 4,
      emptyClusters: "keep",
    }).fit(data.blobs2.points, { centroids: expected.init });
    assert.equal(kMeans.sizes[3], 0);
    assert.deepEqual([...kMeans.centroids.subarray(6)], [50, 50]);
  });

  test("leaves clusters empty beyond the number of distinct points", () => {
    const kMeans = new KMeans({ dimension: 1, k: 3 }).fit([0, 0, 1, 1], {
      centroids: [0, 1, 5],
    });
    assert.deepEqual([...kMeans.sizes], [2, 2, 0]);
  });

  test("spherical ignores the direction of a zero point", () => {
    const kMeans = new KMeans({ dimension: 2, k: 2, spherical: true }).fit(
      [0, 0, 1, 0, 0, 1, 1, 0.1],
      { centroids: [1, 0, 0, 1] },
    );
    assert.ok(kMeans.centroids.every(Number.isFinite));
    assert.deepEqual([...kMeans.labels.subarray(1)], [0, 1, 0]);
  });

  test("integer weights match repeated points", () => {
    const { points, dimension } = data.blobs3;
    const counts = Float64Array.from(
      { length: points.length / dimension },
      (_, i) => 1 + (i % 3),
    );
    const centroids = points.slice(0, 5 * dimension);
    const weighted = new KMeans({ dimension, k: 5, tolerance: 0 }).fit(points, {
      weights: counts,
      centroids,
    });
    const repeated = new KMeans({ dimension, k: 5, tolerance: 0 }).fit(
      repeat(points, dimension, counts),
      { centroids },
    );
    assert.deepEqual(differences(weighted.centroids, repeated.centroids), []);
    assert.deepEqual(differences([weighted.cost], [repeated.cost]), []);
  });

  test("spherical matches cosine Lloyd iterations", () => {
    const { points, dimension } = data.blobs3;
    const k = 4;
    const count = points.length / dimension;
    const kMeans = new KMeans({
      dimension,
      k,
      spherical: true,
      tolerance: 0,
    }).fit(points, { centroids: points.slice(0, k * dimension) });

    const unit = (array, offset) => {
      const norm = Math.hypot(...array.subarray(offset, offset + dimension));
      return array.subarray(offset, offset + dimension).map((x) => x / norm);
    };
    let centroids = Array.from({ length: k }, (_, c) =>
      unit(points, c * dimension),
    );
    let labels = new Int32Array(count).fill(-1);
    for (let iteration = 0; iteration < 300; iteration++) {
      const next = Int32Array.from({ length: count }, (_, i) => {
        let best = 0;
        for (let c = 1; c < k; c++) {
          if (
            distance.cosine(points, i * dimension, centroids[c], 0, dimension) <
            distance.cosine(
              points,
              i * dimension,
              centroids[best],
              0,
              dimension,
            )
          ) {
            best = c;
          }
        }
        return best;
      });
      const changed = next.some((label, i) => label !== labels[i]);
      labels = next;
      centroids = centroids.map((_, c) => {
        const sum = new Float64Array(dimension);
        for (let i = 0; i < count; i++) {
          if (labels[i] !== c) continue;
          const point = unit(points, i * dimension);
          for (let a = 0; a < dimension; a++) sum[a] += point[a];
        }
        return unit(sum, 0);
      });
      if (!changed) break;
    }
    assert.deepEqual([...kMeans.labels], [...labels]);
    assert.deepEqual(
      differences(
        kMeans.centroids,
        centroids.flatMap((c) => [...c]),
        1e-9,
      ),
      [],
    );
    const norms = Array.from({ length: k }, (_, c) =>
      Math.hypot(
        ...kMeans.centroids.subarray(c * dimension, (c + 1) * dimension),
      ),
    );
    assert.deepEqual(
      differences(
        norms,
        Array.from({ length: k }, () => 1),
      ),
      [],
    );
  });

  for (const initialization of [
    init.kMeansPlusPlus,
    init.randomPoints,
    init.farthestFirst,
  ]) {
    test(`is reproducible with a seeded generator (${initialization.name})`, () => {
      const { points, dimension } = data.blobs3;
      const fit = () =>
        new KMeans({
          dimension,
          k: 5,
          init: initialization,
          random: seeded(3),
        }).fit(points);
      assert.deepEqual([...fit().centroids], [...fit().centroids]);
    });
  }

  test("predict and assign match the labels after convergence", () => {
    const { points, dimension } = data.blobs3;
    const kMeans = new KMeans({
      dimension,
      k: 5,
      tolerance: 0,
      random: seeded(1),
    }).fit(points);
    assert.deepEqual([...kMeans.assign(points)], [...kMeans.labels]);
    assert.equal(
      kMeans.predict(points.subarray(0, dimension)),
      kMeans.labels[0],
    );
  });

  test("reuses buffers across fits of different sizes", () => {
    const { points, dimension } = data.blobs3;
    const kMeans = new KMeans({ dimension, k: 5, random: seeded(1) });
    kMeans.fit(points);
    kMeans.fit(points.subarray(0, 30 * dimension));
    assert.equal(kMeans.labels.length, 30);
    kMeans.k = 3;
    kMeans.fit(points);
    assert.equal(kMeans.centroids.length, 3 * dimension);
    assert.equal(kMeans.labels.length, points.length / dimension);
  });

  test("throws on invalid k and centroids", () => {
    assert.throws(
      () => new KMeans({ dimension: 2, k: 4 }).fit([0, 0, 1, 1]),
      RangeError,
    );
    assert.throws(
      () =>
        new KMeans({ dimension: 2, k: 2 }).fit([0, 0, 1, 1, 2, 2], {
          centroids: [0, 0],
        }),
      RangeError,
    );
  });
});

describe("MiniBatchKMeans", () => {
  const { points, dimension } = data.blobs3;

  test("approaches the k-means cost", () => {
    const kMeans = new KMeans({ dimension, k: 5, random: seeded(1) }).fit(
      points,
    );
    const miniBatch = new MiniBatchKMeans({
      dimension,
      k: 5,
      batchSize: 64,
      random: seeded(1),
    }).fit(points);
    assert.ok(
      miniBatch.cost < kMeans.cost * 1.05,
      `${miniBatch.cost} vs ${kMeans.cost}`,
    );
  });

  test("stops without improvement of the smoothed cost", () => {
    const count = points.length / dimension;
    const maxSteps = Math.floor((100 * count) / 64);
    const miniBatch = new MiniBatchKMeans({
      dimension,
      k: 5,
      batchSize: 64,
      random: seeded(1),
    }).fit(points);
    assert.ok(miniBatch.converged);
    assert.ok(miniBatch.steps < maxSteps);
    assert.equal(
      miniBatch.iterations,
      Math.ceil((miniBatch.steps * 64) / count),
    );
  });

  test("runs maxIterations passes without early stopping", () => {
    const count = points.length / dimension;
    const miniBatch = new MiniBatchKMeans({
      dimension,
      k: 5,
      batchSize: 64,
      maxIterations: 3,
      maxNoImprovement: 0,
      random: seeded(1),
    }).fit(points);
    assert.equal(miniBatch.steps, Math.floor((3 * count) / 64));
    assert.equal(miniBatch.iterations, 3);
    assert.ok(!miniBatch.converged);
  });

  test("initializes on a subset of the points", () => {
    const miniBatch = new MiniBatchKMeans({
      dimension,
      k: 5,
      batchSize: 64,
      initSize: 20,
      maxIterations: 0,
      random: seeded(1),
    }).fit(points);
    for (let c = 0; c < 5; c++) {
      const centroid = miniBatch.centroids.subarray(
        c * dimension,
        (c + 1) * dimension,
      );
      assert.ok(
        points.some((_, i) =>
          i % dimension ? false : centroid.every((x, a) => x === points[i + a]),
        ),
      );
    }
    assert.throws(
      () => new MiniBatchKMeans({ dimension, k: 5, initSize: 4 }).fit(points),
      RangeError,
    );
    assert.doesNotThrow(() =>
      new MiniBatchKMeans({ dimension, k: 5, batchSize: 1 }).fit(points),
    );
  });

  test("initializes on a weighted subset", () => {
    const count = points.length / dimension;
    const even = Float64Array.from({ length: count }, (_, i) =>
      i % 2 ? 0 : 1,
    );
    const miniBatch = new MiniBatchKMeans({
      dimension,
      k: 5,
      batchSize: 64,
      initSize: 20,
      maxIterations: 0,
      random: seeded(1),
    }).fit(points, { weights: even });
    for (let c = 0; c < 5; c++) {
      const centroid = miniBatch.centroids.subarray(
        c * dimension,
        (c + 1) * dimension,
      );
      assert.ok(
        points.some((_, i) =>
          i % (2 * dimension)
            ? false
            : centroid.every((x, a) => x === points[i + a]),
        ),
      );
    }
  });

  test("stops within tolerance", () => {
    const miniBatch = new MiniBatchKMeans({
      dimension,
      k: 5,
      tolerance: 1e-2,
      random: seeded(2),
    }).fit(points);
    assert.ok(miniBatch.converged);
    assert.ok(miniBatch.iterations < 100);
    assert.ok(miniBatch.steps > 0);
  });

  test("ignores points of weight 0", () => {
    const zero = Float64Array.from(
      { length: points.length / dimension },
      (_, i) => (i % 2 ? 0 : 1),
    );
    const miniBatch = new MiniBatchKMeans({
      dimension,
      k: 5,
      random: seeded(1),
    }).fit(points, {
      weights: zero,
    });
    assert.ok(miniBatch.centroids.every(Number.isFinite));
  });
});

describe("BisectingKMeans", () => {
  const { points, dimension } = data.blobs3;

  for (const strategy of ["cost", "size"]) {
    test(`splits into k clusters by ${strategy}`, () => {
      const bisecting = new BisectingKMeans({
        dimension,
        k: 5,
        strategy,
        trials: 3,
        random: seeded(1),
      }).fit(points);
      assert.ok(bisecting.sizes.every((size) => size > 0));
      assert.equal(
        bisecting.sizes.reduce((a, b) => a + b),
        points.length / dimension,
      );
      assert.deepEqual(
        differences(
          [bisecting.cost],
          [
            metrics.inertia(points, bisecting.centroids, bisecting.labels, {
              dimension,
            }),
          ],
        ),
        [],
      );
      const kMeans = new KMeans({ dimension, k: 5, random: seeded(1) }).fit(
        points,
      );
      assert.ok(bisecting.cost < kMeans.cost * 1.5);
    });
  }

  test("sizes are total weights", () => {
    const counts = Float64Array.from(
      { length: points.length / dimension },
      (_, i) => 1 + (i % 4),
    );
    const bisecting = new BisectingKMeans({
      dimension,
      k: 5,
      random: seeded(1),
    }).fit(points, { weights: counts });
    assert.equal(
      bisecting.sizes.reduce((a, b) => a + b),
      counts.reduce((a, b) => a + b),
    );
  });

  test("stops at the number of distinct points", () => {
    const bisecting = new BisectingKMeans({
      dimension: 2,
      k: 4,
      random: seeded(1),
    }).fit([0, 0, 0, 0, 1, 1, 1, 1]);
    assert.deepEqual([...bisecting.sizes], [2, 2, 0, 0]);
    assert.equal(bisecting.cost, 0);
  });
});

describe("KMedians", () => {
  test("matches numpy median iterations", () => {
    const expected = reference.kmedians;
    const { points, dimension } = data[expected.data];
    const kMedians = new KMedians({ dimension, k: expected.k }).fit(points, {
      centroids: expected.init,
    });
    assert.deepEqual([...kMedians.labels], expected.labels);
    assert.deepEqual(differences(kMedians.centroids, expected.centroids), []);
    assert.deepEqual(differences([kMedians.cost], [expected.cost]), []);
    assert.equal(kMedians.iterations, expected.iterations);
  });

  test("integer weights match repeated points", () => {
    const { points, dimension } = data.blobs2;
    const counts = Float64Array.from(
      { length: points.length / dimension },
      (_, i) => 1 + (i % 4),
    );
    const centroids = points.slice(0, 4 * dimension);
    const weighted = new KMedians({ dimension, k: 4 }).fit(points, {
      weights: counts,
      centroids,
    });
    const repeated = new KMedians({ dimension, k: 4 }).fit(
      repeat(points, dimension, counts),
      { centroids },
    );
    assert.deepEqual(differences(weighted.centroids, repeated.centroids), []);
    assert.deepEqual(differences([weighted.cost], [repeated.cost]), []);
  });

  test("relocates an empty cluster", () => {
    const { points } = data.blobs2;
    const kMedians = new KMedians({ dimension: 2, k: 3 }).fit(points, {
      centroids: [...points.subarray(0, 4), 100, 100],
    });
    assert.ok(kMedians.sizes.every((size) => size > 0));
  });

  test("keeps an empty cluster centroid", () => {
    const { points } = data.blobs2;
    const kMedians = new KMedians({
      dimension: 2,
      k: 3,
      emptyClusters: "keep",
    }).fit(points, { centroids: [...points.subarray(0, 4), 100, 100] });
    assert.equal(kMedians.sizes[2], 0);
    assert.deepEqual([...kMedians.centroids.subarray(4)], [100, 100]);
  });

  test("splits an even weight between the two middle values", () => {
    const kMedians = new KMedians({ dimension: 1, k: 1 });
    assert.deepEqual([...kMedians.fit([0, 1, 2, 10]).centroids], [1.5]);
    assert.deepEqual(
      [...kMedians.fit([0, 1, 2], { weights: [1, 1, 2] }).centroids],
      [1.5],
    );
  });
});

describe("KMedoids", () => {
  const DISTANCES = {
    euclidean: distance.euclidean,
    cityblock: distance.manhattan,
  };

  for (const expected of reference.kmedoids) {
    const from = expected.build ? "BUILD" : "given medoids";
    test(`${expected.algorithm} from ${from} matches kmedoids (${expected.metric})`, () => {
      const { points, dimension } = data[expected.data];
      const kMedoids = new KMedoids({
        dimension,
        k: expected.k,
        algorithm: expected.algorithm,
        distance: DISTANCES[expected.metric],
      }).fit(points, { medoids: expected.init });
      assert.deepEqual([...kMedoids.medoids], expected.medoids);
      assert.deepEqual([...kMedoids.labels], expected.labels);
      assert.deepEqual(differences([kMedoids.cost], [expected.cost]), []);
      assert.equal(kMedoids.iterations, expected.iterations);
      assert.equal(kMedoids.swaps, expected.swaps);
    });
  }

  test("accepts a dissimilarity matrix", () => {
    const expected = reference.kmedoids[0];
    const { points, dimension } = data[expected.data];
    const count = points.length / dimension;
    const dissimilarities = new Float64Array(count * count);
    for (let i = 0; i < count; i++) {
      for (let j = 0; j < count; j++) {
        dissimilarities[i * count + j] = distance.euclidean(
          points,
          i * dimension,
          points,
          j * dimension,
          dimension,
        );
      }
    }
    const kMedoids = new KMedoids({ dimension, k: expected.k }).fit(points, {
      medoids: expected.init,
      dissimilarities,
    });
    assert.deepEqual([...kMedoids.medoids], expected.medoids);
    assert.throws(
      () =>
        kMedoids.fit(points, {
          dissimilarities: dissimilarities.subarray(1),
        }),
      RangeError,
    );
  });

  test("centroids are the medoid points", () => {
    const { points, dimension } = data.blobs2;
    const kMedoids = new KMedoids({ dimension, k: 4 }).fit(points);
    assert.deepEqual(
      [...kMedoids.centroids],
      [...kMedoids.medoids].flatMap((m) => [
        ...points.subarray(m * dimension, (m + 1) * dimension),
      ]),
    );
  });

  for (const initialization of [
    init.build,
    init.kMeansPlusPlus,
    init.randomPoints,
    init.farthestFirst,
  ]) {
    test(`${initialization.name} initialization gives k distinct medoids`, () => {
      const { points, dimension } = data.blobs2;
      const kMedoids = new KMedoids({
        dimension,
        k: 4,
        init: initialization,
        random: seeded(1),
      }).fit(points);
      assert.equal(new Set(kMedoids.medoids).size, 4);
      assert.ok(kMedoids.converged);
    });
  }

  test("handles a single cluster and duplicates", () => {
    const single = new KMedoids({ dimension: 2, k: 1 }).fit([0, 0, 1, 0, 5, 0]);
    assert.deepEqual([...single.medoids], [1]);
    assert.equal(single.cost, 5);
    const duplicates = new KMedoids({ dimension: 2, k: 3 }).fit([
      0, 0, 0, 0, 1, 1, 1, 1,
    ]);
    assert.equal(new Set(duplicates.medoids).size, 3);
    assert.equal(duplicates.cost, 0);
  });
});

describe("KCenter", () => {
  test("matches the farthest-first traversal", () => {
    const expected = reference.kcenter;
    const { points, dimension } = data[expected.data];
    const kCenter = new KCenter({ dimension, k: expected.k }).fit(points, {
      first: expected.first,
    });
    assert.deepEqual([...kCenter.centers], expected.centers);
    assert.deepEqual([...kCenter.labels], expected.labels);
    assert.deepEqual(differences([kCenter.cost], [expected.cost]), []);
  });

  test("covers every point within the radius", () => {
    const { points, dimension } = data.blobs2;
    const kCenter = new KCenter({ dimension, k: 6, random: seeded(1) }).fit(
      points,
    );
    const distances = Array.from(kCenter.labels, (label, i) =>
      distance.euclidean(
        points,
        i * dimension,
        kCenter.centroids,
        label * dimension,
        dimension,
      ),
    );
    assert.ok(Math.max(...distances) <= kCenter.cost);
  });
});

describe("FuzzyCMeans", () => {
  for (const expected of reference.fuzzy) {
    test(`matches scikit-fuzzy with fuzziness ${expected.fuzziness}`, () => {
      const { points, dimension } = data[expected.data];
      const fuzzy = new FuzzyCMeans({
        dimension,
        k: expected.k,
        fuzziness: expected.fuzziness,
        tolerance: 1e-8,
        maxIterations: 1000,
      }).fit(points, { centroids: expected.init });
      assert.deepEqual(
        differences(fuzzy.centroids, expected.centroids, 1e-8),
        [],
      );
      assert.deepEqual(
        differences(fuzzy.memberships, expected.memberships, 1e-8),
        [],
      );
      assert.equal(fuzzy.iterations, expected.iterations);
    });
  }

  test("throws on fuzziness of 1 or less", () => {
    const { points, dimension } = data.blobs3;
    for (const fuzziness of [1, 0.5, NaN]) {
      assert.throws(
        () => new FuzzyCMeans({ dimension, k: 3, fuzziness }).fit(points),
        RangeError,
      );
    }
  });

  test("memberships sum to 1 and label the highest", () => {
    const { points, dimension } = data.blobs3;
    const k = 5;
    const fuzzy = new FuzzyCMeans({ dimension, k, random: seeded(1) }).fit(
      points,
    );
    const rows = Array.from({ length: fuzzy.count }, (_, i) =>
      fuzzy.memberships.subarray(i * k, (i + 1) * k),
    );
    assert.deepEqual(
      differences(
        rows.map((row) => row.reduce((a, b) => a + b)),
        Array.from({ length: fuzzy.count }, () => 1),
      ),
      [],
    );
    assert.deepEqual(
      [...fuzzy.labels],
      rows.map((row) => row.indexOf(Math.max(...row))),
    );
  });
});

describe("init", () => {
  const { points, dimension } = data.blobs3;
  const count = points.length / dimension;

  test("randomPoints picks distinct points", () => {
    const indices = init.randomPoints(points, 50, {
      dimension,
      random: seeded(1),
    });
    assert.equal(new Set(indices).size, 50);
    assert.ok(indices.every((i) => i < count));
    assert.equal(
      init.randomPoints(points, count, { dimension, random: seeded(1) }).length,
      count,
    );
    assert.equal(
      new Set(
        init.randomPoints(points, count, { dimension, random: seeded(1) }),
      ).size,
      count,
    );
  });

  for (const localTrials of [1, undefined]) {
    test(`kMeansPlusPlus never picks a selected point twice (localTrials=${localTrials})`, () => {
      const indices = init.kMeansPlusPlus([0, 0, 0, 0, 1, 1, 2, 2, 3, 3], 4, {
        dimension: 2,
        random: seeded(1),
        localTrials,
      });
      // Points 0 and 1 are duplicates: either can be picked.
      assert.deepEqual(
        [...indices].map((i) => (i < 2 ? 0 : i)).sort((a, b) => a - b),
        [0, 2, 3, 4],
      );
    });
  }

  test("kMeansPlusPlus skips points of weight 0", () => {
    const indices = init.kMeansPlusPlus(points, 5, {
      dimension,
      random: seeded(2),
      weights: Float64Array.from({ length: count }, (_, i) => (i % 2 ? 0 : 1)),
    });
    assert.ok(indices.every((i) => i % 2 === 0));
  });

  test("farthestFirst starts from the given point", () => {
    const indices = init.farthestFirst([0, 0, 1, 0, 10, 0, 4, 0], 3, {
      dimension: 2,
      first: 1,
    });
    assert.deepEqual([...indices], [1, 2, 3]);
  });

  for (const expected of reference.kmedoids.filter(({ build }) => build)) {
    test(`build matches kmedoids pam_build (${expected.metric})`, () => {
      const { points, dimension } = data[expected.data];
      const metric = {
        euclidean: distance.euclidean,
        cityblock: distance.manhattan,
      }[expected.metric];
      const count = points.length / dimension;
      const dissimilarities = new Float64Array(count * count);
      for (let i = 0; i < count; i++) {
        for (let j = 0; j < count; j++) {
          dissimilarities[i * count + j] = metric(
            points,
            i * dimension,
            points,
            j * dimension,
            dimension,
          );
        }
      }
      for (const options of [{ distance: metric }, { dissimilarities }]) {
        const indices = init.build(points, expected.k, {
          dimension,
          ...options,
        });
        assert.deepEqual([...indices], expected.build);
      }
    });
  }

  test("accepts a custom init returning its own indices", () => {
    const points = [0, 0, 1, 0, 10, 0, 11, 0];
    const kMeans = new KMeans({ dimension: 2, k: 2, init: () => [0, 3] }).fit(
      points,
    );
    assert.deepEqual([...kMeans.labels], [0, 0, 1, 1]);
    const kMedoids = new KMedoids({
      dimension: 2,
      k: 2,
      init: () => [1, 2],
    }).fit(points);
    assert.equal(new Set(kMedoids.medoids).size, 2);
  });
});

describe("metrics", () => {
  const expected = reference.metrics;
  const { points, dimension } = data[expected.data];
  const labels = Int32Array.from(expected.labels);

  test("silhouette matches scikit-learn", () => {
    const out = new Float64Array(labels.length);
    assert.deepEqual(
      differences(
        [metrics.silhouette(points, labels, { dimension, out })],
        [expected.silhouette],
      ),
      [],
    );
    assert.deepEqual(differences(out, expected.silhouetteSamples), []);
    assert.deepEqual(
      differences(
        [
          metrics.silhouette(points, labels, {
            dimension,
            distance: distance.manhattan,
          }),
        ],
        [expected.silhouetteManhattan],
      ),
      [],
    );
  });

  test("daviesBouldin matches scikit-learn", () => {
    assert.deepEqual(
      differences(
        [metrics.daviesBouldin(points, labels, { dimension })],
        [expected.daviesBouldin],
      ),
      [],
    );
  });

  test("calinskiHarabasz matches scikit-learn", () => {
    assert.deepEqual(
      differences(
        [metrics.calinskiHarabasz(points, labels, { dimension })],
        [expected.calinskiHarabasz],
      ),
      [],
    );
  });

  test("inertia matches the k-means cost", () => {
    const kMeans = new KMeans({ dimension, k: 5, random: seeded(1) }).fit(
      points,
      { weights: undefined },
    );
    assert.deepEqual(
      differences(
        [
          metrics.inertia(points, kMeans.centroids, kMeans.labels, {
            dimension,
          }),
        ],
        [kMeans.cost],
      ),
      [],
    );
  });

  test("ignore empty clusters", () => {
    const gaps = labels.map((label) => 2 * label);
    for (const metric of ["silhouette", "daviesBouldin", "calinskiHarabasz"]) {
      assert.deepEqual(
        differences(
          [metrics[metric](points, gaps, { dimension })],
          [metrics[metric](points, labels, { dimension })],
        ),
        [],
      );
    }
  });

  test("silhouette is 0 for a single cluster and a singleton", () => {
    const out = new Float64Array(labels.length);
    const single = new Int32Array(labels.length);
    assert.equal(metrics.silhouette(points, single, { dimension, out }), 0);
    const singleton = labels.map((label, i) =>
      i ? label : Math.max(...labels) + 1,
    );
    metrics.silhouette(points, singleton, { dimension, out });
    assert.equal(out[0], 0);
  });

  test("score duplicate points as scikit-learn", () => {
    const duplicates = [0, 0, 0, 0];
    const split = [0, 0, 1, 1];
    assert.equal(metrics.silhouette(duplicates, split, { dimension: 1 }), 0);
    assert.equal(
      metrics.calinskiHarabasz([0, 0, 1, 1], split, { dimension: 1 }),
      1,
    );
  });
});

describe("distance", () => {
  test("measures between offsets", () => {
    const a = [9, 0, 0];
    const b = [3, 4, 9];
    assert.equal(distance.squaredEuclidean(a, 1, b, 0, 2), 25);
    assert.equal(distance.euclidean(a, 1, b, 0, 2), 5);
    assert.equal(distance.manhattan(a, 1, b, 0, 2), 7);
    assert.equal(distance.chebyshev(a, 1, b, 0, 2), 4);
    assert.deepEqual(
      differences([distance.cosine([1, 0], 0, [0, 2], 0, 2)], [1]),
      [],
    );
    assert.deepEqual(
      differences([distance.cosine([1, 1], 0, [2, 2], 0, 2)], [0]),
      [],
    );
    assert.equal(distance.cosine([0, 0], 0, [1, 0], 0, 2), 1);
  });
});
