import { KdTree } from "spatial-partition";

import {
  KMeans,
  MiniBatchKMeans,
  BisectingKMeans,
  KMedians,
  KMedoids,
  KCenter,
  FuzzyCMeans,
} from "k-clustering";

// Tolerance 0 runs k-means to convergence: algorithms differ by speed only.
// Quadratic or k-dependent methods are skipped above about a second of work.
export const ALGORITHMS = {
  "k-means": {
    create: (options, config) =>
      new KMeans({
        ...options,
        ...(config.algorithm === "kdTree"
          ? {
              algorithm: "lloyd",
              index: new KdTree({ dimension: options.dimension }),
            }
          : { algorithm: config.algorithm }),
        tolerance: 0,
      }),
  },
  "spherical k-means": {
    create: (options) => new KMeans({ ...options, spherical: true }),
  },
  "mini-batch k-means": {
    create: (options) => new MiniBatchKMeans({ ...options, batchSize: 64 }),
  },
  "bisecting k-means": { create: (options) => new BisectingKMeans(options) },
  "k-medians": { create: (options) => new KMedians(options) },
  "k-medoids": {
    create: (options) => new KMedoids(options),
    skip: ({ count, k }) => count * count * k > 2e8,
  },
  "k-center": { create: (options) => new KCenter(options) },
  "fuzzy c-means": {
    create: (options) => new FuzzyCMeans(options),
    skip: ({ count, k }) => count * k > 1e5,
  },
};
