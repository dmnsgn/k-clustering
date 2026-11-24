# k-clustering

[![npm version](https://img.shields.io/npm/v/k-clustering)](https://www.npmjs.com/package/k-clustering)
[![stability-stable](https://img.shields.io/badge/stability-stable-green.svg)](https://www.npmjs.com/package/k-clustering)
[![npm minzipped size](https://img.shields.io/bundlephobia/minzip/k-clustering)](https://bundlephobia.com/package/k-clustering)
[![dependencies](https://img.shields.io/librariesio/release/npm/k-clustering)](https://github.com/dmnsgn/k-clustering/blob/main/package.json)
[![types](https://img.shields.io/npm/types/k-clustering)](https://github.com/microsoft/TypeScript)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-fa6673.svg)](https://conventionalcommits.org)
[![styled with prettier](https://img.shields.io/badge/styled_with-Prettier-f8bc45.svg?logo=prettier)](https://github.com/prettier/prettier)
[![linted with eslint](https://img.shields.io/badge/linted_with-ES_Lint-4B32C3.svg?logo=eslint)](https://github.com/eslint/eslint)
[![license](https://img.shields.io/github/license/dmnsgn/k-clustering)](https://github.com/dmnsgn/k-clustering/blob/main/LICENSE.md)

Center-based clustering of flat point arrays: k-means (Lloyd, Elkan, Hamerly, spherical, mini-batch, bisecting), k-medians, k-medoids (FasterPAM, PAM, Alternate), k-center and fuzzy c-means, with seeding and validation metrics.

[![paypal](https://img.shields.io/badge/donate-paypal-informational?logo=paypal)](https://paypal.me/dmnsgn)
[![coinbase](https://img.shields.io/badge/donate-coinbase-informational?logo=coinbase)](https://commerce.coinbase.com/checkout/56cbdf28-e323-48d8-9c98-7019e72c97f3)
[![twitter](https://img.shields.io/twitter/follow/dmnsgn?style=social)](https://twitter.com/dmnsgn)
[![bluesky](https://img.shields.io/badge/-blue?logo=bluesky&label=Follow%20%40dmnsgn.me&style=social)](https://bsky.app/profile/dmnsgn.me)

[![k-clustering screenshot](https://raw.githubusercontent.com/dmnsgn/k-clustering/main/screenshot.png)](https://dmnsgn.github.io/k-clustering/)

## Installation

```bash
npm install k-clustering
```

## Features

| Class             | Objective                            | Centers        | Iteration        | Best when                                     |
| ----------------- | ------------------------------------ | -------------- | ---------------- | --------------------------------------------- |
| `KMeans`          | Sum of squared Euclidean distances   | Means          | O(k n), or fewer | Compact clusters, color quantization, Voronoi |
| `MiniBatchKMeans` | Sum of squared Euclidean distances   | Means          | O(k b) per batch | Large inputs, approximate clusters            |
| `BisectingKMeans` | Sum of squared Euclidean distances   | Means          | O(n) per split   | Many clusters, hierarchical splits            |
| `KMedians`        | Sum of Manhattan distances           | Medians        | O(k n + n log n) | Outliers                                      |
| `KMedoids`        | Sum of distances, any dissimilarity  | Input points   | O(n²)            | Custom distances, actual points as centers    |
| `KCenter`         | Largest distance                     | Input points   | O(k n) in total  | Coverage, sampling, spreading points apart    |
| `FuzzyCMeans`     | Membership weighted squared distance | Weighted means | O(k n)           | Overlapping clusters, soft boundaries         |

`KMeans` options:

- `algorithm`: the same clusters at different speeds. `"hamerly"` (default) is fastest in low dimension, `"elkan"` in high dimension with many clusters: both use the triangle inequality to skip distance computations. `"lloyd"` uses no extra memory and is required by `index`.
- `spherical`: cluster directions by cosine distance.
- `index`: with `"lloyd"`, a nearest neighbour structure rebuilt over the centroids, such as a [spatial-partition](https://github.com/dmnsgn/spatial-partition) `KdTree`, for hundreds of clusters.

Initializations (`init` namespace), passed as the `init` option: `kMeansPlusPlus` D² sampling with greedy local trials (default), `randomPoints` (Forgy), `farthestFirst` traversal (maximin) and `build` (PAM BUILD, default of k-medoids). Any function returning `k` point indices works, eg. `(points, k, options) => kMeansPlusPlus(points, k, { ...options, localTrials: 1 })`.

Metrics (`metrics` namespace): inertia, silhouette, Davies–Bouldin and Calinski–Harabasz.

Helpers (`utils` namespace): the building blocks shared by the clusterings, such as `normalize` and `nearestCentroid`.

All structures are allocation-free once sized: buffers are reused across fits.

## Usage

```js
import { KMeans, KMedoids, metrics, distance } from "k-clustering";

// 4 colors, stored flat: r, g, b, r, g, b...
// prettier-ignore
const points = new Float32Array([
  0.9, 0.1, 0.1,
  1, 0.2, 0.1,
  0.1, 0.2, 0.9,
  0.2, 0.2, 1,
]);

// Find 2 clusters. Call fit again whenever the points change.
const kMeans = new KMeans({
  dimension: 3, // Default. 2 for x, y, x, y...
  k: 2,
  random: Math.random, // Default. A seeded generator gives reproducible results.
}).fit(points);
kMeans.labels; // [0, 0, 1, 1]: cluster of each point, in any order
kMeans.centroids; // [0.95, 0.15, 0.1, 0.15, 0.2, 0.95]: flat too
kMeans.predict([0.8, 0, 0.2]); // 0: cluster of the nearest centroid to a new point

// Refit with weighted points, starting from given centroids
kMeans.fit(points, { weights: [1, 1, 4, 1], centroids: [1, 0, 0, 0, 0, 1] });
kMeans.centroids; // [0.95, 0.15, 0.1, 0.12, 0.2, 0.92]

// How well the clusters are separated, up to 1: compare across k to pick one
metrics.silhouette(points, kMeans.labels); // 0.879

// Centers are input points, with any distance
const kMedoids = new KMedoids({ k: 2, distance: distance.manhattan });
kMedoids.fit(points).medoids; // [0, 2]: point indices
```

**Use cases**

`KMeans`

- Dominant colors and color quantization
- Centroidal Voronoi tessellations and stippling (Lloyd relaxation, weighted by image darkness)
- Grouping markers, particles or vertices
- Vector quantization of embeddings (`spherical`)

`MiniBatchKMeans`

- Palettes of large images without downscaling
- Clustering millions of points

`BisectingKMeans`

- Hierarchical palettes and levels of detail
- Many clusters with a stable split order

`KMedians`

- Data with outliers, such as noisy sensor readings
- Grid and city-block distances

`KMedoids`

- Clusters of strings, graphs or any custom dissimilarity: fit point indices in 1D with a precomputed `dissimilarities` matrix
- Picking representative samples: centers are real points

`KCenter`

- Facility placement and coverage
- Farthest point sampling of meshes and point clouds
- Spreading out a few distinct colors

`FuzzyCMeans`

- Soft segmentation and gradients between clusters
- Points shared by several groups

## API

<!-- api-start -->

## Modules

<dl>
<dt><a href="#module_kClustering">kClustering</a></dt>
<dd></dd>
<dt><a href="#module_bisectingKMeans">bisectingKMeans</a></dt>
<dd></dd>
<dt><a href="#module_clustering">clustering</a></dt>
<dd></dd>
<dt><a href="#module_distance">distance</a></dt>
<dd></dd>
<dt><a href="#module_fuzzyCMeans">fuzzyCMeans</a></dt>
<dd></dd>
<dt><a href="#module_init">init</a></dt>
<dd></dd>
<dt><a href="#module_kCenter">kCenter</a></dt>
<dd></dd>
<dt><a href="#module_kMeans">kMeans</a></dt>
<dd></dd>
<dt><a href="#module_kMedians">kMedians</a></dt>
<dd></dd>
<dt><a href="#module_kMedoids">kMedoids</a></dt>
<dd></dd>
<dt><a href="#module_metrics">metrics</a></dt>
<dd></dd>
<dt><a href="#module_miniBatchKMeans">miniBatchKMeans</a></dt>
<dd></dd>
<dt><a href="#module_utils">utils</a></dt>
<dd></dd>
</dl>

<a name="module_kClustering"></a>

## kClustering

<a name="module_bisectingKMeans"></a>

## bisectingKMeans

- [bisectingKMeans](#module_bisectingKMeans)
  - _static_
    - [.BisectingKMeans](#module_bisectingKMeans.BisectingKMeans) ⇐ <code>Clustering</code>
      - [new BisectingKMeans([options])](#new_module_bisectingKMeans.BisectingKMeans_new)
      - [.strategy](#module_bisectingKMeans.BisectingKMeans+strategy) : <code>&quot;cost&quot;</code> \| <code>&quot;size&quot;</code>
      - [.trials](#module_bisectingKMeans.BisectingKMeans+trials) : <code>number</code>
      - [.fit(points, [options])](#module_bisectingKMeans.BisectingKMeans+fit) ⇒ <code>this</code>
  - _inner_
    - [~BisectingKMeansOptions](#module_bisectingKMeans..BisectingKMeansOptions) : <code>object</code>
    - [~BisectingKMeansFitOptions](#module_bisectingKMeans..BisectingKMeansFitOptions) : <code>object</code>

<a name="module_bisectingKMeans.BisectingKMeans"></a>

### bisectingKMeans.BisectingKMeans ⇐ <code>Clustering</code>

Bisecting k-means clustering: start from one cluster and split one cluster in
two with k-means until there are `k`.

Deterministic cluster hierarchy, less sensitive to initialization and faster
than k-means with many clusters. Splits aren't refined afterwards: pass
`centroids` to `KMeans` for that.

Stops early with clusters of size 0 when points hold fewer than `k` distinct
values.

**Kind**: static class of [<code>bisectingKMeans</code>](#module_bisectingKMeans)
**Extends**: <code>Clustering</code>
**See**: Steinbach, M., Karypis, G., & Kumar, V. (2000). A comparison of document clustering techniques. KDD Workshop on Text Mining.

- [.BisectingKMeans](#module_bisectingKMeans.BisectingKMeans) ⇐ <code>Clustering</code>
  - [new BisectingKMeans([options])](#new_module_bisectingKMeans.BisectingKMeans_new)
  - [.strategy](#module_bisectingKMeans.BisectingKMeans+strategy) : <code>&quot;cost&quot;</code> \| <code>&quot;size&quot;</code>
  - [.trials](#module_bisectingKMeans.BisectingKMeans+trials) : <code>number</code>
  - [.fit(points, [options])](#module_bisectingKMeans.BisectingKMeans+fit) ⇒ <code>this</code>

<a name="new_module_bisectingKMeans.BisectingKMeans_new"></a>

#### new BisectingKMeans([options])

| Param     | Type                                                                                                                                                        |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [options] | <code>ClusteringOptions</code> \| <code>InitializedOptions</code> \| [<code>BisectingKMeansOptions</code>](#module_bisectingKMeans..BisectingKMeansOptions) |

<a name="module_bisectingKMeans.BisectingKMeans+strategy"></a>

#### bisectingKMeans.strategy : <code>&quot;cost&quot;</code> \| <code>&quot;size&quot;</code>

**Kind**: instance property of [<code>BisectingKMeans</code>](#module_bisectingKMeans.BisectingKMeans)
<a name="module_bisectingKMeans.BisectingKMeans+trials"></a>

#### bisectingKMeans.trials : <code>number</code>

**Kind**: instance property of [<code>BisectingKMeans</code>](#module_bisectingKMeans.BisectingKMeans)
<a name="module_bisectingKMeans.BisectingKMeans+fit"></a>

#### bisectingKMeans.fit(points, [options]) ⇒ <code>this</code>

**Kind**: instance method of [<code>BisectingKMeans</code>](#module_bisectingKMeans.BisectingKMeans)

| Param     | Type                                                                                         | Description                                            |
| --------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| points    | <code>Points</code>                                                                          | Flat array of point components, `dimension` per point. |
| [options] | [<code>BisectingKMeansFitOptions</code>](#module_bisectingKMeans..BisectingKMeansFitOptions) |                                                        |

<a name="module_bisectingKMeans..BisectingKMeansOptions"></a>

### bisectingKMeans~BisectingKMeansOptions : <code>object</code>

**Kind**: inner typedef of [<code>bisectingKMeans</code>](#module_bisectingKMeans)
**Properties**

| Name            | Type                                                           | Default                       | Description                                                                         |
| --------------- | -------------------------------------------------------------- | ----------------------------- | ----------------------------------------------------------------------------------- |
| [strategy]      | <code>&quot;cost&quot;</code> \| <code>&quot;size&quot;</code> | <code>&quot;cost&quot;</code> | Split the cluster of highest cost or of largest size. scikit-learn's default.       |
| [trials]        | <code>number</code>                                            | <code>1</code>                | Bisections tried per split, keeping the one of lowest cost. scikit-learn's default. |
| [maxIterations] | <code>number</code>                                            | <code>300</code>              | Iterations limit of each bisection. See `KMeans`.                                   |
| [tolerance]     | <code>number</code>                                            | <code>1e-4</code>             | Tolerance of each bisection. See `KMeans`.                                          |

<a name="module_bisectingKMeans..BisectingKMeansFitOptions"></a>

### bisectingKMeans~BisectingKMeansFitOptions : <code>object</code>

**Kind**: inner typedef of [<code>bisectingKMeans</code>](#module_bisectingKMeans)
**Properties**

| Name      | Type                 | Description                               |
| --------- | -------------------- | ----------------------------------------- |
| [weights] | <code>Weights</code> | Weight of each point. All 1 when omitted. |

<a name="module_clustering"></a>

## clustering

- [clustering](#module_clustering)
  - _static_
    - [.Clustering](#module_clustering.Clustering)
      - [new Clustering([options])](#new_module_clustering.Clustering_new)
      - [.dimension](#module_clustering.Clustering+dimension) : <code>number</code>
      - [.k](#module_clustering.Clustering+k) : <code>number</code>
      - [.random](#module_clustering.Clustering+random) : <code>function</code>
      - [.distance](#module_clustering.Clustering+distance) : <code>Distance</code>
      - [.init](#module_clustering.Clustering+init) : <code>Init</code>
      - [.centroids](#module_clustering.Clustering+centroids) : <code>Float64Array</code>
      - [.labels](#module_clustering.Clustering+labels) : <code>Int32Array</code>
      - [.sizes](#module_clustering.Clustering+sizes) : <code>Float64Array</code>
      - [.count](#module_clustering.Clustering+count) : <code>number</code>
      - [.cost](#module_clustering.Clustering+cost) : <code>number</code>
      - [.iterations](#module_clustering.Clustering+iterations) : <code>number</code>
      - [.converged](#module_clustering.Clustering+converged) : <code>boolean</code>
      - [._distances](#module_clustering.Clustering+_distances) : <code>Float64Array</code>
      - [._indices](#module_clustering.Clustering+_indices) : <code>Uint32Array</code>
      - _[.fit(points, [options])](#module_clustering.Clustering+fit) ⇒ <code>this</code>_
      - [.predict(position)](#module_clustering.Clustering+predict) ⇒ <code>number</code>
      - [.assign(points, [out])](#module_clustering.Clustering+assign) ⇒ <code>Int32Array</code>
      - [._prepare(points)](#module_clustering.Clustering+_prepare)
      - [._initIndices(points, indices, options)](#module_clustering.Clustering+_initIndices)
      - [._setCentroids(points, indices)](#module_clustering.Clustering+_setCentroids)
      - [._initCentroids(points, options, distance)](#module_clustering.Clustering+_initCentroids)
      - [._iterate(maxIterations, tolerance, assign, update)](#module_clustering.Clustering+_iterate)
      - [._tally(points, [weights])](#module_clustering.Clustering+_tally) ⇒ <code>number</code>
  - _inner_
    - [~Points](#module_clustering..Points) : <code>Float32Array</code> \| <code>Float64Array</code> \| <code>Array.&lt;number&gt;</code>
    - [~Position](#module_clustering..Position) : <code>ArrayLike.&lt;number&gt;</code>
    - [~Weights](#module_clustering..Weights) : <code>Float32Array</code> \| <code>Float64Array</code> \| <code>Array.&lt;number&gt;</code>
    - [~EmptyClusters](#module_clustering..EmptyClusters) : <code>&quot;farthest&quot;</code> \| <code>&quot;keep&quot;</code>
    - [~ClusteringOptions](#module_clustering..ClusteringOptions) : <code>object</code>
    - [~InitializedOptions](#module_clustering..InitializedOptions) : <code>object</code>
    - [~BaseOptions](#module_clustering..BaseOptions) : <code>object</code>
    - [~FitOptions](#module_clustering..FitOptions) : <code>object</code>

<a name="module_clustering.Clustering"></a>

### clustering.Clustering

Shared state and nearest centroid queries of all clusterings. Subclasses
implement `fit`.

**Kind**: static class of [<code>clustering</code>](#module_clustering)

- [.Clustering](#module_clustering.Clustering)
  - [new Clustering([options])](#new_module_clustering.Clustering_new)
  - [.dimension](#module_clustering.Clustering+dimension) : <code>number</code>
  - [.k](#module_clustering.Clustering+k) : <code>number</code>
  - [.random](#module_clustering.Clustering+random) : <code>function</code>
  - [.distance](#module_clustering.Clustering+distance) : <code>Distance</code>
  - [.init](#module_clustering.Clustering+init) : <code>Init</code>
  - [.centroids](#module_clustering.Clustering+centroids) : <code>Float64Array</code>
  - [.labels](#module_clustering.Clustering+labels) : <code>Int32Array</code>
  - [.sizes](#module_clustering.Clustering+sizes) : <code>Float64Array</code>
  - [.count](#module_clustering.Clustering+count) : <code>number</code>
  - [.cost](#module_clustering.Clustering+cost) : <code>number</code>
  - [.iterations](#module_clustering.Clustering+iterations) : <code>number</code>
  - [.converged](#module_clustering.Clustering+converged) : <code>boolean</code>
  - [._distances](#module_clustering.Clustering+_distances) : <code>Float64Array</code>
  - [._indices](#module_clustering.Clustering+_indices) : <code>Uint32Array</code>
  - _[.fit(points, [options])](#module_clustering.Clustering+fit) ⇒ <code>this</code>_
  - [.predict(position)](#module_clustering.Clustering+predict) ⇒ <code>number</code>
  - [.assign(points, [out])](#module_clustering.Clustering+assign) ⇒ <code>Int32Array</code>
  - [._prepare(points)](#module_clustering.Clustering+_prepare)
  - [._initIndices(points, indices, options)](#module_clustering.Clustering+_initIndices)
  - [._setCentroids(points, indices)](#module_clustering.Clustering+_setCentroids)
  - [._initCentroids(points, options, distance)](#module_clustering.Clustering+_initCentroids)
  - [._iterate(maxIterations, tolerance, assign, update)](#module_clustering.Clustering+_iterate)
  - [._tally(points, [weights])](#module_clustering.Clustering+_tally) ⇒ <code>number</code>

<a name="new_module_clustering.Clustering_new"></a>

#### new Clustering([options])

| Param     | Type                                                                                                                                                                                                                |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [options] | [<code>ClusteringOptions</code>](#module_clustering..ClusteringOptions) \| [<code>InitializedOptions</code>](#module_clustering..InitializedOptions) \| [<code>BaseOptions</code>](#module_clustering..BaseOptions) |

<a name="module_clustering.Clustering+dimension"></a>

#### clustering.dimension : <code>number</code>

Number of components per point.

**Kind**: instance property of [<code>Clustering</code>](#module_clustering.Clustering)
<a name="module_clustering.Clustering+k"></a>

#### clustering.k : <code>number</code>

Number of clusters.

**Kind**: instance property of [<code>Clustering</code>](#module_clustering.Clustering)
<a name="module_clustering.Clustering+random"></a>

#### clustering.random : <code>function</code>

Uniform random number in [0, 1).

**Kind**: instance property of [<code>Clustering</code>](#module_clustering.Clustering)
<a name="module_clustering.Clustering+distance"></a>

#### clustering.distance : <code>Distance</code>

Distance assigning points to centroids.

**Kind**: instance property of [<code>Clustering</code>](#module_clustering.Clustering)
<a name="module_clustering.Clustering+init"></a>

#### clustering.init : <code>Init</code>

Centroids initialization.

**Kind**: instance property of [<code>Clustering</code>](#module_clustering.Clustering)
<a name="module_clustering.Clustering+centroids"></a>

#### clustering.centroids : <code>Float64Array</code>

Flat cluster centers of the last fit, `k * dimension` components.

**Kind**: instance property of [<code>Clustering</code>](#module_clustering.Clustering)
<a name="module_clustering.Clustering+labels"></a>

#### clustering.labels : <code>Int32Array</code>

Cluster of each point of the last fit.

**Kind**: instance property of [<code>Clustering</code>](#module_clustering.Clustering)
<a name="module_clustering.Clustering+sizes"></a>

#### clustering.sizes : <code>Float64Array</code>

Sum of the point weights of each cluster: point counts without weights. 0
for an empty cluster.

**Kind**: instance property of [<code>Clustering</code>](#module_clustering.Clustering)
<a name="module_clustering.Clustering+count"></a>

#### clustering.count : <code>number</code>

Number of points of the last fit.

**Kind**: instance property of [<code>Clustering</code>](#module_clustering.Clustering)
<a name="module_clustering.Clustering+cost"></a>

#### clustering.cost : <code>number</code>

Objective minimized by the last fit. Lower is better.

**Kind**: instance property of [<code>Clustering</code>](#module_clustering.Clustering)
<a name="module_clustering.Clustering+iterations"></a>

#### clustering.iterations : <code>number</code>

Iterations run by the last fit.

**Kind**: instance property of [<code>Clustering</code>](#module_clustering.Clustering)
<a name="module_clustering.Clustering+converged"></a>

#### clustering.converged : <code>boolean</code>

Whether the last fit converged before its iterations limit.

**Kind**: instance property of [<code>Clustering</code>](#module_clustering.Clustering)
<a name="module_clustering.Clustering+_distances"></a>

#### clustering.\_distances : <code>Float64Array</code>

Scratch distances, one per point.

**Kind**: instance property of [<code>Clustering</code>](#module_clustering.Clustering)
**Access**: protected
<a name="module_clustering.Clustering+_indices"></a>

#### clustering.\_indices : <code>Uint32Array</code>

Selected point indices of the initialization.

**Kind**: instance property of [<code>Clustering</code>](#module_clustering.Clustering)
**Access**: protected
<a name="module_clustering.Clustering+fit"></a>

#### _clustering.fit(points, [options]) ⇒ <code>this</code>_

Cluster points. Buffers are reused across fits.

**Kind**: instance abstract method of [<code>Clustering</code>](#module_clustering.Clustering)

| Param     | Type                                              | Description                                            |
| --------- | ------------------------------------------------- | ------------------------------------------------------ |
| points    | [<code>Points</code>](#module_clustering..Points) | Flat array of point components, `dimension` per point. |
| [options] | <code>object</code>                               |                                                        |

<a name="module_clustering.Clustering+predict"></a>

#### clustering.predict(position) ⇒ <code>number</code>

Index of the nearest centroid of a single position, such as a new point:
the Voronoi cell containing it.

**Kind**: instance method of [<code>Clustering</code>](#module_clustering.Clustering)

| Param    | Type                                                  |
| -------- | ----------------------------------------------------- |
| position | [<code>Position</code>](#module_clustering..Position) |

<a name="module_clustering.Clustering+assign"></a>

#### clustering.assign(points, [out]) ⇒ <code>Int32Array</code>

Index of the nearest centroid of each point, such as labelling a full
dataset after fitting a sample. scikit-learn's `predict`.

**Kind**: instance method of [<code>Clustering</code>](#module_clustering.Clustering)

| Param  | Type                                              | Description                                         |
| ------ | ------------------------------------------------- | --------------------------------------------------- |
| points | [<code>Points</code>](#module_clustering..Points) | Flat array of point components.                     |
| [out]  | <code>Int32Array</code>                           | Receives the indices. Reuse it to avoid allocating. |

<a name="module_clustering.Clustering+_prepare"></a>

#### clustering.\_prepare(points)

Size the buffers for `points`.

**Kind**: instance method of [<code>Clustering</code>](#module_clustering.Clustering)
**Access**: protected

| Param  | Type                                              |
| ------ | ------------------------------------------------- |
| points | [<code>Points</code>](#module_clustering..Points) |

<a name="module_clustering.Clustering+_initIndices"></a>

#### clustering.\_initIndices(points, indices, options)

Select `k` points with `init` into `indices`.

**Kind**: instance method of [<code>Clustering</code>](#module_clustering.Clustering)
**Access**: protected

| Param   | Type                                              | Description                                                    |
| ------- | ------------------------------------------------- | -------------------------------------------------------------- |
| points  | [<code>Points</code>](#module_clustering..Points) |                                                                |
| indices | <code>Uint32Array</code>                          |                                                                |
| options | <code>InitOptions</code>                          | Distance of the objective and other options of the clustering. |

<a name="module_clustering.Clustering+_setCentroids"></a>

#### clustering.\_setCentroids(points, indices)

Copy the points at `indices` into `centroids`.

**Kind**: instance method of [<code>Clustering</code>](#module_clustering.Clustering)
**Access**: protected

| Param   | Type                                              |
| ------- | ------------------------------------------------- |
| points  | [<code>Points</code>](#module_clustering..Points) |
| indices | <code>ArrayLike.&lt;number&gt;</code>             |

<a name="module_clustering.Clustering+_initCentroids"></a>

#### clustering.\_initCentroids(points, options, distance)

Set the initial centroids: the given ones, or selected points.

**Kind**: instance method of [<code>Clustering</code>](#module_clustering.Clustering)
**Access**: protected

| Param    | Type                                                      | Description                                        |
| -------- | --------------------------------------------------------- | -------------------------------------------------- |
| points   | [<code>Points</code>](#module_clustering..Points)         |                                                    |
| options  | [<code>FitOptions</code>](#module_clustering..FitOptions) |                                                    |
| distance | <code>Distance</code>                                     | Distance of the objective, for the initialization. |

<a name="module_clustering.Clustering+_iterate"></a>

#### clustering.\_iterate(maxIterations, tolerance, assign, update)

Alternate assignments and centroid updates (Lloyd iterations) until no
label changes or the centroids shift by `tolerance` at most.

**Kind**: instance method of [<code>Clustering</code>](#module_clustering.Clustering)
**Access**: protected

| Param         | Type                  | Description                                                                            |
| ------------- | --------------------- | -------------------------------------------------------------------------------------- |
| maxIterations | <code>number</code>   |                                                                                        |
| tolerance     | <code>number</code>   | Absolute tolerance on the sum of squared centroid shifts. See `utils.scaledTolerance`. |
| assign        | <code>function</code> | Called with whether an update follows. Returns whether a label changed.                |
| update        | <code>function</code> | Returns the sum of squared centroid shifts.                                            |

<a name="module_clustering.Clustering+_tally"></a>

#### clustering.\_tally(points, [weights]) ⇒ <code>number</code>

Fill `sizes` from `labels`.

**Kind**: instance method of [<code>Clustering</code>](#module_clustering.Clustering)
**Returns**: <code>number</code> - Weighted sum of the distances of the points to their
centroid.
**Access**: protected

| Param     | Type                                                |
| --------- | --------------------------------------------------- |
| points    | [<code>Points</code>](#module_clustering..Points)   |
| [weights] | [<code>Weights</code>](#module_clustering..Weights) |

<a name="module_clustering..Points"></a>

### clustering~Points : <code>Float32Array</code> \| <code>Float64Array</code> \| <code>Array.&lt;number&gt;</code>

**Kind**: inner typedef of [<code>clustering</code>](#module_clustering)
<a name="module_clustering..Position"></a>

### clustering~Position : <code>ArrayLike.&lt;number&gt;</code>

**Kind**: inner typedef of [<code>clustering</code>](#module_clustering)
<a name="module_clustering..Weights"></a>

### clustering~Weights : <code>Float32Array</code> \| <code>Float64Array</code> \| <code>Array.&lt;number&gt;</code>

**Kind**: inner typedef of [<code>clustering</code>](#module_clustering)
<a name="module_clustering..EmptyClusters"></a>

### clustering~EmptyClusters : <code>&quot;farthest&quot;</code> \| <code>&quot;keep&quot;</code>

Empty cluster handling: move it to the point farthest from its centroid, as
in scikit-learn, or keep its centroid with a size of 0.

**Kind**: inner typedef of [<code>clustering</code>](#module_clustering)
<a name="module_clustering..ClusteringOptions"></a>

### clustering~ClusteringOptions : <code>object</code>

**Kind**: inner typedef of [<code>clustering</code>](#module_clustering)
**Properties**

| Name        | Type                  | Default                  | Description                                                                        |
| ----------- | --------------------- | ------------------------ | ---------------------------------------------------------------------------------- |
| [dimension] | <code>number</code>   | <code>3</code>           | Number of components per point.                                                    |
| [k]         | <code>number</code>   | <code>8</code>           | Number of clusters. scikit-learn's default.                                        |
| [random]    | <code>function</code> | <code>Math.random</code> | Uniform random number in [0, 1). Pass a seeded generator for reproducible results. |

<a name="module_clustering..InitializedOptions"></a>

### clustering~InitializedOptions : <code>object</code>

**Kind**: inner typedef of [<code>clustering</code>](#module_clustering)
**Properties**

| Name   | Type              | Default                     | Description                                                                                     |
| ------ | ----------------- | --------------------------- | ----------------------------------------------------------------------------------------------- |
| [init] | <code>Init</code> | <code>kMeansPlusPlus</code> | Centroids initialization when `fit` isn't given centroids, from the `init` namespace or custom. |

<a name="module_clustering..BaseOptions"></a>

### clustering~BaseOptions : <code>object</code>

**Kind**: inner typedef of [<code>clustering</code>](#module_clustering)
**Properties**

| Name       | Type                  | Default                       | Description                              |
| ---------- | --------------------- | ----------------------------- | ---------------------------------------- |
| [distance] | <code>Distance</code> | <code>squaredEuclidean</code> | Distance of the points to the centroids. |

<a name="module_clustering..FitOptions"></a>

### clustering~FitOptions : <code>object</code>

**Kind**: inner typedef of [<code>clustering</code>](#module_clustering)
**Properties**

| Name        | Type                                                | Description                                                  |
| ----------- | --------------------------------------------------- | ------------------------------------------------------------ |
| [weights]   | [<code>Weights</code>](#module_clustering..Weights) | Weight of each point. All 1 when omitted.                    |
| [centroids] | <code>ArrayLike.&lt;number&gt;</code>               | Initial centroids, `k * dimension` components. Skips `init`. |

<a name="module_distance"></a>

## distance

- [distance](#module_distance)
  - _static_
    - [.squaredEuclidean()](#module_distance.squaredEuclidean) : <code>Distance</code>
    - [.euclidean()](#module_distance.euclidean) : <code>Distance</code>
    - [.manhattan()](#module_distance.manhattan) : <code>Distance</code>
    - [.chebyshev()](#module_distance.chebyshev) : <code>Distance</code>
    - [.cosine()](#module_distance.cosine) : <code>Distance</code>
  - _inner_
    - [~Distance](#module_distance..Distance) ⇒ <code>number</code>

<a name="module_distance.squaredEuclidean"></a>

### distance.squaredEuclidean() : <code>Distance</code>

**Kind**: static method of [<code>distance</code>](#module_distance)
<a name="module_distance.euclidean"></a>

### distance.euclidean() : <code>Distance</code>

**Kind**: static method of [<code>distance</code>](#module_distance)
<a name="module_distance.manhattan"></a>

### distance.manhattan() : <code>Distance</code>

**Kind**: static method of [<code>distance</code>](#module_distance)
<a name="module_distance.chebyshev"></a>

### distance.chebyshev() : <code>Distance</code>

**Kind**: static method of [<code>distance</code>](#module_distance)
<a name="module_distance.cosine"></a>

### distance.cosine() : <code>Distance</code>

One minus the cosine similarity, between 0 and 2.

**Kind**: static method of [<code>distance</code>](#module_distance)
<a name="module_distance..Distance"></a>

### distance~Distance ⇒ <code>number</code>

Distance between the points at `aOffset` in `a` and `bOffset` in `b`.

**Kind**: inner typedef of [<code>distance</code>](#module_distance)

| Param     | Type                                  | Description                          |
| --------- | ------------------------------------- | ------------------------------------ |
| a         | <code>ArrayLike.&lt;number&gt;</code> |                                      |
| aOffset   | <code>number</code>                   | Index of the first component in `a`. |
| b         | <code>ArrayLike.&lt;number&gt;</code> |                                      |
| bOffset   | <code>number</code>                   | Index of the first component in `b`. |
| dimension | <code>number</code>                   |                                      |

<a name="module_fuzzyCMeans"></a>

## fuzzyCMeans

- [fuzzyCMeans](#module_fuzzyCMeans)
  - _static_
    - [.FuzzyCMeans](#module_fuzzyCMeans.FuzzyCMeans) ⇐ <code>Clustering</code>
      - [new FuzzyCMeans([options])](#new_module_fuzzyCMeans.FuzzyCMeans_new)
      - [.fuzziness](#module_fuzzyCMeans.FuzzyCMeans+fuzziness) : <code>number</code>
      - [.maxIterations](#module_fuzzyCMeans.FuzzyCMeans+maxIterations) : <code>number</code>
      - [.tolerance](#module_fuzzyCMeans.FuzzyCMeans+tolerance) : <code>number</code>
      - [.memberships](#module_fuzzyCMeans.FuzzyCMeans+memberships) : <code>Float64Array</code>
      - [.fit(points, [options])](#module_fuzzyCMeans.FuzzyCMeans+fit) ⇒ <code>this</code>
  - _inner_
    - [~FuzzyCMeansOptions](#module_fuzzyCMeans..FuzzyCMeansOptions) : <code>object</code>

<a name="module_fuzzyCMeans.FuzzyCMeans"></a>

### fuzzyCMeans.FuzzyCMeans ⇐ <code>Clustering</code>

Fuzzy c-means clustering: each point belongs to every cluster with a
membership between 0 and 1, minimizing the sum of squared distances weighted
by the memberships raised to `fuzziness`.

Suited to overlapping clusters and soft boundaries, like color gradients.
`labels` hold the cluster of highest membership.

**Kind**: static class of [<code>fuzzyCMeans</code>](#module_fuzzyCMeans)
**Extends**: <code>Clustering</code>
**See**

- Dunn, J. C. (1973). A fuzzy relative of the ISODATA process and its use in detecting compact well-separated clusters. Journal of Cybernetics, 3(3), 32–57. https://doi.org/10.1080/01969727308546046
- Bezdek, J. C. (1981). Pattern recognition with fuzzy objective function algorithms. Plenum Press. https://doi.org/10.1007/978-1-4757-0450-1
- Bezdek, J. C., Ehrlich, R., & Full, W. (1984). FCM: the fuzzy c-means clustering algorithm. Computers & Geosciences, 10(2–3), 191–203. https://doi.org/10.1016/0098-3004(84)90020-7
- Pal, N. R., & Bezdek, J. C. (1995). On cluster validity for the fuzzy c-means model. IEEE Transactions on Fuzzy Systems, 3(3), 370–379. https://doi.org/10.1109/91.413225 (fuzziness)
- https://github.com/scikit-fuzzy/scikit-fuzzy/blob/v0.5.0/skfuzzy/cluster/_cmeans.py (convergence, distances floor)
- https://github.com/omadson/fuzzy-c-means (defaults)

* [.FuzzyCMeans](#module_fuzzyCMeans.FuzzyCMeans) ⇐ <code>Clustering</code>
  - [new FuzzyCMeans([options])](#new_module_fuzzyCMeans.FuzzyCMeans_new)
  - [.fuzziness](#module_fuzzyCMeans.FuzzyCMeans+fuzziness) : <code>number</code>
  - [.maxIterations](#module_fuzzyCMeans.FuzzyCMeans+maxIterations) : <code>number</code>
  - [.tolerance](#module_fuzzyCMeans.FuzzyCMeans+tolerance) : <code>number</code>
  - [.memberships](#module_fuzzyCMeans.FuzzyCMeans+memberships) : <code>Float64Array</code>
  - [.fit(points, [options])](#module_fuzzyCMeans.FuzzyCMeans+fit) ⇒ <code>this</code>

<a name="new_module_fuzzyCMeans.FuzzyCMeans_new"></a>

#### new FuzzyCMeans([options])

| Param     | Type                                                                                                                                            |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| [options] | <code>ClusteringOptions</code> \| <code>InitializedOptions</code> \| [<code>FuzzyCMeansOptions</code>](#module_fuzzyCMeans..FuzzyCMeansOptions) |

<a name="module_fuzzyCMeans.FuzzyCMeans+fuzziness"></a>

#### fuzzyCMeans.fuzziness : <code>number</code>

**Kind**: instance property of [<code>FuzzyCMeans</code>](#module_fuzzyCMeans.FuzzyCMeans)
<a name="module_fuzzyCMeans.FuzzyCMeans+maxIterations"></a>

#### fuzzyCMeans.maxIterations : <code>number</code>

**Kind**: instance property of [<code>FuzzyCMeans</code>](#module_fuzzyCMeans.FuzzyCMeans)
<a name="module_fuzzyCMeans.FuzzyCMeans+tolerance"></a>

#### fuzzyCMeans.tolerance : <code>number</code>

**Kind**: instance property of [<code>FuzzyCMeans</code>](#module_fuzzyCMeans.FuzzyCMeans)
<a name="module_fuzzyCMeans.FuzzyCMeans+memberships"></a>

#### fuzzyCMeans.memberships : <code>Float64Array</code>

Membership of each point in each cluster of the last fit, `count * k`
values. Each point's memberships sum to 1.

**Kind**: instance property of [<code>FuzzyCMeans</code>](#module_fuzzyCMeans.FuzzyCMeans)
<a name="module_fuzzyCMeans.FuzzyCMeans+fit"></a>

#### fuzzyCMeans.fit(points, [options]) ⇒ <code>this</code>

**Kind**: instance method of [<code>FuzzyCMeans</code>](#module_fuzzyCMeans.FuzzyCMeans)

| Param     | Type                    | Description                                            |
| --------- | ----------------------- | ------------------------------------------------------ |
| points    | <code>Points</code>     | Flat array of point components, `dimension` per point. |
| [options] | <code>FitOptions</code> |                                                        |

<a name="module_fuzzyCMeans..FuzzyCMeansOptions"></a>

### fuzzyCMeans~FuzzyCMeansOptions : <code>object</code>

**Kind**: inner typedef of [<code>fuzzyCMeans</code>](#module_fuzzyCMeans)
**Properties**

| Name            | Type                | Default           | Description                                                                                                                                           |
| --------------- | ------------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| [fuzziness]     | <code>number</code> | <code>2</code>    | Exponent m above 1 weighting the memberships: higher blends clusters more. The usual choice, middle of the [1.5, 2.5] range of Pal and Bezdek (1995). |
| [maxIterations] | <code>number</code> | <code>150</code>  | The fuzzy-c-means package's default.                                                                                                                  |
| [tolerance]     | <code>number</code> | <code>1e-5</code> | Stop when the Frobenius norm of the memberships change falls under it, as in scikit-fuzzy. The fuzzy-c-means package's default.                       |

<a name="module_init"></a>

## init

- [init](#module_init)
  - _static_
    - [.randomPoints(points, k, [options])](#module_init.randomPoints) ⇒ <code>Uint32Array</code>
    - [.kMeansPlusPlus(points, k, [options])](#module_init.kMeansPlusPlus) ⇒ <code>Uint32Array</code>
    - [.farthestFirst(points, k, [options])](#module_init.farthestFirst) ⇒ <code>Uint32Array</code>
    - [.build(points, k, [options])](#module_init.build) ⇒ <code>Uint32Array</code>
  - _inner_
    - [~InitOptions](#module_init..InitOptions) : <code>object</code>
    - [~Init](#module_init..Init) ⇒ <code>ArrayLike.&lt;number&gt;</code>
    - [~KMeansPlusPlusOptions](#module_init..KMeansPlusPlusOptions) : <code>object</code>
    - [~FarthestFirstOptions](#module_init..FarthestFirstOptions) : <code>object</code>
    - [~BuildOptions](#module_init..BuildOptions) : <code>object</code>

<a name="module_init.randomPoints"></a>

### init.randomPoints(points, k, [options]) ⇒ <code>Uint32Array</code>

Pick `k` distinct points uniformly (Forgy initialization).

**Kind**: static method of [<code>init</code>](#module_init)
**Returns**: <code>Uint32Array</code> - Indices of the selected points.
**See**

- Forgy, E. W. (1965). Cluster analysis of multivariate data: efficiency versus interpretability of classifications. Biometrics, 21, 768–769.
- Bentley, J., & Floyd, R. W. (1987). Programming pearls: a sample of brilliance. Communications of the ACM, 30(9), 754–757. https://doi.org/10.1145/30401.315746 (sampling without replacement)

| Param     | Type                                                  | Description                     |
| --------- | ----------------------------------------------------- | ------------------------------- |
| points    | <code>ArrayLike.&lt;number&gt;</code>                 | Flat array of point components. |
| k         | <code>number</code>                                   |                                 |
| [options] | [<code>InitOptions</code>](#module_init..InitOptions) |                                 |

<a name="module_init.kMeansPlusPlus"></a>

### init.kMeansPlusPlus(points, k, [options]) ⇒ <code>Uint32Array</code>

Pick `k` points spread out by D² sampling (k-means++).

**Kind**: static method of [<code>init</code>](#module_init)
**Returns**: <code>Uint32Array</code> - Indices of the selected points.
**See**

- Arthur, D., & Vassilvitskii, S. (2007). k-means++: the advantages of careful seeding. Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms, 1027–1035. https://dl.acm.org/doi/10.5555/1283383.1283494
- https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/cluster/_kmeans.py (greedy local trials)

| Param     | Type                                                                                                                               | Description                     |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| points    | <code>ArrayLike.&lt;number&gt;</code>                                                                                              | Flat array of point components. |
| k         | <code>number</code>                                                                                                                |                                 |
| [options] | [<code>InitOptions</code>](#module_init..InitOptions) \| [<code>KMeansPlusPlusOptions</code>](#module_init..KMeansPlusPlusOptions) |                                 |

<a name="module_init.farthestFirst"></a>

### init.farthestFirst(points, k, [options]) ⇒ <code>Uint32Array</code>

Pick `k` points, each the farthest from the ones already selected
(farthest-first traversal, maximin initialization).

Also a 2-approximation of the k-center problem. `distances` receives the
distance of each point to its nearest selected point.

**Kind**: static method of [<code>init</code>](#module_init)
**Returns**: <code>Uint32Array</code> - Indices of the selected points.
**See**

- Gonzalez, T. F. (1985). Clustering to minimize the maximum intercluster distance. Theoretical Computer Science, 38, 293–306. https://doi.org/10.1016/0304-3975(85)90224-5
- Katsavounidis, I., Kuo, C.-C. J., & Zhang, Z. (1994). A new initialization technique for generalized Lloyd iteration. IEEE Signal Processing Letters, 1(10), 144–146. https://doi.org/10.1109/97.329844

| Param     | Type                                                                                                                             | Description                     |
| --------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| points    | <code>ArrayLike.&lt;number&gt;</code>                                                                                            | Flat array of point components. |
| k         | <code>number</code>                                                                                                              |                                 |
| [options] | [<code>InitOptions</code>](#module_init..InitOptions) \| [<code>FarthestFirstOptions</code>](#module_init..FarthestFirstOptions) |                                 |

<a name="module_init.build"></a>

### init.build(points, k, [options]) ⇒ <code>Uint32Array</code>

Pick `k` points greedily, each reducing the sum of distances of the points to
their nearest selected one the most (PAM BUILD), in O(k n²).

**Kind**: static method of [<code>init</code>](#module_init)
**Returns**: <code>Uint32Array</code> - Indices of the selected points.
**See**: Kaufman, L., & Rousseeuw, P. J. (1990). Partitioning around medoids (program PAM). In Finding groups in data: an introduction to cluster analysis, 68–125. Wiley. https://doi.org/10.1002/9780470316801.ch2

| Param     | Type                                                                                                             | Description                     |
| --------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| points    | <code>ArrayLike.&lt;number&gt;</code>                                                                            | Flat array of point components. |
| k         | <code>number</code>                                                                                              |                                 |
| [options] | [<code>InitOptions</code>](#module_init..InitOptions) \| [<code>BuildOptions</code>](#module_init..BuildOptions) |                                 |

<a name="module_init..InitOptions"></a>

### init~InitOptions : <code>object</code>

Options given by the clusterings to their `init`. Each initialization reads
the ones it needs.

**Kind**: inner typedef of [<code>init</code>](#module_init)
**Properties**

| Name        | Type                                  | Default                  | Description                                                        |
| ----------- | ------------------------------------- | ------------------------ | ------------------------------------------------------------------ |
| [dimension] | <code>number</code>                   | <code>3</code>           | Number of components per point.                                    |
| [random]    | <code>function</code>                 | <code>Math.random</code> | Uniform random number in [0, 1).                                   |
| [indices]   | <code>Uint32Array</code>              |                          | Receives the selected point indices. Reuse it to avoid allocating. |
| [weights]   | <code>ArrayLike.&lt;number&gt;</code> |                          | Weight of each point.                                              |
| [distance]  | <code>Distance</code>                 |                          | Distance of the clustering objective.                              |
| [distances] | <code>Float64Array</code>             |                          | Scratch buffer of at least one element per point.                  |

<a name="module_init..Init"></a>

### init~Init ⇒ <code>ArrayLike.&lt;number&gt;</code>

Select `k` distinct points to initialize a clustering, such as
`kMeansPlusPlus`. Wrap one to set its own options: `(points, k, options) =>
kMeansPlusPlus(points, k, { ...options, localTrials: 1 })`.

**Kind**: inner typedef of [<code>init</code>](#module_init)
**Returns**: <code>ArrayLike.&lt;number&gt;</code> - Indices of the selected points.

| Param   | Type                                                  | Description                     |
| ------- | ----------------------------------------------------- | ------------------------------- |
| points  | <code>ArrayLike.&lt;number&gt;</code>                 | Flat array of point components. |
| k       | <code>number</code>                                   |                                 |
| options | [<code>InitOptions</code>](#module_init..InitOptions) |                                 |

<a name="module_init..KMeansPlusPlusOptions"></a>

### init~KMeansPlusPlusOptions : <code>object</code>

**Kind**: inner typedef of [<code>init</code>](#module_init)
**Properties**

| Name          | Type                  | Default                       | Description                                                                                                                                                                            |
| ------------- | --------------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [distance]    | <code>Distance</code> | <code>squaredEuclidean</code> | Points are sampled proportionally to their distance to the nearest selected point: squared Euclidean for k-means, Manhattan for k-medians.                                             |
| [localTrials] | <code>number</code>   |                               | Candidates sampled per point, keeping the one reducing the potential the most. Defaults to 2 + ln(k) like the authors' and scikit-learn's implementations. 1 is the paper's algorithm. |

<a name="module_init..FarthestFirstOptions"></a>

### init~FarthestFirstOptions : <code>object</code>

**Kind**: inner typedef of [<code>init</code>](#module_init)
**Properties**

| Name       | Type                    | Default                       | Description                                                                     |
| ---------- | ----------------------- | ----------------------------- | ------------------------------------------------------------------------------- |
| [distance] | <code>Distance</code>   | <code>squaredEuclidean</code> |                                                                                 |
| [first]    | <code>number</code>     |                               | Index of the first point. Random when omitted.                                  |
| [labels]   | <code>Int32Array</code> |                               | Receives the position in `indices` of the nearest selected point of each point. |

<a name="module_init..BuildOptions"></a>

### init~BuildOptions : <code>object</code>

**Kind**: inner typedef of [<code>init</code>](#module_init)
**Properties**

| Name              | Type                                  | Default                | Description                                                                    |
| ----------------- | ------------------------------------- | ---------------------- | ------------------------------------------------------------------------------ |
| [distance]        | <code>Distance</code>                 | <code>euclidean</code> |                                                                                |
| [dissimilarities] | <code>ArrayLike.&lt;number&gt;</code> |                        | Precomputed `count * count` dissimilarity matrix, row-major. Skips `distance`. |

<a name="module_kCenter"></a>

## kCenter

- [kCenter](#module_kCenter)
  - _static_
    - [.KCenter](#module_kCenter.KCenter) ⇐ <code>Clustering</code>
      - [new KCenter([options])](#new_module_kCenter.KCenter_new)
      - [.centers](#module_kCenter.KCenter+centers) : <code>Uint32Array</code>
      - [.fit(points, [options])](#module_kCenter.KCenter+fit) ⇒ <code>this</code>
  - _inner_
    - [~KCenterOptions](#module_kCenter..KCenterOptions) : <code>object</code>
    - [~KCenterFitOptions](#module_kCenter..KCenterFitOptions) : <code>object</code>

<a name="module_kCenter.KCenter"></a>

### kCenter.KCenter ⇐ <code>Clustering</code>

K-center clustering: minimize the largest distance of a point to its center,
a point of the input, by farthest-first traversal.

Within twice the optimal radius, in O(k n). Covers outliers rather than dense
regions: suited to coverage, sampling and spreading colors apart. Point
weights don't change the radius and aren't supported.

**Kind**: static class of [<code>kCenter</code>](#module_kCenter)
**Extends**: <code>Clustering</code>
**See**

- Gonzalez, T. F. (1985). Clustering to minimize the maximum intercluster distance. Theoretical Computer Science, 38, 293–306. https://doi.org/10.1016/0304-3975(85)90224-5
- Hochbaum, D. S., & Shmoys, D. B. (1985). A best possible heuristic for the k-center problem. Mathematics of Operations Research, 10(2), 180–184. https://doi.org/10.1287/moor.10.2.180 (2 is the best ratio unless P = NP)

* [.KCenter](#module_kCenter.KCenter) ⇐ <code>Clustering</code>
  - [new KCenter([options])](#new_module_kCenter.KCenter_new)
  - [.centers](#module_kCenter.KCenter+centers) : <code>Uint32Array</code>
  - [.fit(points, [options])](#module_kCenter.KCenter+fit) ⇒ <code>this</code>

<a name="new_module_kCenter.KCenter_new"></a>

#### new KCenter([options])

| Param     | Type                                                                                             |
| --------- | ------------------------------------------------------------------------------------------------ |
| [options] | <code>ClusteringOptions</code> \| [<code>KCenterOptions</code>](#module_kCenter..KCenterOptions) |

<a name="module_kCenter.KCenter+centers"></a>

#### kCenter.centers : <code>Uint32Array</code>

Center point indices of the last fit.

**Kind**: instance property of [<code>KCenter</code>](#module_kCenter.KCenter)
<a name="module_kCenter.KCenter+fit"></a>

#### kCenter.fit(points, [options]) ⇒ <code>this</code>

**Kind**: instance method of [<code>KCenter</code>](#module_kCenter.KCenter)

| Param     | Type                                                                 | Description                                            |
| --------- | -------------------------------------------------------------------- | ------------------------------------------------------ |
| points    | <code>Points</code>                                                  | Flat array of point components, `dimension` per point. |
| [options] | [<code>KCenterFitOptions</code>](#module_kCenter..KCenterFitOptions) |                                                        |

<a name="module_kCenter..KCenterOptions"></a>

### kCenter~KCenterOptions : <code>object</code>

**Kind**: inner typedef of [<code>kCenter</code>](#module_kCenter)
**Properties**

| Name       | Type                  | Default                | Description                                                      |
| ---------- | --------------------- | ---------------------- | ---------------------------------------------------------------- |
| [distance] | <code>Distance</code> | <code>euclidean</code> | A metric: the 2-approximation relies on the triangle inequality. |

<a name="module_kCenter..KCenterFitOptions"></a>

### kCenter~KCenterFitOptions : <code>object</code>

**Kind**: inner typedef of [<code>kCenter</code>](#module_kCenter)
**Properties**

| Name    | Type                | Description                                     |
| ------- | ------------------- | ----------------------------------------------- |
| [first] | <code>number</code> | Index of the first center. Random when omitted. |

<a name="module_kMeans"></a>

## kMeans

- [kMeans](#module_kMeans)
  - _static_
    - [.KMeans](#module_kMeans.KMeans) ⇐ <code>Clustering</code>
      - [new KMeans([options])](#new_module_kMeans.KMeans_new)
      - [.algorithm](#module_kMeans.KMeans+algorithm) : <code>&quot;lloyd&quot;</code> \| <code>&quot;elkan&quot;</code> \| <code>&quot;hamerly&quot;</code>
      - [.maxIterations](#module_kMeans.KMeans+maxIterations) : <code>number</code>
      - [.tolerance](#module_kMeans.KMeans+tolerance) : <code>number</code>
      - [.emptyClusters](#module_kMeans.KMeans+emptyClusters) : <code>EmptyClusters</code>
      - [.spherical](#module_kMeans.KMeans+spherical) : <code>boolean</code>
      - [.index](#module_kMeans.KMeans+index) : [<code>CentroidIndex</code>](#module_kMeans..CentroidIndex) \| <code>undefined</code>
      - [.fit(points, [options])](#module_kMeans.KMeans+fit) ⇒ <code>this</code>
  - _inner_
    - [~CentroidIndex](#module_kMeans..CentroidIndex) : <code>object</code>
    - [~KMeansOptions](#module_kMeans..KMeansOptions) : <code>object</code>

<a name="module_kMeans.KMeans"></a>

### kMeans.KMeans ⇐ <code>Clustering</code>

K-means clustering: minimize the sum of squared distances of the points to
their cluster mean.

Best for compact, similarly sized clusters, color quantization and centroidal
Voronoi tessellations (Lloyd relaxation).

**Kind**: static class of [<code>kMeans</code>](#module_kMeans)
**Extends**: <code>Clustering</code>
**See**

- Lloyd, S. P. (1982). Least squares quantization in PCM. IEEE Transactions on Information Theory, 28(2), 129–137. https://doi.org/10.1109/TIT.1982.1056489
- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, 1, 281–297.
- Elkan, C. (2003). Using the triangle inequality to accelerate k-means. Proceedings of the Twentieth International Conference on Machine Learning, 147–153. https://dl.acm.org/doi/10.5555/3041838.3041857
- Hamerly, G. (2010). Making k-means even faster. Proceedings of the 2010 SIAM International Conference on Data Mining, 130–140. https://doi.org/10.1137/1.9781611972801.12
- Dhillon, I. S., & Modha, D. S. (2001). Concept decompositions for large sparse text data using clustering. Machine Learning, 42(1), 143–175. https://doi.org/10.1023/A:1007612920971 (spherical k-means)
- https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/cluster/_kmeans.py (convergence, empty clusters)

* [.KMeans](#module_kMeans.KMeans) ⇐ <code>Clustering</code>
  - [new KMeans([options])](#new_module_kMeans.KMeans_new)
  - [.algorithm](#module_kMeans.KMeans+algorithm) : <code>&quot;lloyd&quot;</code> \| <code>&quot;elkan&quot;</code> \| <code>&quot;hamerly&quot;</code>
  - [.maxIterations](#module_kMeans.KMeans+maxIterations) : <code>number</code>
  - [.tolerance](#module_kMeans.KMeans+tolerance) : <code>number</code>
  - [.emptyClusters](#module_kMeans.KMeans+emptyClusters) : <code>EmptyClusters</code>
  - [.spherical](#module_kMeans.KMeans+spherical) : <code>boolean</code>
  - [.index](#module_kMeans.KMeans+index) : [<code>CentroidIndex</code>](#module_kMeans..CentroidIndex) \| <code>undefined</code>
  - [.fit(points, [options])](#module_kMeans.KMeans+fit) ⇒ <code>this</code>

<a name="new_module_kMeans.KMeans_new"></a>

#### new KMeans([options])

| Param     | Type                                                                                                                             |
| --------- | -------------------------------------------------------------------------------------------------------------------------------- |
| [options] | <code>ClusteringOptions</code> \| <code>InitializedOptions</code> \| [<code>KMeansOptions</code>](#module_kMeans..KMeansOptions) |

<a name="module_kMeans.KMeans+algorithm"></a>

#### kMeans.algorithm : <code>&quot;lloyd&quot;</code> \| <code>&quot;elkan&quot;</code> \| <code>&quot;hamerly&quot;</code>

**Kind**: instance property of [<code>KMeans</code>](#module_kMeans.KMeans)
<a name="module_kMeans.KMeans+maxIterations"></a>

#### kMeans.maxIterations : <code>number</code>

**Kind**: instance property of [<code>KMeans</code>](#module_kMeans.KMeans)
<a name="module_kMeans.KMeans+tolerance"></a>

#### kMeans.tolerance : <code>number</code>

**Kind**: instance property of [<code>KMeans</code>](#module_kMeans.KMeans)
<a name="module_kMeans.KMeans+emptyClusters"></a>

#### kMeans.emptyClusters : <code>EmptyClusters</code>

**Kind**: instance property of [<code>KMeans</code>](#module_kMeans.KMeans)
<a name="module_kMeans.KMeans+spherical"></a>

#### kMeans.spherical : <code>boolean</code>

**Kind**: instance property of [<code>KMeans</code>](#module_kMeans.KMeans)
<a name="module_kMeans.KMeans+index"></a>

#### kMeans.index : [<code>CentroidIndex</code>](#module_kMeans..CentroidIndex) \| <code>undefined</code>

**Kind**: instance property of [<code>KMeans</code>](#module_kMeans.KMeans)
<a name="module_kMeans.KMeans+fit"></a>

#### kMeans.fit(points, [options]) ⇒ <code>this</code>

**Kind**: instance method of [<code>KMeans</code>](#module_kMeans.KMeans)

| Param     | Type                    | Description                                            |
| --------- | ----------------------- | ------------------------------------------------------ |
| points    | <code>Points</code>     | Flat array of point components, `dimension` per point. |
| [options] | <code>FitOptions</code> |                                                        |

<a name="module_kMeans..CentroidIndex"></a>

### kMeans~CentroidIndex : <code>object</code>

Nearest neighbour structure over the centroids, such as a spatial-partition
`KdTree`.

**Kind**: inner typedef of [<code>kMeans</code>](#module_kMeans)
**Properties**

| Name         | Type                  | Description                                                                                                            |
| ------------ | --------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| build        | <code>function</code> | Index the centroids.                                                                                                   |
| queryNearest | <code>function</code> | Called with a position, `k`, a maximum distance and an output array. Returns the indices of the `k` nearest centroids. |

<a name="module_kMeans..KMeansOptions"></a>

### kMeans~KMeansOptions : <code>object</code>

**Kind**: inner typedef of [<code>kMeans</code>](#module_kMeans)
**Properties**

| Name            | Type                                                                                                 | Default                           | Description                                                                                                                                                                                                                        |
| --------------- | ---------------------------------------------------------------------------------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [algorithm]     | <code>&quot;lloyd&quot;</code> \| <code>&quot;elkan&quot;</code> \| <code>&quot;hamerly&quot;</code> | <code>&quot;hamerly&quot;</code>  | Same clusters, different speed. Hamerly stores 2 bounds per point: fastest in low dimension. Elkan stores `k` bounds per point: fastest in high dimension with many clusters. Lloyd stores none and is the only one using `index`. |
| [maxIterations] | <code>number</code>                                                                                  | <code>300</code>                  | Scikit-learn's default.                                                                                                                                                                                                            |
| [tolerance]     | <code>number</code>                                                                                  | <code>1e-4</code>                 | Stop when the sum of squared centroid shifts falls under `tolerance` times the mean variance of the axes. scikit-learn's default: cuts the tail of vanishing shifts before labels settle. 0 runs until no point changes cluster.   |
| [emptyClusters] | <code>EmptyClusters</code>                                                                           | <code>&quot;farthest&quot;</code> |                                                                                                                                                                                                                                    |
| [spherical]     | <code>boolean</code>                                                                                 | <code>false</code>                | Cluster directions: points and centroids are normalized and compared by cosine distance.                                                                                                                                           |
| [index]         | [<code>CentroidIndex</code>](#module_kMeans..CentroidIndex)                                          |                                   | Lloyd only. Rebuilt over the centroids at each iteration, with the same `dimension`: faster with hundreds of clusters. Equidistant centroids may break ties differently.                                                           |

<a name="module_kMedians"></a>

## kMedians

- [kMedians](#module_kMedians)
  - _static_
    - [.KMedians](#module_kMedians.KMedians) ⇐ <code>Clustering</code>
      - [new KMedians([options])](#new_module_kMedians.KMedians_new)
      - [.maxIterations](#module_kMedians.KMedians+maxIterations) : <code>number</code>
      - [.tolerance](#module_kMedians.KMedians+tolerance) : <code>number</code>
      - [.emptyClusters](#module_kMedians.KMedians+emptyClusters) : <code>EmptyClusters</code>
      - [.fit(points, [options])](#module_kMedians.KMedians+fit) ⇒ <code>this</code>
  - _inner_
    - [~KMediansOptions](#module_kMedians..KMediansOptions) : <code>object</code>

<a name="module_kMedians.KMedians"></a>

### kMedians.KMedians ⇐ <code>Clustering</code>

K-medians clustering: minimize the sum of Manhattan distances of the points
to their cluster median, taken on each axis.

Less sensitive to outliers than k-means.

**Kind**: static class of [<code>kMedians</code>](#module_kMedians)
**Extends**: <code>Clustering</code>
**See**

- Jain, A. K., & Dubes, R. C. (1988). Algorithms for clustering data, §3.3. Prentice Hall.
- Bradley, P. S., Mangasarian, O. L., & Street, W. N. (1997). Clustering via concave minimization. Advances in Neural Information Processing Systems 9, 368–374.

* [.KMedians](#module_kMedians.KMedians) ⇐ <code>Clustering</code>
  - [new KMedians([options])](#new_module_kMedians.KMedians_new)
  - [.maxIterations](#module_kMedians.KMedians+maxIterations) : <code>number</code>
  - [.tolerance](#module_kMedians.KMedians+tolerance) : <code>number</code>
  - [.emptyClusters](#module_kMedians.KMedians+emptyClusters) : <code>EmptyClusters</code>
  - [.fit(points, [options])](#module_kMedians.KMedians+fit) ⇒ <code>this</code>

<a name="new_module_kMedians.KMedians_new"></a>

#### new KMedians([options])

| Param     | Type                                                                                                                                   |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| [options] | <code>ClusteringOptions</code> \| <code>InitializedOptions</code> \| [<code>KMediansOptions</code>](#module_kMedians..KMediansOptions) |

<a name="module_kMedians.KMedians+maxIterations"></a>

#### kMedians.maxIterations : <code>number</code>

**Kind**: instance property of [<code>KMedians</code>](#module_kMedians.KMedians)
<a name="module_kMedians.KMedians+tolerance"></a>

#### kMedians.tolerance : <code>number</code>

**Kind**: instance property of [<code>KMedians</code>](#module_kMedians.KMedians)
<a name="module_kMedians.KMedians+emptyClusters"></a>

#### kMedians.emptyClusters : <code>EmptyClusters</code>

**Kind**: instance property of [<code>KMedians</code>](#module_kMedians.KMedians)
<a name="module_kMedians.KMedians+fit"></a>

#### kMedians.fit(points, [options]) ⇒ <code>this</code>

**Kind**: instance method of [<code>KMedians</code>](#module_kMedians.KMedians)

| Param     | Type                    | Description                                            |
| --------- | ----------------------- | ------------------------------------------------------ |
| points    | <code>Points</code>     | Flat array of point components, `dimension` per point. |
| [options] | <code>FitOptions</code> |                                                        |

<a name="module_kMedians..KMediansOptions"></a>

### kMedians~KMediansOptions : <code>object</code>

**Kind**: inner typedef of [<code>kMedians</code>](#module_kMedians)
**Properties**

| Name            | Type                       | Default                           | Description                                                                                                                                                                                                                            |
| --------------- | -------------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [maxIterations] | <code>number</code>        | <code>300</code>                  | As `KMeans`.                                                                                                                                                                                                                           |
| [tolerance]     | <code>number</code>        | <code>0</code>                    | Stop when the sum of squared centroid shifts falls under `tolerance` times the mean variance of the axes. 0 runs until no point changes cluster: medians move between point components, without the tail of vanishing shifts of means. |
| [emptyClusters] | <code>EmptyClusters</code> | <code>&quot;farthest&quot;</code> |                                                                                                                                                                                                                                        |

<a name="module_kMedoids"></a>

## kMedoids

- [kMedoids](#module_kMedoids)
  - _static_
    - [.KMedoids](#module_kMedoids.KMedoids) ⇐ <code>Clustering</code>
      - [new KMedoids([options])](#new_module_kMedoids.KMedoids_new)
      - [.algorithm](#module_kMedoids.KMedoids+algorithm) : <code>&quot;fasterpam&quot;</code> \| <code>&quot;pam&quot;</code> \| <code>&quot;alternate&quot;</code>
      - [.maxIterations](#module_kMedoids.KMedoids+maxIterations) : <code>number</code>
      - [.medoids](#module_kMedoids.KMedoids+medoids) : <code>Uint32Array</code>
      - [.swaps](#module_kMedoids.KMedoids+swaps) : <code>number</code>
      - [.fit(points, [options])](#module_kMedoids.KMedoids+fit) ⇒ <code>this</code>
  - _inner_
    - [~KMedoidsOptions](#module_kMedoids..KMedoidsOptions) : <code>object</code>
    - [~KMedoidsFitOptions](#module_kMedoids..KMedoidsFitOptions) : <code>object</code>

<a name="module_kMedoids.KMedoids"></a>

### kMedoids.KMedoids ⇐ <code>Clustering</code>

K-medoids clustering: minimize the sum of distances of the points to their
cluster medoid, a point of the cluster.

Works with any dissimilarity and returns actual points. Stores a `count *
count` dissimilarity matrix: suited to a few thousand points.

Point weights aren't supported.

**Kind**: static class of [<code>kMedoids</code>](#module_kMedoids)
**Extends**: <code>Clustering</code>
**See**

- Kaufman, L., & Rousseeuw, P. J. (1990). Partitioning around medoids (program PAM). In Finding groups in data: an introduction to cluster analysis, 68–125. Wiley. https://doi.org/10.1002/9780470316801.ch2
- Schubert, E., & Rousseeuw, P. J. (2021). Fast and eager k-medoids clustering: O(k) runtime improvement of the PAM, CLARA, and CLARANS algorithms. Information Systems, 101, 101804. https://doi.org/10.1016/j.is.2021.101804
- Park, H.-S., & Jun, C.-H. (2009). A simple and fast algorithm for K-medoids clustering. Expert Systems with Applications, 36(2), 3336–3341. https://doi.org/10.1016/j.eswa.2008.01.039 (alternate)
- https://github.com/kno10/rust-kmedoids (original implementation)

* [.KMedoids](#module_kMedoids.KMedoids) ⇐ <code>Clustering</code>
  - [new KMedoids([options])](#new_module_kMedoids.KMedoids_new)
  - [.algorithm](#module_kMedoids.KMedoids+algorithm) : <code>&quot;fasterpam&quot;</code> \| <code>&quot;pam&quot;</code> \| <code>&quot;alternate&quot;</code>
  - [.maxIterations](#module_kMedoids.KMedoids+maxIterations) : <code>number</code>
  - [.medoids](#module_kMedoids.KMedoids+medoids) : <code>Uint32Array</code>
  - [.swaps](#module_kMedoids.KMedoids+swaps) : <code>number</code>
  - [.fit(points, [options])](#module_kMedoids.KMedoids+fit) ⇒ <code>this</code>

<a name="new_module_kMedoids.KMedoids_new"></a>

#### new KMedoids([options])

| Param     | Type                                                                                                |
| --------- | --------------------------------------------------------------------------------------------------- |
| [options] | <code>ClusteringOptions</code> \| [<code>KMedoidsOptions</code>](#module_kMedoids..KMedoidsOptions) |

<a name="module_kMedoids.KMedoids+algorithm"></a>

#### kMedoids.algorithm : <code>&quot;fasterpam&quot;</code> \| <code>&quot;pam&quot;</code> \| <code>&quot;alternate&quot;</code>

**Kind**: instance property of [<code>KMedoids</code>](#module_kMedoids.KMedoids)
<a name="module_kMedoids.KMedoids+maxIterations"></a>

#### kMedoids.maxIterations : <code>number</code>

**Kind**: instance property of [<code>KMedoids</code>](#module_kMedoids.KMedoids)
<a name="module_kMedoids.KMedoids+medoids"></a>

#### kMedoids.medoids : <code>Uint32Array</code>

Medoid point indices of the last fit.

**Kind**: instance property of [<code>KMedoids</code>](#module_kMedoids.KMedoids)
<a name="module_kMedoids.KMedoids+swaps"></a>

#### kMedoids.swaps : <code>number</code>

Medoid swaps of the last fit.

**Kind**: instance property of [<code>KMedoids</code>](#module_kMedoids.KMedoids)
<a name="module_kMedoids.KMedoids+fit"></a>

#### kMedoids.fit(points, [options]) ⇒ <code>this</code>

**Kind**: instance method of [<code>KMedoids</code>](#module_kMedoids.KMedoids)

| Param     | Type                                                                    | Description                                            |
| --------- | ----------------------------------------------------------------------- | ------------------------------------------------------ |
| points    | <code>Points</code>                                                     | Flat array of point components, `dimension` per point. |
| [options] | [<code>KMedoidsFitOptions</code>](#module_kMedoids..KMedoidsFitOptions) |                                                        |

<a name="module_kMedoids..KMedoidsOptions"></a>

### kMedoids~KMedoidsOptions : <code>object</code>

**Kind**: inner typedef of [<code>kMedoids</code>](#module_kMedoids)
**Properties**

| Name            | Type                                                                                                     | Default                            | Description                                                                                                                                                                                                 |
| --------------- | -------------------------------------------------------------------------------------------------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [algorithm]     | <code>&quot;fasterpam&quot;</code> \| <code>&quot;pam&quot;</code> \| <code>&quot;alternate&quot;</code> | <code>&quot;fasterpam&quot;</code> | FasterPAM eagerly swaps medoids, as good as PAM in O(n²) per iteration. PAM makes the best swap per iteration, in O(k n²). Alternate updates medoids within fixed clusters, like k-means: faster but worse. |
| [init]          | <code>Init</code>                                                                                        | <code>build</code>                 | Medoids initialization when `fit` isn't given medoids. Reads the dissimilarity matrix with `init.build`.                                                                                                    |
| [distance]      | <code>Distance</code>                                                                                    | <code>euclidean</code>             | Any dissimilarity, not necessarily a metric.                                                                                                                                                                |
| [maxIterations] | <code>number</code>                                                                                      | <code>100</code>                   | The kmedoids package's default, also bounding the swaps of `"pam"` and the partition updates of `"alternate"`.                                                                                              |

<a name="module_kMedoids..KMedoidsFitOptions"></a>

### kMedoids~KMedoidsFitOptions : <code>object</code>

**Kind**: inner typedef of [<code>kMedoids</code>](#module_kMedoids)
**Properties**

| Name              | Type                                  | Description                                                                    |
| ----------------- | ------------------------------------- | ------------------------------------------------------------------------------ |
| [medoids]         | <code>ArrayLike.&lt;number&gt;</code> | Initial medoid point indices. Skips `init`.                                    |
| [dissimilarities] | <code>Float64Array</code>             | Precomputed `count * count` dissimilarity matrix, row-major. Skips `distance`. |

<a name="module_metrics"></a>

## metrics

- [metrics](#module_metrics)
  - _static_
    - [.inertia(points, centroids, labels, [options])](#module_metrics.inertia) ⇒ <code>number</code>
    - [.silhouette(points, labels, [options])](#module_metrics.silhouette) ⇒ <code>number</code>
    - [.daviesBouldin(points, labels, [options])](#module_metrics.daviesBouldin) ⇒ <code>number</code>
    - [.calinskiHarabasz(points, labels, [options])](#module_metrics.calinskiHarabasz) ⇒ <code>number</code>
  - _inner_
    - [~MetricOptions](#module_metrics..MetricOptions) : <code>object</code>
    - [~InertiaOptions](#module_metrics..InertiaOptions) : <code>object</code>
    - [~SilhouetteOptions](#module_metrics..SilhouetteOptions) : <code>object</code>

<a name="module_metrics.inertia"></a>

### metrics.inertia(points, centroids, labels, [options]) ⇒ <code>number</code>

Weighted sum of squared distances of the points to their centroid: the
k-means objective.

**Kind**: static method of [<code>metrics</code>](#module_metrics)

| Param     | Type                                                                                                                           | Description                        |
| --------- | ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------- |
| points    | <code>ArrayLike.&lt;number&gt;</code>                                                                                          | Flat array of point components.    |
| centroids | <code>ArrayLike.&lt;number&gt;</code>                                                                                          | Flat array of centroid components. |
| labels    | <code>ArrayLike.&lt;number&gt;</code>                                                                                          | Cluster of each point.             |
| [options] | [<code>MetricOptions</code>](#module_metrics..MetricOptions) \| [<code>InertiaOptions</code>](#module_metrics..InertiaOptions) |                                    |

<a name="module_metrics.silhouette"></a>

### metrics.silhouette(points, labels, [options]) ⇒ <code>number</code>

Mean silhouette coefficient, between -1 and 1: how much closer each point is
to its own cluster than to the nearest other one. Higher is better. O(n²).

Points alone in their cluster score 0.

**Kind**: static method of [<code>metrics</code>](#module_metrics)
**See**: Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics, 20, 53–65. https://doi.org/10.1016/0377-0427(87)90125-7

| Param     | Type                                                                                                                                 | Description                     |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------- |
| points    | <code>ArrayLike.&lt;number&gt;</code>                                                                                                | Flat array of point components. |
| labels    | <code>ArrayLike.&lt;number&gt;</code>                                                                                                | Cluster of each point.          |
| [options] | [<code>MetricOptions</code>](#module_metrics..MetricOptions) \| [<code>SilhouetteOptions</code>](#module_metrics..SilhouetteOptions) |                                 |

<a name="module_metrics.daviesBouldin"></a>

### metrics.daviesBouldin(points, labels, [options]) ⇒ <code>number</code>

Davies–Bouldin index: mean similarity of each cluster with its most similar
one, comparing their spread to their distance. Lower is better, 0 at best.

**Kind**: static method of [<code>metrics</code>](#module_metrics)
**See**: Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-1(2), 224–227. https://doi.org/10.1109/TPAMI.1979.4766909

| Param     | Type                                                         | Description                     |
| --------- | ------------------------------------------------------------ | ------------------------------- |
| points    | <code>ArrayLike.&lt;number&gt;</code>                        | Flat array of point components. |
| labels    | <code>ArrayLike.&lt;number&gt;</code>                        | Cluster of each point.          |
| [options] | [<code>MetricOptions</code>](#module_metrics..MetricOptions) |                                 |

<a name="module_metrics.calinskiHarabasz"></a>

### metrics.calinskiHarabasz(points, labels, [options]) ⇒ <code>number</code>

Calinski–Harabasz index (variance ratio criterion): dispersion between
clusters over dispersion within clusters, scaled by their degrees of freedom.
Higher is better.

**Kind**: static method of [<code>metrics</code>](#module_metrics)
**See**: Caliński, T., & Harabasz, J. (1974). A dendrite method for cluster analysis. Communications in Statistics, 3(1), 1–27. https://doi.org/10.1080/03610927408827101

| Param     | Type                                                         | Description                     |
| --------- | ------------------------------------------------------------ | ------------------------------- |
| points    | <code>ArrayLike.&lt;number&gt;</code>                        | Flat array of point components. |
| labels    | <code>ArrayLike.&lt;number&gt;</code>                        | Cluster of each point.          |
| [options] | [<code>MetricOptions</code>](#module_metrics..MetricOptions) |                                 |

<a name="module_metrics..MetricOptions"></a>

### metrics~MetricOptions : <code>object</code>

**Kind**: inner typedef of [<code>metrics</code>](#module_metrics)
**Properties**

| Name        | Type                | Default        | Description                     |
| ----------- | ------------------- | -------------- | ------------------------------- |
| [dimension] | <code>number</code> | <code>3</code> | Number of components per point. |

<a name="module_metrics..InertiaOptions"></a>

### metrics~InertiaOptions : <code>object</code>

**Kind**: inner typedef of [<code>metrics</code>](#module_metrics)
**Properties**

| Name      | Type                                  | Description                               |
| --------- | ------------------------------------- | ----------------------------------------- |
| [weights] | <code>ArrayLike.&lt;number&gt;</code> | Weight of each point. All 1 when omitted. |

<a name="module_metrics..SilhouetteOptions"></a>

### metrics~SilhouetteOptions : <code>object</code>

**Kind**: inner typedef of [<code>metrics</code>](#module_metrics)
**Properties**

| Name       | Type                      | Default                | Description                             |
| ---------- | ------------------------- | ---------------------- | --------------------------------------- |
| [distance] | <code>Distance</code>     | <code>euclidean</code> |                                         |
| [out]      | <code>Float64Array</code> |                        | Receives the coefficient of each point. |

<a name="module_miniBatchKMeans"></a>

## miniBatchKMeans

- [miniBatchKMeans](#module_miniBatchKMeans)
  - _static_
    - [.MiniBatchKMeans](#module_miniBatchKMeans.MiniBatchKMeans) ⇐ <code>Clustering</code>
      - [new MiniBatchKMeans([options])](#new_module_miniBatchKMeans.MiniBatchKMeans_new)
      - [.batchSize](#module_miniBatchKMeans.MiniBatchKMeans+batchSize) : <code>number</code>
      - [.maxIterations](#module_miniBatchKMeans.MiniBatchKMeans+maxIterations) : <code>number</code>
      - [.maxNoImprovement](#module_miniBatchKMeans.MiniBatchKMeans+maxNoImprovement) : <code>number</code>
      - [.initSize](#module_miniBatchKMeans.MiniBatchKMeans+initSize) : <code>number</code> \| <code>undefined</code>
      - [.tolerance](#module_miniBatchKMeans.MiniBatchKMeans+tolerance) : <code>number</code>
      - [.steps](#module_miniBatchKMeans.MiniBatchKMeans+steps) : <code>number</code>
      - [.fit(points, [options])](#module_miniBatchKMeans.MiniBatchKMeans+fit) ⇒ <code>this</code>
  - _inner_
    - [~MiniBatchKMeansOptions](#module_miniBatchKMeans..MiniBatchKMeansOptions) : <code>object</code>

<a name="module_miniBatchKMeans.MiniBatchKMeans"></a>

### miniBatchKMeans.MiniBatchKMeans ⇐ <code>Clustering</code>

Mini-batch k-means clustering: k-means updated from small random batches,
each centroid moving towards its points with a decreasing learning rate.

Orders of magnitude faster than k-means on large inputs, for a slightly
higher cost.

**Kind**: static class of [<code>miniBatchKMeans</code>](#module_miniBatchKMeans)
**Extends**: <code>Clustering</code>
**See**

- Sculley, D. (2010). Web-scale k-means clustering. Proceedings of the 19th International Conference on World Wide Web, 1177–1178. https://doi.org/10.1145/1772690.1772862
- https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/cluster/_kmeans.py (passes, early stopping, initialization subset)

* [.MiniBatchKMeans](#module_miniBatchKMeans.MiniBatchKMeans) ⇐ <code>Clustering</code>
  - [new MiniBatchKMeans([options])](#new_module_miniBatchKMeans.MiniBatchKMeans_new)
  - [.batchSize](#module_miniBatchKMeans.MiniBatchKMeans+batchSize) : <code>number</code>
  - [.maxIterations](#module_miniBatchKMeans.MiniBatchKMeans+maxIterations) : <code>number</code>
  - [.maxNoImprovement](#module_miniBatchKMeans.MiniBatchKMeans+maxNoImprovement) : <code>number</code>
  - [.initSize](#module_miniBatchKMeans.MiniBatchKMeans+initSize) : <code>number</code> \| <code>undefined</code>
  - [.tolerance](#module_miniBatchKMeans.MiniBatchKMeans+tolerance) : <code>number</code>
  - [.steps](#module_miniBatchKMeans.MiniBatchKMeans+steps) : <code>number</code>
  - [.fit(points, [options])](#module_miniBatchKMeans.MiniBatchKMeans+fit) ⇒ <code>this</code>

<a name="new_module_miniBatchKMeans.MiniBatchKMeans_new"></a>

#### new MiniBatchKMeans([options])

| Param     | Type                                                                                                                                                        |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [options] | <code>ClusteringOptions</code> \| <code>InitializedOptions</code> \| [<code>MiniBatchKMeansOptions</code>](#module_miniBatchKMeans..MiniBatchKMeansOptions) |

<a name="module_miniBatchKMeans.MiniBatchKMeans+batchSize"></a>

#### miniBatchKMeans.batchSize : <code>number</code>

**Kind**: instance property of [<code>MiniBatchKMeans</code>](#module_miniBatchKMeans.MiniBatchKMeans)
<a name="module_miniBatchKMeans.MiniBatchKMeans+maxIterations"></a>

#### miniBatchKMeans.maxIterations : <code>number</code>

**Kind**: instance property of [<code>MiniBatchKMeans</code>](#module_miniBatchKMeans.MiniBatchKMeans)
<a name="module_miniBatchKMeans.MiniBatchKMeans+maxNoImprovement"></a>

#### miniBatchKMeans.maxNoImprovement : <code>number</code>

**Kind**: instance property of [<code>MiniBatchKMeans</code>](#module_miniBatchKMeans.MiniBatchKMeans)
<a name="module_miniBatchKMeans.MiniBatchKMeans+initSize"></a>

#### miniBatchKMeans.initSize : <code>number</code> \| <code>undefined</code>

**Kind**: instance property of [<code>MiniBatchKMeans</code>](#module_miniBatchKMeans.MiniBatchKMeans)
<a name="module_miniBatchKMeans.MiniBatchKMeans+tolerance"></a>

#### miniBatchKMeans.tolerance : <code>number</code>

**Kind**: instance property of [<code>MiniBatchKMeans</code>](#module_miniBatchKMeans.MiniBatchKMeans)
<a name="module_miniBatchKMeans.MiniBatchKMeans+steps"></a>

#### miniBatchKMeans.steps : <code>number</code>

Batches processed by the last fit. `iterations` counts passes over the
points.

**Kind**: instance property of [<code>MiniBatchKMeans</code>](#module_miniBatchKMeans.MiniBatchKMeans)
<a name="module_miniBatchKMeans.MiniBatchKMeans+fit"></a>

#### miniBatchKMeans.fit(points, [options]) ⇒ <code>this</code>

**Kind**: instance method of [<code>MiniBatchKMeans</code>](#module_miniBatchKMeans.MiniBatchKMeans)

| Param     | Type                    | Description                                            |
| --------- | ----------------------- | ------------------------------------------------------ |
| points    | <code>Points</code>     | Flat array of point components, `dimension` per point. |
| [options] | <code>FitOptions</code> |                                                        |

<a name="module_miniBatchKMeans..MiniBatchKMeansOptions"></a>

### miniBatchKMeans~MiniBatchKMeansOptions : <code>object</code>

**Kind**: inner typedef of [<code>miniBatchKMeans</code>](#module_miniBatchKMeans)
**Properties**

| Name               | Type                | Default           | Description                                                                                                                                                                                                    |
| ------------------ | ------------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [batchSize]        | <code>number</code> | <code>1024</code> | Points sampled, with replacement, per batch. scikit-learn's default.                                                                                                                                           |
| [maxIterations]    | <code>number</code> | <code>100</code>  | Passes over the points: at most `maxIterations * count / batchSize` batches. scikit-learn's default.                                                                                                           |
| [maxNoImprovement] | <code>number</code> | <code>10</code>   | Stop after this many batches without improving the smoothed batch cost. scikit-learn's default. 0 disables it.                                                                                                 |
| [initSize]         | <code>number</code> |                   | Points sampled, with replacement, to initialize the centroids on. Defaults to `3 * batchSize`, or `3 * k` when that is under `k`, as in scikit-learn: on all points, the initialization costs at least O(k n). |
| [tolerance]        | <code>number</code> | <code>0</code>    | Stop when the sum of squared centroid shifts of a batch falls under `tolerance` times the mean variance of the axes. 0 disables it, as scikit-learn's default: a single batch shift is noisy.                  |

<a name="module_utils"></a>

## utils

- [utils](#module_utils)
  - [.weightOf](#module_utils.weightOf) ⇒ <code>number</code>
  - [.resize(array, length)](#module_utils.resize) ⇒ <code>T</code>
  - [.gather(points, indices, count, dimension, out)](#module_utils.gather) ⇒ <code>Float64Array</code>
  - [.nearestCentroid(points, offset, centroids, k, dimension, distance, [out])](#module_utils.nearestCentroid) ⇒ <code>number</code>
  - [.meanVariance(points, count, dimension)](#module_utils.meanVariance) ⇒ <code>number</code>
  - [.scaledTolerance(points, count, dimension, tolerance)](#module_utils.scaledTolerance) ⇒ <code>number</code>
  - [.normalize(points, count, dimension)](#module_utils.normalize)
  - [.relocateEmptyClusters(sizes, distances, count, move)](#module_utils.relocateEmptyClusters)

<a name="module_utils.weightOf"></a>

### utils.weightOf ⇒ <code>number</code>

Weight of a point: 1 without weights.

**Kind**: static constant of [<code>utils</code>](#module_utils)

| Param   | Type                                                            |
| ------- | --------------------------------------------------------------- |
| weights | <code>ArrayLike.&lt;number&gt;</code> \| <code>undefined</code> |
| i       | <code>number</code>                                             |

<a name="module_utils.resize"></a>

### utils.resize(array, length) ⇒ <code>T</code>

Typed array of exactly `length` elements, reusing the storage of `array` when
large enough. A new view is only created when the length changes.

**Kind**: static method of [<code>utils</code>](#module_utils)

| Param  | Type                |
| ------ | ------------------- |
| array  | <code>T</code>      |
| length | <code>number</code> |

<a name="module_utils.gather"></a>

### utils.gather(points, indices, count, dimension, out) ⇒ <code>Float64Array</code>

Copy the points at `indices` into `out`, contiguously.

**Kind**: static method of [<code>utils</code>](#module_utils)

| Param     | Type                                  | Description                              |
| --------- | ------------------------------------- | ---------------------------------------- |
| points    | <code>ArrayLike.&lt;number&gt;</code> |                                          |
| indices   | <code>ArrayLike.&lt;number&gt;</code> |                                          |
| count     | <code>number</code>                   | Number of indices to copy.               |
| dimension | <code>number</code>                   |                                          |
| out       | <code>Float64Array</code>             | Receives `count * dimension` components. |

<a name="module_utils.nearestCentroid"></a>

### utils.nearestCentroid(points, offset, centroids, k, dimension, distance, [out]) ⇒ <code>number</code>

Index of the nearest of `k` centroids. Ties go to the lowest index.

**Kind**: static method of [<code>utils</code>](#module_utils)

| Param     | Type                                  | Description                                               |
| --------- | ------------------------------------- | --------------------------------------------------------- |
| points    | <code>ArrayLike.&lt;number&gt;</code> |                                                           |
| offset    | <code>number</code>                   | Index of the first component of the point.                |
| centroids | <code>Float64Array</code>             |                                                           |
| k         | <code>number</code>                   |                                                           |
| dimension | <code>number</code>                   |                                                           |
| distance  | <code>Distance</code>                 |                                                           |
| [out]     | <code>Float64Array</code>             | Receives the distance to the nearest centroid at index 0. |

<a name="module_utils.meanVariance"></a>

### utils.meanVariance(points, count, dimension) ⇒ <code>number</code>

Mean of the variances of each axis.

**Kind**: static method of [<code>utils</code>](#module_utils)

| Param     | Type                                  |
| --------- | ------------------------------------- |
| points    | <code>ArrayLike.&lt;number&gt;</code> |
| count     | <code>number</code>                   |
| dimension | <code>number</code>                   |

<a name="module_utils.scaledTolerance"></a>

### utils.scaledTolerance(points, count, dimension, tolerance) ⇒ <code>number</code>

Absolute tolerance on the sum of squared centroid shifts: `tolerance` times
the mean variance of the axes, as in scikit-learn, so that it doesn't depend
on the scale of the points. 0 when `tolerance` is 0.

**Kind**: static method of [<code>utils</code>](#module_utils)

| Param     | Type                                  |
| --------- | ------------------------------------- |
| points    | <code>ArrayLike.&lt;number&gt;</code> |
| count     | <code>number</code>                   |
| dimension | <code>number</code>                   |
| tolerance | <code>number</code>                   |

<a name="module_utils.normalize"></a>

### utils.normalize(points, count, dimension)

Normalize points to unit length. Zero vectors are kept.

**Kind**: static method of [<code>utils</code>](#module_utils)

| Param     | Type                      |
| --------- | ------------------------- |
| points    | <code>Float64Array</code> |
| count     | <code>number</code>       |
| dimension | <code>number</code>       |

<a name="module_utils.relocateEmptyClusters"></a>

### utils.relocateEmptyClusters(sizes, distances, count, move)

Move each empty cluster to the point farthest from its centroid, as in
scikit-learn. Points are taken once, farthest first.

**Kind**: static method of [<code>utils</code>](#module_utils)

| Param     | Type                      | Description                                                        |
| --------- | ------------------------- | ------------------------------------------------------------------ |
| sizes     | <code>Float64Array</code> | Cluster weights: 0 when empty.                                     |
| distances | <code>Float64Array</code> | Distance of each point to its centroid. Moved points are set to 0. |
| count     | <code>number</code>       |                                                                    |
| move      | <code>function</code>     | Called with a point index and the empty cluster.                   |

<!-- api-end -->

## License

MIT. See [license file](https://github.com/dmnsgn/k-clustering/blob/main/LICENSE.md).
