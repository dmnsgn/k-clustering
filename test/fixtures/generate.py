"""Generate reference.json from the reference implementations.

Usage: uv run --with scikit-learn --with kmedoids --with scikit-fuzzy test/fixtures/generate.py
"""

import json
import pathlib

import kmedoids
import numpy as np
import skfuzzy
from skfuzzy.cluster.normalize_columns import normalize_power_columns
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

HERE = pathlib.Path(__file__).parent
rng = np.random.default_rng(1)


def flat(array):
    return np.asarray(array, dtype=float).ravel().tolist()


def ints(array):
    return np.asarray(array).astype(int).ravel().tolist()


datasets = {
    "blobs2": make_blobs(300, n_features=2, centers=4, cluster_std=0.8, random_state=0)[0],
    "blobs3": make_blobs(500, n_features=3, centers=5, random_state=1)[0],
}
weights = rng.uniform(0.1, 2, len(datasets["blobs2"]))


def seeds(points, k):
    return points[rng.choice(len(points), k, replace=False)]


# K-means: fixed initial centroids, so results only depend on the iterations.
kmeans = []
for name, points, k, sample_weight in (
    ("blobs2", datasets["blobs2"], 4, None),
    ("blobs2", datasets["blobs2"], 4, weights),
    ("blobs3", datasets["blobs3"], 5, None),
    ("blobs3", datasets["blobs3"], 8, None),
):
    centroids = seeds(points, k)
    for tolerance in (0, 1e-4):
        model = KMeans(k, init=centroids, n_init=1, tol=tolerance, algorithm="lloyd")
        model.fit(points, sample_weight=sample_weight)
        kmeans.append(
            {
                "data": name,
                "k": k,
                "weighted": sample_weight is not None,
                "tolerance": tolerance,
                "init": flat(centroids),
                "centroids": flat(model.cluster_centers_),
                "labels": ints(model.labels_),
                "inertia": float(model.inertia_),
                "iterations": int(model.n_iter_),
            }
        )

# A far centroid attracts no point and is relocated.
empty_init = np.vstack([seeds(datasets["blobs2"], 3), [[50, 50]]])
model = KMeans(4, init=empty_init, n_init=1, tol=0).fit(datasets["blobs2"])
kmeans.append(
    {
        "data": "blobs2",
        "k": 4,
        "weighted": False,
        "tolerance": 0,
        "init": flat(empty_init),
        "centroids": flat(model.cluster_centers_),
        "labels": ints(model.labels_),
        "inertia": float(model.inertia_),
        "iterations": int(model.n_iter_),
    }
)


# K-medians: no reference library, Lloyd iterations with numpy medians.
def kmedians(points, centroids):
    labels = np.full(len(points), -1)
    for iteration in range(1, 301):
        new_labels = cdist(points, centroids, "cityblock").argmin(axis=1)
        changed = not np.array_equal(new_labels, labels)
        labels = new_labels
        centroids = np.array([np.median(points[labels == c], axis=0) for c in range(len(centroids))])
        if not changed:
            break
    return centroids, labels, iteration


points = datasets["blobs3"]
init = seeds(points, 5)
centroids, labels, iterations = kmedians(points, init)
kmedians_reference = {
    "data": "blobs3",
    "k": 5,
    "init": flat(init),
    "centroids": flat(centroids),
    "labels": ints(labels),
    "cost": float(np.abs(points - centroids[labels]).sum()),
    "iterations": iterations,
}

# K-medoids: the authors' implementation.
points = datasets["blobs2"]
k = 4
init = rng.choice(len(points), k, replace=False)
medoids = []
for metric in ("euclidean", "cityblock"):
    diss = cdist(points, points, metric)
    for algorithm, result in (
        ("fasterpam", kmedoids.fasterpam(diss, init.copy(), n_cpu=1)),
        ("pam", kmedoids.pam(diss, init.copy())),
        ("alternate", kmedoids.alternating(diss, init.copy())),
    ):
        medoids.append(
            {
                "data": "blobs2",
                "metric": metric,
                "algorithm": algorithm,
                "k": k,
                "init": ints(init),
                "medoids": ints(result.medoids),
                "labels": ints(result.labels),
                "cost": float(result.loss),
                "iterations": int(result.n_iter),
                "swaps": int(result.n_swap or 0),
            }
        )
    build = kmedoids.pam_build(diss, k)
    pam = kmedoids.pam(diss, k, init="build")
    medoids.append(
        {
            "data": "blobs2",
            "metric": metric,
            "algorithm": "pam",
            "build": ints(build.medoids),
            "k": k,
            "medoids": ints(pam.medoids),
            "labels": ints(pam.labels),
            "cost": float(pam.loss),
            "iterations": int(pam.n_iter),
            "swaps": int(pam.n_swap),
        }
    )


# K-center: farthest-first traversal from the first point.
def farthest_first(points, k, first):
    centers = [first]
    nearest = cdist(points, points[[first]]).ravel()
    labels = np.zeros(len(points), dtype=int)
    for c in range(1, k):
        centers.append(int(nearest.argmax()))
        d = cdist(points, points[[centers[-1]]]).ravel()
        labels[d < nearest] = c
        nearest = np.minimum(nearest, d)
    return centers, labels, float(nearest.max())


centers, labels, radius = farthest_first(datasets["blobs3"], 6, 0)
kcenter = {"data": "blobs3", "k": 6, "first": 0, "centers": centers, "labels": ints(labels), "cost": radius}

# Fuzzy c-means: initial memberships from the initial centroids, as computed by
# the first iteration.
points = datasets["blobs2"]
fuzzy = []
for fuzziness in (2, 1.5):
    init = seeds(points, 4)
    distances = np.fmax(cdist(points, init).T, np.finfo(np.float64).eps)
    u0 = normalize_power_columns(distances, -2 / (fuzziness - 1))
    centroids, u, _, _, _, iterations, _ = skfuzzy.cmeans(
        points.T, 4, fuzziness, error=1e-8, maxiter=1000, init=u0
    )
    fuzzy.append(
        {
            "data": "blobs2",
            "k": 4,
            "fuzziness": fuzziness,
            "init": flat(init),
            "centroids": flat(centroids),
            "memberships": flat(u.T),
            "iterations": int(iterations),
        }
    )

# Metrics on a k-means clustering.
points = datasets["blobs3"]
labels = np.array(kmeans[4]["labels"])
metrics = {
    "data": "blobs3",
    "labels": ints(labels),
    "silhouette": float(silhouette_score(points, labels)),
    "silhouetteManhattan": float(silhouette_score(points, labels, metric="manhattan")),
    "silhouetteSamples": flat(silhouette_samples(points, labels)),
    "daviesBouldin": float(davies_bouldin_score(points, labels)),
    "calinskiHarabasz": float(calinski_harabasz_score(points, labels)),
}

(HERE / "reference.json").write_text(
    json.dumps(
        {
            "data": {name: flat(points) for name, points in datasets.items()},
            "weights": flat(weights),
            "kmeans": kmeans,
            "kmedians": kmedians_reference,
            "kmedoids": medoids,
            "kcenter": kcenter,
            "fuzzy": fuzzy,
            "metrics": metrics,
        }
    )
)
