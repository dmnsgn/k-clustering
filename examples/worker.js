self.importScripts("../web_modules/es-module-shims/wasm.js");

const FIELDS = ["centroids", "labels", "count", "cost", "iterations"];

// Module workers don't read the page's import map. Its paths are relative to
// the map file.
const mapURL = new URL("../web_modules/import-map.json", location.href);
const modules = fetch(mapURL)
  .then((response) => response.json())
  .then(({ imports }) => {
    self.importShim.addImportMap({
      imports: Object.fromEntries(
        Object.entries(imports).map(([name, url]) => [
          name,
          new URL(url, mapURL).href,
        ]),
      ),
    });
    return Promise.all(
      ["pex-random", "k-clustering", "./algorithms.js"].map((specifier) =>
        self.importShim(specifier),
      ),
    );
  });

onmessage = async ({ data: { id, name, points, config, dimension } }) => {
  const [{ default: random }, { metrics }, { ALGORITHMS }] = await modules;
  // Same seed for every algorithm: differences come from the algorithms.
  const rng = random.create(config.seed);
  const model = ALGORITHMS[name].create(
    { dimension, k: config.k, random: () => rng.float() },
    config,
  );
  const start = performance.now();
  model.fit(points);
  const time = performance.now() - start;

  // Davies–Bouldin is O(n + k²): silhouette is O(n²) per panel.
  const daviesBouldin = metrics.daviesBouldin(points, model.labels, {
    dimension,
  });
  const state = Object.fromEntries(
    FIELDS.map((field) => [field, model[field]]),
  );
  if (model.memberships) state.memberships = model.memberships;
  postMessage({ id, state, time, daviesBouldin }, [
    state.centroids.buffer,
    state.labels.buffer,
    ...(state.memberships ? [state.memberships.buffer] : []),
  ]);
};
