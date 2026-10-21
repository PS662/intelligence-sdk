# Improvements

Out of scope for this version, noted so they are decisions rather than gaps.

- **Dataset versioning.** A saved dataset is identified by its path only. A registry
  such as DVC would give it a content hash and make "the same dataset" checkable
  across machines and time; the `DataLoader` port is where that loader plugs in.
- **Leakage checks.** Nothing stops a test from scoring rows a model was fitted on.
  Excluding `{split: train}` is the user's responsibility and needs the dataset to
  have been saved. A guard could compare the scored ids with the model's recorded
  `fitted_on` rows.
- **Concurrent writers.** Save paths are plain directories with an overwrite flag.
  Two runs writing the same path at once are not protected.
- **Joins.** Datasets combine by row concatenation only; feature joins on an id
  column are not supported.
- **Preprocess node type.** The `Node` union and the resolver are ready for a node
  that transforms features (encoding, scaling as data rather than as a pipeline
  step); it is not built.
- **Per-node seeds.** Each spec and node carries one seed for its split and folds;
  there is no recipe-wide seed.
- **Non-numeric features.** Categorical columns must be encoded before loading.
