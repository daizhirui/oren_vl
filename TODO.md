[ ] Online surface occupancy estimation
[ ] Combine SDF and occupancy estimation in a single model
[ ] General octree field representation configuration:
    [ ] mode: explicit, implicit, hybrid (explicit + implicit)
    [ ] gradient_augmentation: True/False
    [ ] explicit_prior_init: float
    [ ] implicit_feature_dim_range: (min, max)   # For implicit features, we can have a range of dimensions to allow for more flexibility in the representation. The model can learn to use more or fewer features as needed, which can help with generalization and efficiency.
    [ ] implicit_feature_level: int              # The number of levels from the leaf node to collect implicit features.
    [ ] implicit_feature_aggregation: cat, sum, max_pooling, mean_pooling # The method to aggregate implicit features from multiple levels. This can help the model learn to combine information from different scales effectively.
