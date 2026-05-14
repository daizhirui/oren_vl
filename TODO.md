- [ ] Online surface occupancy estimation
- [ ] Combine SDF and occupancy estimation in a single model
- [ ] General octree field representation configuration: check [DESIGN.md](DESIGN.md) for details.
- [ ] Hybrid mode: pick coupling for spatial-gradient losses (e.g. Eikonal).
    The decision is whether prior should receive Eikonal updates shaped by
    the joint field (prior + implicit) or not.
    - one-way (FD on `pred = prior.detach() + impl`): prior is updated only
      by its own `*_prior` terms; implicit absorbs the joint Eikonal
      residual. Symmetric with the data-loss convention; easier debugging.
    - two-way (`prior + impl`, FD or autograd): prior also feels the
      joint-field Eikonal, with update direction reshaped by implicit (the
      L2 norm `||g_prior + g_impl||` couples direction in prior-param
      space). Useful when prior-only supervision is too weak.

    Note: autograd on `prior.detach() + impl` is broken — it minimizes
    `||∇impl||` instead of `||∇field||`. FD is fine because it operates on
    numeric values, not the autograd graph.

    Encode as a criterion-level flag (e.g., `joint_grad_couples_prior:
    bool`); FieldStorage stays neutral (returns `prior`, `implicit`, `pred`).
    Default to one-way; revisit if prior-only convergence is too slow on a
    workload with sparse direct supervision.
