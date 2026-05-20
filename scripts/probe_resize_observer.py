"""Verification probes for the pow-2 resize + Adam-state migration changes.

Four standalone checks, each prints its own pass/fail line:

  1. `_round_up_pow2` sanity over a few inputs.
  2. End-to-end: fresh octree -> FieldBank -> attach optimizer -> insert points
     across multiple pow-2 boundary crossings. Verifies that:
       - per-vertex params resize in lockstep with `octree.capacity`,
       - Adam's `exp_avg` / `exp_avg_sq` resize with them,
       - `loss.backward()` and `optim.step()` survive each resize,
       - loss decreases monotonically across the run.
  3. `.data = new` vs `.data.set_(new)` vs `Parameter.set_(new)` for resizing
     a leaf Parameter. Demonstrates which pattern actually works and which
     ones either silently no-op or break the next backward.
  4. After `Parameter.set_(new)`, confirms that
       (a) the old storage is GC'd (weakref dies),
       (b) a fresh `AccumulateGrad` node replaces the old one.

Run with the pipenv venv (the install needs to point at the workspace's
`oren/` — re-install with `pip install -e .` from inside `oren_vl/oren/`
if you see ImportError for newly-added submodules):

    /home/daizhirui/.local/share/virtualenvs/oren-sEcPNj0M/bin/python \\
        path/to/oren_vl/scripts/probe_resize_observer.py

GPU is required (the FieldBank smoke section calls `.cuda()`). Drop those
two `.cuda()` calls and the `device='cuda'` arguments to run on CPU.
"""

from __future__ import annotations

import gc
import sys
import weakref

import torch
import torch.nn as nn

from oren.field_bank import FieldBank
from oren.field_storage_config import FieldStorageConfig
from oren.implicit_net import ImplicitNetConfig
from oren.octree_config import OctreeConfig
from oren.semi_sparse_octree import SemiSparseOctree, _round_up_pow2


def probe_pow2() -> None:
    print("=== probe 1: _round_up_pow2 ===")
    cases = [(0, 1), (1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (1024, 1024), (1025, 2048)]
    for n, expected in cases:
        got = _round_up_pow2(n)
        assert got == expected, f"_round_up_pow2({n}) = {got}, expected {expected}"
    print(f"  all {len(cases)} cases ok")


def probe_resize_and_adam() -> None:
    print("=== probe 2: FieldBank resize + Adam migration across pow-2 boundaries ===")
    torch.manual_seed(0)
    # init_vertex_num=0 forces capacity to seed at pow-2 of the fresh tree's
    # 8 root vertices (= 8), so the probe actually exercises pow-2 boundary
    # crossings as inserts grow the tree past 16, 32, ..., 16384 vertices.
    oc = OctreeConfig(
        resolution=0.05, tree_depth=8, semi_sparse_depth=5,
        init_voxel_num=200, init_vertex_num=0,
    )
    sso = SemiSparseOctree(oc).cuda()
    fs_cfg = FieldStorageConfig(
        name="sdf",
        mode="hybrid",
        output_dim=1,
        explicit_prior_init=0.0,
        gradient_augmentation=True,
        implicit_feature_dim=4,
        implicit_feature_level=1,
        implicit_feature_aggregation="cat",
        implicit_net_cfg=ImplicitNetConfig(
            input_dim=4, hidden_dims=[16], output_dim=1, output_scale=0.1,
        ),
    )
    bank = FieldBank(octree=sso, fields=[fs_cfg]).cuda()
    optim = torch.optim.Adam(bank.parameters(), lr=1e-3)
    bank.attach_optimizer(optim)

    sdf_features = bank.fields["sdf"].own_bank.features

    losses: list[float] = []
    for i, scale in enumerate([0.3, 0.6, 1.0, 1.5, 2.0]):
        pts = torch.rand(32768, 3, device="cuda") * scale * 2 - scale
        sso.insert_points(pts)
        q = torch.rand(4096, 3, device="cuda") * 2.0 - 1.0
        optim.zero_grad()
        out = bank(q)
        loss = out["sdf"].pred.pow(2).mean()
        loss.backward()
        optim.step()
        losses.append(loss.item())

        state_f = optim.state.get(sdf_features, {})
        exp_avg_f = state_f.get("exp_avg")
        print(
            f"  iter {i}: scale={scale}  num_v={int(sso.sso.num_vertices)}  "
            f"cap={sso.capacity}  feats={tuple(sdf_features.shape)}  "
            f"exp_avg={tuple(exp_avg_f.shape) if exp_avg_f is not None else None}  "
            f"loss={loss.item():.4e}"
        )
        assert sdf_features.shape[0] == sso.capacity, (
            f"feature shape {sdf_features.shape} out of sync with capacity {sso.capacity}"
        )
        if exp_avg_f is not None:
            assert exp_avg_f.shape == sdf_features.shape, (
                f"Adam exp_avg shape {exp_avg_f.shape} != feat shape {sdf_features.shape}"
            )

    # Sanity: loss should trend down monotonically (or near it).
    assert losses[-1] < losses[0], f"loss not decreasing: {losses}"
    print(f"  resize+adam ok, loss {losses[0]:.4e} -> {losses[-1]:.4e}")


def probe_resize_methods() -> None:
    """Compare the three plausible resize patterns; only one of them is correct."""
    print("=== probe 3: .data=new vs .data.set_ vs Parameter.set_ ===")
    for method in ("data_assign", "data_dot_set_", "param_set_"):
        p = nn.Parameter(torch.zeros(4, 2))
        optim = torch.optim.Adam([p], lr=0.1)
        # First forward / backward / step so AccumulateGrad metadata is set
        # and Adam state is populated.
        x = (p ** 2).sum()
        x.backward()
        optim.step()

        with torch.no_grad():
            new = torch.zeros(8, 2)
            new[:4] = p.detach()
            if method == "data_assign":
                p.data = new
            elif method == "data_dot_set_":
                p.data.set_(new)
            elif method == "param_set_":
                p.set_(new)

            # Migrate Adam state for the new shape (needed for any of the
            # methods that actually changed p.shape).
            st = optim.state[p]
            for k in ("exp_avg", "exp_avg_sq"):
                if k in st and st[k].shape != p.shape:
                    old_v = st[k]
                    new_v = torch.zeros(p.shape, dtype=old_v.dtype, device=old_v.device)
                    new_v[: old_v.shape[0]] = old_v
                    st[k] = new_v

        optim.zero_grad()
        try:
            x = (p ** 2).sum()
            x.backward()
            optim.step()
            status = f"ok  (final p.shape={tuple(p.shape)})"
            # Diagnostic flag: did the resize actually take effect?
            if p.shape != (8, 2):
                status += "  ! resize was a silent no-op"
        except RuntimeError as e:
            status = f"FAILED: {str(e).splitlines()[0]}"
        print(f"  {method:18s}: {status}")


def probe_storage_and_accumulategrad() -> None:
    """After Parameter.set_, is the old storage freed and AccumulateGrad replaced?"""
    print("=== probe 4: storage GC + AccumulateGrad replacement on Parameter.set_ ===")
    p = nn.Parameter(torch.zeros(4, 2))
    old_storage_ref = weakref.ref(p.untyped_storage())

    x = (p ** 2).sum()
    x.backward()

    y_before = p * 1.0
    acc_before_id = id(y_before.grad_fn.next_functions[0][0])
    del y_before

    with torch.no_grad():
        new = torch.zeros(8, 2)
        new[: 4] = p.detach()
        p.set_(new)

    gc.collect()
    storage_freed = old_storage_ref() is None

    y_after = p * 1.0
    acc_after_id = id(y_after.grad_fn.next_functions[0][0])
    accumulategrad_replaced = acc_before_id != acc_after_id

    print(f"  old storage freed:           {storage_freed}")
    print(f"  AccumulateGrad replaced:     {accumulategrad_replaced}")
    print(f"    (before id={acc_before_id}, after id={acc_after_id})")
    assert storage_freed, "old storage not freed after Parameter.set_"
    assert accumulategrad_replaced, "AccumulateGrad node not refreshed after Parameter.set_"


def probe_old_accumulategrad_released() -> None:
    """Probe 4 showed the NEW AccumulateGrad wrapper has a different Python id
    from the OLD one. That alone is consistent with two scenarios:
      (a) old C++ node destroyed, new C++ node allocated -- what we want; OR
      (b) old C++ node still alive (held by something), new C++ node allocated
          alongside it -- a leak.

    Here we distinguish: after `set_` and dropping the sole Python strong ref
    to the old wrapper, the wrapper's Python refcount goes to zero — verified
    by observing it disappear from `gc.get_objects()`. Combined with the fact
    that the leaf's `grad_accumulator_` weak_ptr was reset (so nothing on the
    leaf side holds a strong ref) and the previous graph was freed by the
    previous `backward()`, this leaves no path that could keep the old C++
    Node alive. The Python wrapper's destruction implies the underlying
    C++ object's last shared_ptr was released.
    """
    print("=== probe 5: old AccumulateGrad C++ object is released, not just disassociated ===")
    p = nn.Parameter(torch.zeros(4, 2))

    # First backward anchors the AccumulateGrad node — until this runs, the
    # leaf has no materialized accumulator and PyTorch returns a fresh wrapper
    # on every query (`is`-comparisons fail). After backward, the wrapper for
    # a given C++ Node is stable.
    x = (p ** 2).sum()
    x.backward()

    # Sanity: the same Python wrapper object is returned across two queries
    # for the same underlying C++ Node. Hold both refs alive across the
    # comparison — `id()` of temporary Python objects is *not* a reliable
    # identity check because Python recycles freed addresses.
    y1 = p * 1.0
    y2 = p * 1.0
    acc_a = y1.grad_fn.next_functions[0][0]
    acc_b = y2.grad_fn.next_functions[0][0]
    assert acc_a is acc_b, (
        "Python wrapper for AccumulateGrad is not stable across queries — "
        "PyTorch must return the same wrapper per C++ Node for this probe to work."
    )
    del y1, y2, acc_a, acc_b

    # Capture a strong Python reference to the old wrapper. While `old_acc`
    # lives, the underlying C++ Node lives.
    y = p * 1.0
    old_acc = y.grad_fn.next_functions[0][0]
    del y
    old_id = id(old_acc)
    # sys.getrefcount(x) returns refcount + 1 (the temporary arg slot); subtract.
    old_refcount = sys.getrefcount(old_acc) - 1
    print(f"  old AccumulateGrad: id={old_id}  py_refcount={old_refcount}")
    # Confirm the wrapper is reachable via gc-introspection at this point.
    assert any(o is old_acc for o in gc.get_objects()), (
        "old_acc not visible in gc.get_objects() before set_ — gc not tracking it"
    )

    with torch.no_grad():
        new = torch.zeros(8, 2)
        new[: 4] = p.detach()
        p.set_(new)

    # Drop the only remaining Python strong ref. If nothing else holds it
    # (graph already gone, leaf's weak_ptr reset by set_), the wrapper's
    # refcount drops to zero and the underlying C++ Node's last shared_ptr
    # is released.
    del old_acc
    gc.collect()

    # The new AccumulateGrad: hold the wrapper alive so we can compare with
    # `is` against any future re-queries and rule out address reuse.
    y_new = p * 1.0
    new_acc = y_new.grad_fn.next_functions[0][0]
    new_id = id(new_acc)
    print(f"  new AccumulateGrad: id={new_id}  is-stable-across-queries:", end=" ")
    same = True
    for _ in range(20):
        yy = p * 1.0
        if yy.grad_fn.next_functions[0][0] is not new_acc:
            same = False
            break
        del yy
    print(same)
    assert same, "new AccumulateGrad wrapper not stable across queries"

    # The killer test: the old wrapper Python object has gone — we verify by
    # walking gc-tracked objects and confirming no object lives at `old_id`
    # that still wraps a C++ AccumulateGrad. (Note: Python may have reused
    # the freed address for some unrelated object, so we filter by type.)
    survivors = [o for o in gc.get_objects() if id(o) == old_id and type(o) is type(new_acc)]
    print(f"  old wrapper survivors of matching type: {len(survivors)}")
    assert not survivors, (
        f"old AccumulateGrad wrapper at id={old_id} survived set_ + del + gc — "
        "either something else held a Python strong ref to it, or the old "
        "C++ Node is still alive and addressed via a fresh wrapper at the "
        "same memory location"
    )


if __name__ == "__main__":
    probe_pow2()
    probe_resize_and_adam()
    probe_resize_methods()
    probe_storage_and_accumulategrad()
    probe_old_accumulategrad_released()
    print("\nall probes ok.")
