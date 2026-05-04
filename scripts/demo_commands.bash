#!/usr/bin/bash
# Reference commands for the oren_vl pipeline. Copy whichever line(s) you need;
# this file is not meant to be executed top-to-bottom.

# 1. Extract per-frame VL features from an RGB-D scan.
pipenv run python -m oren_vl.dataset.generate_vl_features \
    --config configs_local/replica-room0-generate-vl-features.yaml

# 2. Scatter the VL features into a SemiSparseOctree as residuals.
pipenv run python -m oren_vl.demo_semi_sparse_octree_vl \
    --config configs_local/replica-room0-octree-vl.yaml

# 3. Visualize the octree.
#    Default (no flags): opens an Open3D window.
pipenv run python -m oren_vl.utils.visualize_octree_vl \
    --config configs_local/replica-room0-visualize-octree-vl.yaml

#    Headless: render to PNG only.
pipenv run python -m oren_vl.utils.visualize_octree_vl \
    --config configs_local/replica-room0-visualize-octree-vl.yaml \
    --save-screenshot /tmp/octree_vl_room0.png

#    Save the PCA-colored mesh as PLY (any o3d-supported extension works).
pipenv run python -m oren_vl.utils.visualize_octree_vl \
    --config configs_local/replica-room0-visualize-octree-vl.yaml \
    --save-mesh /tmp/octree_vl_room0.ply

#    Save mesh + screenshot, and also open the window.
pipenv run python -m oren_vl.utils.visualize_octree_vl \
    --config configs_local/replica-room0-visualize-octree-vl.yaml \
    --save-mesh /tmp/octree_vl_room0.ply \
    --save-screenshot /tmp/octree_vl_room0.png \
    --interactive

# 4. Visualize the dataset as a colorized point cloud.
pipenv run python -m oren_vl.utils.visualize_pcd_vl \
    --config configs_local/replica-room0-visualize-pcd-vl.yaml

#    Save the PCA-colored cloud as PLY.
pipenv run python -m oren_vl.utils.visualize_pcd_vl \
    --config configs_local/replica-room0-visualize-pcd-vl.yaml \
    --save-pcd /tmp/pcd_vl_room0.ply

#    Save cloud + screenshot, and also open the window.
pipenv run python -m oren_vl.utils.visualize_pcd_vl \
    --config configs_local/replica-room0-visualize-pcd-vl.yaml \
    --save-pcd /tmp/pcd_vl_room0.ply \
    --save-screenshot /tmp/pcd_vl_room0.png \
    --interactive
