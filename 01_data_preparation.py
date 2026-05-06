# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:09:57 2026

@author: MNiger
"""
import anndata as ad


import scanpy as sc

import os
import numpy as np
import scanpy as sc

#adata = ad.read_h5ad("Library/CloudStorage/OneDrive-InsideMDAnderson/linghua_all/linghua_gastric cancer/combined_corrected.h5ad")


import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.sparse import issparse
from sklearn.neighbors import kneighbors_graph
import os
import gc

INPUT_PATH = "Library/CloudStorage/OneDrive-InsideMDAnderson/linghua_all/linghua_gastric cancer/combined_corrected.h5ad" 
OUTPUT_DIR = "Library/CloudStorage/OneDrive-InsideMDAnderson/linghua_all/stinr_data"
SLICES = ['B01', 'D02', 'E02']
N_NEIGHBORS = 6


os.makedirs(OUTPUT_DIR, exist_ok=True)


#Process each slice individually, save per-slice files

print("=" * 70)
print("PHASE 1: Process slices one at a time")
print("=" * 70)

slice_infos = []  # collect metadata per slice

for i, s in enumerate(SLICES):
    print(f"\n--- Processing slice {s} (idx {i}) ---")
    
    # Load full data
    print("  Loading h5ad...")
    adata_full = sc.read_h5ad(INPUT_PATH)
    
    # Filter to this slice ONLY
    mask = adata_full.obs['sample'] == s
    adata = adata_full[mask].copy()
    del adata_full
    gc.collect()
    
    n_spots = adata.shape[0]
    n_genes = adata.shape[1]
    print(f"  {s}: {n_spots:,} spots × {n_genes} genes")
    
    # Extract info
    coords_3d = adata.obsm['spatial'].astype(np.float64)
    z_val = coords_3d[0, 2]
    anno = adata.obs['anno_initial'].values.copy()
    obs_index = adata.obs.index.copy()
    gene_names = adata.var_names.copy()
    
    # Raw counts
    if issparse(adata.X):
        X_raw = adata.X.toarray().astype(np.float64)
    else:
        X_raw = adata.X.astype(np.float64)
    
    library_size = X_raw.sum(axis=1)
    
    # Save per-slice raw
    adata_raw = ad.AnnData(
        X=X_raw,
        obs=pd.DataFrame({
            'sample': [s] * n_spots,
            'anno_initial': anno,
        }, index=[f"{idx}-slice{i}" for idx in obs_index]),
        var=pd.DataFrame(index=gene_names)
    )
    adata_raw.obsm['spatial'] = coords_3d[:, :2]
    adata_raw.write_h5ad(os.path.join(OUTPUT_DIR, f"adata_st_list_raw{i}.h5ad"))
    print(f"  Saved adata_st_list_raw{i}.h5ad")
    del adata_raw
    gc.collect()
    
    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if issparse(adata.X):
        X_norm = adata.X.toarray().astype(np.float32)
    else:
        X_norm = adata.X.astype(np.float32)
    
    # Build graph for this slice
    coords_2d = coords_3d[:, :2]
    graph = kneighbors_graph(coords_2d, N_NEIGHBORS, mode='connectivity',
                             include_self=False)
    graph_indices = graph.indices.reshape(-1, N_NEIGHBORS)
    
    # Store slice info
    slice_infos.append({
        'name': s,
        'idx': i,
        'n_spots': n_spots,
        'z': z_val,
        'X_norm': X_norm,
        'X_raw': X_raw,
        'coords_3d': coords_3d,
        'graph_indices': graph_indices,
        'library_size': library_size,
        'anno': anno,
        'obs_index': obs_index,
        'gene_names': gene_names,
    })
    
    del adata, graph, coords_2d
    gc.collect()
    print(f"  {s} done: z={z_val}, lib_size=[{library_size.min():.0f}, {library_size.max():.0f}]")


# Combine slices into adata_st

print(f"\n{'='*70}")
print("PHASE 2: Combining slices")
print("=" * 70)

gene_names = slice_infos[0]['gene_names']

# Stack arrays
X_norm_all = np.vstack([si['X_norm'] for si in slice_infos])
X_raw_all = np.vstack([si['X_raw'] for si in slice_infos])
coords_all = np.vstack([si['coords_3d'] for si in slice_infos])
lib_all = np.concatenate([si['library_size'] for si in slice_infos])
anno_all = np.concatenate([si['anno'] for si in slice_infos])
samples_all = np.concatenate([[si['name']] * si['n_spots'] for si in slice_infos])

# Slice labels
slice_labels = np.concatenate([
    np.full(si['n_spots'], si['idx'], dtype=int) for si in slice_infos
])

# Graph: offset indices to global
global_offset = 0
graph_parts = []
for si in slice_infos:
    graph_parts.append(si['graph_indices'] + global_offset)
    global_offset += si['n_spots']
graph_all = np.vstack(graph_parts)

# Obs index
obs_idx_all = np.concatenate([
    [f"{idx}-slice{si['idx']}" for idx in si['obs_index']]
    for si in slice_infos
])

print(f"  Total spots: {X_norm_all.shape[0]:,}")
print(f"  Genes: {X_norm_all.shape[1]}")
print(f"  Graph: {graph_all.shape}")

# Free per-slice data
del slice_infos
gc.collect()

# Build adata_st
obs_df = pd.DataFrame({
    'slice': slice_labels,
    'library_size': lib_all,
    'sample': samples_all,
    'anno_initial': anno_all,
}, index=obs_idx_all)

adata_st = ad.AnnData(
    X=X_norm_all,
    obs=obs_df,
    var=pd.DataFrame(index=gene_names)
)
adata_st.obsm['3D_coor'] = coords_all
adata_st.obsm['graph'] = graph_all
adata_st.obsm['count'] = X_raw_all

adata_st.write_h5ad(os.path.join(OUTPUT_DIR, "adata_st.h5ad"))
print("  Saved adata_st.h5ad")


# Build basis matrix
print(f"\n{'='*70}")
print("PHASE 3: Building basis matrix")
print("=" * 70)

cell_types = sorted(np.unique(anno_all))
basis_data = []
for ct in cell_types:
    ct_mask = anno_all == ct
    mean_expr = X_norm_all[ct_mask].mean(axis=0)
    basis_data.append(mean_expr)

basis_matrix = np.array(basis_data).astype(np.float32)
adata_basis = ad.AnnData(
    X=basis_matrix,
    obs=pd.DataFrame(index=cell_types),
    var=pd.DataFrame(index=gene_names)
)
adata_basis.write_h5ad(os.path.join(OUTPUT_DIR, "adata_basis.h5ad"))
print(f"  adata_basis: {adata_basis.shape} — saved")
print(f"\n{'='*70}")
print("DONE!")
print(f"  adata_st:    {adata_st.shape}")
print(f"  adata_basis: {adata_basis.shape}")
print(f"  Files in {OUTPUT_DIR}/")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size_mb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1e6
    print(f"    {f}: {size_mb:.1f} MB")
