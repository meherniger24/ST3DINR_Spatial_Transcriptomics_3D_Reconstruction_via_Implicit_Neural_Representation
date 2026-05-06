# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:09:57 2026

@author: MNiger
"""
"""
Train STINR on gastric cancer spatial transcriptomics data.

Adapted from STINR (CVPR 2025) for 3-slice gastric cancer 3D reconstruction.
Includes subsampling for initial pipeline testing on Mac M4 Max.

Usage:
    python train_stinr_gastric.py
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch
import random
from sklearn.neighbors import kneighbors_graph
from sklearn.mixture import GaussianMixture
from scipy.sparse import issparse
sys.path.insert(0, "/Users/mniger/Library/CloudStorage/OneDrive-InsideMDAnderson/linghua_all/STINR/STINR/")  

DATA_DIR = "Library/CloudStorage/OneDrive-InsideMDAnderson/linghua_all/stinr_data_9_slices"  

# Subsampling: set to None for full data, or a number like 50000 for testing
SUBSAMPLE_PER_SLICE = 20000  # ~60K total spots for initial test (set None for full)

SEED = 42
TRAINING_STEPS = 14001  # reduce for testing (original: 14001)
LR = 0.001
N_NEIGHBORS = 6
N_CLUSTERS = 22  # number of cell types for GMM clustering

# Device
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "Library/CloudStorage/OneDrive-InsideMDAnderson/linghua_all/stinr_results_9_slices"


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subsample_data(adata_st, adata_st_list_raw, n_per_slice, n_neighbors):
    """Subsample spots from each slice for faster testing."""
    if n_per_slice is None:
        return adata_st, adata_st_list_raw
    
    print(f"\nSubsampling to {n_per_slice} spots per slice...")
    
    slices = sorted(adata_st.obs['slice'].unique())
    keep_indices = []
    
    for s in slices:
        s_mask = np.where(adata_st.obs['slice'].values == s)[0]
        n_available = len(s_mask)
        n_take = min(n_per_slice, n_available)
        chosen = np.random.choice(s_mask, n_take, replace=False)
        chosen.sort()
        keep_indices.append(chosen)
    
    keep_indices = np.concatenate(keep_indices)
    
    # Subsample adata_st
    adata_sub = ad.AnnData(
        X=adata_st.X[keep_indices] if not issparse(adata_st.X) 
            else adata_st.X[keep_indices].toarray(),
        obs=adata_st.obs.iloc[keep_indices].reset_index(drop=True),
        var=adata_st.var.copy()
    )
    adata_sub.obsm['3D_coor'] = adata_st.obsm['3D_coor'][keep_indices]
    adata_sub.obsm['count'] = adata_st.obsm['count'][keep_indices]
    
    # Rebuild graph on subsampled data (per slice)
    all_graph_indices = []
    for s in slices:
        s_mask = np.where(adata_sub.obs['slice'].values == s)[0]
        coords_2d = adata_sub.obsm['3D_coor'][s_mask, :2]
        graph = kneighbors_graph(coords_2d, n_neighbors, mode='connectivity', 
                                 include_self=False)
        indices = graph.indices.reshape(-1, n_neighbors)
        global_offset = s_mask[0]
        indices_global = indices + global_offset
        all_graph_indices.append(indices_global)
    
    adata_sub.obsm['graph'] = np.vstack(all_graph_indices)
    
    # Rebuild obs index
    adata_sub.obs.index = [f"spot_{i}" for i in range(len(adata_sub))]
    
    # Subsample per-slice raw data
    adata_list_sub = []
    for s_idx, s in enumerate(slices):
        s_mask_orig = np.where(adata_sub.obs['slice'].values == s)[0]
        raw_X = adata_sub.obsm['count'][s_mask_orig]
        adata_raw_s = ad.AnnData(
            X=raw_X,
            obs=adata_sub.obs.iloc[s_mask_orig].reset_index(drop=True),
            var=adata_sub.var.copy()
        )
        adata_raw_s.obs.index = [f"spot_{i}-slice{s}" for i in range(len(adata_raw_s))]
        adata_raw_s.obsm['spatial'] = adata_sub.obsm['3D_coor'][s_mask_orig, :2]
        adata_list_sub.append(adata_raw_s)
    
    for s_idx, s in enumerate(slices):
        print(f"  Slice {s}: {adata_list_sub[s_idx].shape[0]:,} spots")
    print(f"  Total: {adata_sub.shape[0]:,} spots")
    
    return adata_sub, adata_list_sub


def check_mps_compatibility():
    """Check and warn about MPS (Apple Silicon) limitations."""
    if DEVICE == "mps":
        print("\n⚠  Running on Apple MPS (Metal Performance Shaders)")
        print("   Some PyTorch operations may fall back to CPU.")
        print("   If you see errors, try setting DEVICE = 'cpu'\n")
    return DEVICE


def main():
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    device_str = check_mps_compatibility()
    print(f"Device: {device_str}")
    
    # ── Load data ──
    print("=" * 70)
    print("Loading prepared STINR data...")
    print("=" * 70)
    
    adata_st = ad.read_h5ad(os.path.join(DATA_DIR, "adata_st.h5ad"))
    adata_basis = ad.read_h5ad(os.path.join(DATA_DIR, "adata_basis.h5ad"))
    
    adata_st_list_raw = []
    for i in range(9):
        path = os.path.join(DATA_DIR, f"adata_st_list_raw{i}.h5ad")
        adata_st_list_raw.append(ad.read_h5ad(path))
    
    print(f"  adata_st: {adata_st.shape}")
    print(f"  adata_basis: {adata_basis.shape}")
    print(f"  Slices: {[a.shape for a in adata_st_list_raw]}")
    
    # ── Handle zero-count spots ──
    lib_sizes = adata_st.obs['library_size'].values
    zero_mask = lib_sizes == 0
    n_zero = zero_mask.sum()
    if n_zero > 0:
        print(f"\n  Warning: {n_zero} spots have zero library size.")
        print(f"  Setting them to 1 to avoid log(0) in loss function.")
        adata_st.obs.loc[zero_mask, 'library_size'] = 1.0
    
    # ── Subsample for testing ──
    adata_st, adata_st_list_raw = subsample_data(
        adata_st, adata_st_list_raw, SUBSAMPLE_PER_SLICE, N_NEIGHBORS)
    
    # ── Import STINR model ──
    # Add STINR to path
    stinr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               '..', 'STINR')  
    # Try common locations
    for candidate in [stinr_path, './STINR', '../STINR', 
                       os.path.expanduser('~/STINR')]:
        if os.path.isdir(candidate):
            sys.path.insert(0, os.path.abspath(candidate))
            break
    
    from STINR.model import Model
    
    # ── Patch the model for our use case ──
    # STINR hardcodes DLPFC-specific slice indices and file paths in model.py
    # We need to override the training steps and handle our device
    
    print("\n" + "=" * 70)
    print("Initializing STINR model...")
    print("=" * 70)
    print(f"  Spots: {adata_st.shape[0]:,}")
    print(f"  Genes: {adata_st.shape[1]}")
    print(f"  Cell types: {adata_basis.shape[0]}")
    print(f"  Slices: {len(sorted(set(adata_st.obs['slice'].values)))}")
    print(f"  Training steps: {TRAINING_STEPS}")
    
    model = Model(
        adata_st_list_raw=adata_st_list_raw,
        adata_st=adata_st,
        adata_basis=adata_basis,
        hidden_dims=[512, 128],
        n_heads=1,
        slice_emb_dim=16,
        coef_fe=0.1,
        training_steps=TRAINING_STEPS,
        lr=LR,
        seed=SEED,
        distribution="Poisson"
    )
    
    # Override training steps in the net
    model.net.training_steps = TRAINING_STEPS
    
    # ── Train ──
    print("\n" + "=" * 70)
    print("Training STINR...")
    print("=" * 70)
    
    # Note: STINR's train() has hardcoded DLPFC evaluation code.
    # We'll run the training loop manually to avoid that.
    model.net.train()
    
    from tqdm import tqdm
    for step in tqdm(range(TRAINING_STEPS)):
        loss, recon, denoise, Z_, ind_min, ind_max = model.net(
            coord=model.coord,
            adj_matrix=model.A,
            node_feats=model.X,
            count_matrix=model.Y,
            library_size=model.lY,
            slice_label=model.slice,
            basis=model.basis,
            step=step
        )
        
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        
        if step % 500 == 0:
            print(f"  Step {step}: loss = {loss.item():.4f}")
    
    # ── Evaluate ──
    print("\n" + "=" * 70)
    print("Evaluating...")
    print("=" * 70)
    
    result = model.eval(adata_st_list_raw, save=True, output_path=SAVE_DIR)
    
    # ── Clustering ──
    print("\nRunning GMM clustering on latent representations...")
    np.random.seed(SEED)
    gm = GaussianMixture(n_components=N_CLUSTERS, covariance_type='tied',
                         reg_covar=1e-3, init_params='kmeans', random_state=SEED)
    latent = model.adata_st.obsm['latent']
    y = gm.fit_predict(latent)
    model.adata_st.obs["GM_cluster"] = y
    
   # ── Save results ──
    print("\nSaving results...")
    model.adata_st.write_h5ad(os.path.join(SAVE_DIR, "adata_st_with_results.h5ad"))
    
    # Save deconvolution results per slice
    #slice_names = ['B01', 'E01', 'E02']
    slice_names = ['B01', 'C01', 'D01', 'E01', 'A02', 'B02', 'C02', 'D02', 'E02']
    for i, (adata_res, slice_name) in enumerate(zip(result, slice_names)):
        adata_res.write_h5ad(os.path.join(SAVE_DIR, f"result_slice_{slice_name}.h5ad"))
        
        # Print deconvolution summary
        print(f"\n  Slice {slice_name} deconvolution (top 5 cell types by mean proportion):")
        ct_cols = [c for c in adata_res.obs.columns if c in adata_basis.obs.index]
        if ct_cols:
            means = adata_res.obs[ct_cols].mean().sort_values(ascending=False)
            for ct, val in means.head(5).items():
                print(f"    {ct}: {val:.4f}")
    
    # Save latent representations
    latent = model.adata_st.obsm['latent']
    np.save(os.path.join(SAVE_DIR, "latent_representations.npy"), latent)
    
    # Save the trained model for interpolation later
    torch.save(model.net.state_dict(), os.path.join(SAVE_DIR, "stinr_model.pt"))
    
    print(f"\n{'='*70}")
    print(f"DONE! Results saved to {SAVE_DIR}/")
    print(f"  adata_st_with_results.h5ad  — combined with latent embeddings")
    print(f"  result_slice_*.h5ad         — per-slice deconvolution results")
    print(f"  latent_representations.npy  — latent embeddings")
    print(f"  stinr_model.pt              — trained model weights (for interpolation)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()