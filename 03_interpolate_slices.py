
import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import torch
from scipy.sparse import issparse


DATA_DIR = "Library/CloudStorage/OneDrive-InsideMDAnderson/linghua_all/stinr_data"              
RESULTS_DIR = "Library/CloudStorage/OneDrive-InsideMDAnderson/linghua_all/stinr_results"         
OUTPUT_DIR = "Library/CloudStorage/OneDrive-InsideMDAnderson/linghua_all/interpolated_slices"

# STINR repo path
STINR_PATH = "/Users/mniger/Library/CloudStorage/OneDrive-InsideMDAnderson/linghua_all/STINR/STINR/"                
sys.path.insert(0, STINR_PATH)
 
# Interpolation settings
Z_STEP = 5.0        # µm between virtual slices
Z_MIN = 0.0         # B01
Z_MAX = 80.0        # E02
 
# Real slice z-values
#REAL_SLICES = {
#   'B01': 0.0,
#  'E01': 30.0,
#    'E02': 80.0
#}
 
REAL_SLICES = {
   'B01': 0.0,
   'E01': 30.0,
   'E02': 80.0
}
# Must match training settings
SUBSAMPLE_PER_SLICE = None  # set to None if trained on full data
N_NEIGHBORS = 6
SEED = 42
 
DEVICE = "cpu"  # CPU is safer for inference; MPS can cause issues
BATCH_SIZE = 10000  # process virtual spots in batches to save memory

 
 
def load_trained_model(data_dir, results_dir, subsample_per_slice, n_neighbors):
    
    from sklearn.neighbors import kneighbors_graph
    
    print("Loading data...")
    adata_st = ad.read_h5ad(os.path.join(data_dir, "adata_st.h5ad"))
    adata_basis = ad.read_h5ad(os.path.join(data_dir, "adata_basis.h5ad"))
    
    adata_st_list_raw = []
    for i in range(3):
        adata_st_list_raw.append(
            ad.read_h5ad(os.path.join(data_dir, f"adata_st_list_raw{i}.h5ad")))
    
    # Handle zero library size
    zero_mask = adata_st.obs['library_size'].values == 0
    if zero_mask.sum() > 0:
        adata_st.obs.loc[zero_mask, 'library_size'] = 1.0
    
    # Subsample if training used subsampling
    if subsample_per_slice is not None:
        print(f"Subsampling to {subsample_per_slice} per slice (matching training)...")
        np.random.seed(SEED)
        import random
        random.seed(SEED)
        
        slices = sorted(adata_st.obs['slice'].unique())
        keep_indices = []
        for s in slices:
            s_mask = np.where(adata_st.obs['slice'].values == s)[0]
            n_take = min(subsample_per_slice, len(s_mask))
            chosen = np.random.choice(s_mask, n_take, replace=False)
            chosen.sort()
            keep_indices.append(chosen)
        keep_indices = np.concatenate(keep_indices)
        
        X_full = adata_st.X[keep_indices].toarray() if issparse(adata_st.X) else adata_st.X[keep_indices]
        
        adata_sub = ad.AnnData(
            X=X_full,
            obs=adata_st.obs.iloc[keep_indices].reset_index(drop=True),
            var=adata_st.var.copy()
        )
        adata_sub.obsm['3D_coor'] = adata_st.obsm['3D_coor'][keep_indices]
        adata_sub.obsm['count'] = adata_st.obsm['count'][keep_indices]
        adata_sub.obs.index = [f"spot_{i}" for i in range(len(adata_sub))]
        
        # Rebuild graph
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
        
        # Rebuild per-slice raw
        adata_st_list_raw_sub = []
        for s_idx, s in enumerate(slices):
            s_mask = np.where(adata_sub.obs['slice'].values == s)[0]
            raw_X = adata_sub.obsm['count'][s_mask]
            adata_raw_s = ad.AnnData(
                X=raw_X,
                obs=adata_sub.obs.iloc[s_mask].reset_index(drop=True),
                var=adata_sub.var.copy()
            )
            adata_raw_s.obs.index = [f"spot_{i}-slice{s}" for i in range(len(adata_raw_s))]
            adata_raw_s.obsm['spatial'] = adata_sub.obsm['3D_coor'][s_mask, :2]
            adata_st_list_raw_sub.append(adata_raw_s)
        
        adata_st = adata_sub
        adata_st_list_raw = adata_st_list_raw_sub
    
    print(f"  adata_st: {adata_st.shape}")
    print(f"  adata_basis: {adata_basis.shape}")
    
    # Reconstruct model
    print("Reconstructing model architecture...")
    from STINR.model import Model
    
    model = Model(
        adata_st_list_raw=adata_st_list_raw,
        adata_st=adata_st,
        adata_basis=adata_basis,
        hidden_dims=[512, 128],
        n_heads=1,
        slice_emb_dim=16,
        coef_fe=0.1,
        training_steps=1,  # doesn't matter for inference
        lr=0.001,
        seed=SEED,
        distribution="Poisson"
    )
    
    # Load trained weights
    weights_path = os.path.join(results_dir, "stinr_model.pt")
    print(f"Loading weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=True)
    model.net.load_state_dict(state_dict)
    model.net.eval()
    print("  Model loaded successfully!")
    
    return model, adata_st, adata_basis
 
 
def get_xy_positions_for_z(adata_st, target_z, real_slices):
    
    # Find nearest real slice
    real_z_values = sorted(real_slices.values())
    nearest_z = min(real_z_values, key=lambda rz: abs(rz - target_z))
    
    # Find the slice index corresponding to nearest_z
    slice_names = list(real_slices.keys())
    slice_z_values = [real_slices[s] for s in slice_names]
    nearest_slice_idx = slice_z_values.index(nearest_z)
    
    # Get spots from that slice
    slice_mask = adata_st.obs['slice'].values == nearest_slice_idx
    xy_positions = adata_st.obsm['3D_coor'][slice_mask, :2].copy()
    
    return xy_positions, nearest_slice_idx
 
 
def interpolate_at_z(model, adata_st, target_z, xy_positions, batch_size, device):
   
    n_spots = xy_positions.shape[0]
    
    # Build 3D coordinates: (x, y, target_z)
    coords_3d = np.zeros((n_spots, 3), dtype=np.float64)
    coords_3d[:, 0] = xy_positions[:, 0]
    coords_3d[:, 1] = xy_positions[:, 1]
    coords_3d[:, 2] = target_z
    
    # Find nearest real slice index for slice embedding
    nearest_slice_idx = 0
    min_dist = float('inf')
    for s_name, s_z in REAL_SLICES.items():
        s_idx = list(REAL_SLICES.keys()).index(s_name)
        dist = abs(s_z - target_z)
        if dist < min_dist:
            min_dist = dist
            nearest_slice_idx = s_idx
    
    with torch.no_grad():
        model.net.eval()
        
        all_beta = []
        all_latent = []
        
        for start in range(0, n_spots, batch_size):
            end = min(start + batch_size, n_spots)
            batch_coords = torch.from_numpy(coords_3d[start:end]).float().to(device)
            
            # Scale coordinates the same way as training (line 118: self.coord = coord/100)
            batch_coords_scaled = batch_coords / 100.0
            
            # encoder_layer0: SIREN maps coordinates to mid-features
            mid_fea = model.net.encoder_layer0(batch_coords_scaled)
            
            # encoder_layer1: mid-features to latent Z
            Z = model.net.encoder_layer1(mid_fea)
            
            # deconv_beta_layer: latent Z to cell-type proportions
            beta = model.net.deconv_beta_layer(torch.sin(Z))
            beta = torch.nn.functional.softmax(beta, dim=1)
            
            all_beta.append(beta.cpu().numpy())
            all_latent.append(Z.cpu().numpy())
    
    beta_matrix = np.vstack(all_beta)
    latent_matrix = np.vstack(all_latent)
    
    return beta_matrix, latent_matrix, coords_3d
 
 
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("STINR 3D Interpolation")
    print("=" * 70)
    
    
    model, adata_st, adata_basis = load_trained_model(
        DATA_DIR, RESULTS_DIR, SUBSAMPLE_PER_SLICE, N_NEIGHBORS)
    
    celltypes = list(adata_basis.obs.index)
    print(f"  Cell types ({len(celltypes)}): {celltypes}")
    
    
    z_values = np.arange(Z_MIN, Z_MAX + Z_STEP / 2, Z_STEP)
    real_z = set(REAL_SLICES.values())
    print(f"\n  Interpolating at {len(z_values)} z-values: {z_values.tolist()}")
    print(f"  Real slices at z = {sorted(real_z)}")
    print(f"  Virtual slices at z = {sorted(set(z_values) - real_z)}")
    
    
    all_results = []
    
    for z in z_values:
        is_real = z in real_z
        label = "REAL" if is_real else "VIRTUAL"
        
        print(f"\n  Processing z = {z:.1f} µm ({label})...")
        
        # Get XY positions from nearest real slice
        xy_positions, nearest_idx = get_xy_positions_for_z(
            adata_st, z, REAL_SLICES)
        print(f"    Using {len(xy_positions):,} spots from nearest slice (idx {nearest_idx})")
        
        # Query model
        beta_matrix, latent_matrix, coords_3d = interpolate_at_z(
            model, adata_st, z, xy_positions, BATCH_SIZE, DEVICE)
        
        # Assign dominant cell type
        dominant_ct_idx = beta_matrix.argmax(axis=1)
        dominant_ct = [celltypes[i] for i in dominant_ct_idx]
        
        # Build result adata
        obs_df = pd.DataFrame({
            'cell_type_predicted': dominant_ct,
            'z': z,
            'is_real_slice': is_real,
            'nearest_real_slice': nearest_idx,
        })
        
        # Add cell-type proportions as columns
        for j, ct in enumerate(celltypes):
            obs_df[ct] = beta_matrix[:, j]
        
        adata_slice = ad.AnnData(
            X=beta_matrix.astype(np.float32),
            obs=obs_df,
            var=pd.DataFrame(index=celltypes)
        )
        adata_slice.obsm['spatial_3d'] = coords_3d.astype(np.float32)
        adata_slice.obsm['spatial'] = coords_3d[:, :2].astype(np.float32)
        adata_slice.obsm['latent'] = latent_matrix.astype(np.float32)
        
        # Save individual slice
        z_str = f"{z:05.1f}".replace('.', '_')
        adata_slice.write_h5ad(os.path.join(OUTPUT_DIR, f"slice_z{z_str}.h5ad"))
        
        # Print summary
        ct_counts = pd.Series(dominant_ct).value_counts()
        top3 = ct_counts.head(3)
        print(f"    Top 3 cell types: {', '.join(f'{ct}: {n}' for ct, n in top3.items())}")
        
        all_results.append(adata_slice)
    
    # Save combined result
    print(f"\n{'='*70}")
    print("Combining all slices...")
    adata_combined = ad.concat(all_results, join='outer')
    adata_combined.write_h5ad(os.path.join(OUTPUT_DIR, "all_interpolated_slices.h5ad"))
    
    # summary 
    print(f"\n{'='*70}")
    print("INTERPOLATION COMPLETE!")
    print(f"{'='*70}")
    print(f"  Total slices: {len(z_values)} ({len(real_z)} real + {len(z_values) - len(real_z)} virtual)")
    print(f"  Total spots: {sum(a.shape[0] for a in all_results):,}")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"    all_interpolated_slices.h5ad  — combined")
    print(f"    slice_z*.h5ad                 — individual slices")
    print(f"\n  Cell type distribution across all slices:")
    
    all_cts = pd.concat([a.obs['cell_type_predicted'] for a in all_results])
    ct_summary = all_cts.value_counts()
    for ct, n in ct_summary.items():
        pct = n / len(all_cts) * 100
        print(f"    {ct:25s}: {n:>8,} ({pct:5.1f}%)")
    
    
 
if __name__ == "__main__":
    main()
