
import os
import numpy as np
import pandas as pd
import anndata as ad
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
import gc

INTERPOLATED_DIR = "Library/CloudStorage/OneDrive-InsideMDAnderson/linghua_all/interpolated_slices"
OUTPUT_DIR = "Library/CloudStorage/OneDrive-InsideMDAnderson/linghua_all/STINR_meshes_3d"

# Density grid parameters 
GRID_RESOLUTION = 300      # number of bins along x and y axes
N_Z_INTERP = 300           # number of interpolation points along z
CELL_SIGMA_XY = 3.0        # Gaussian sigma in x,y (in grid units)
CELL_SIGMA_Z = 12.0         # Gaussian sigma in z (in grid units)
LEVEL_FRAC = 0.1         # isosurface threshold as fraction of max density


CELL_TYPES_TO_MESH = None  # None = all, or e.g. ['Tumor-PCNA', 'Tumor-PD-L1', 'Fibro', 'SMC']


USE_PROPORTIONS = True  # False = use dominant cell type (like your original pipeline)

def load_interpolated_data(interp_dir):
    
    print("Loading interpolated slices...")
    
    adata = ad.read_h5ad(os.path.join(interp_dir, "all_interpolated_slices.h5ad"))
    
    # Extract coordinates and cell types
    coords_3d = adata.obsm['spatial_3d']
    cell_types = adata.obs['cell_type_predicted'].values
    z_values = adata.obs['z'].values
    
    print(f"  Total spots: {len(cell_types):,}")
    print(f"  Unique z-values: {sorted(np.unique(z_values))}")
    print(f"  Coordinate ranges:")
    print(f"    x: [{coords_3d[:,0].min():.1f}, {coords_3d[:,0].max():.1f}]")
    print(f"    y: [{coords_3d[:,1].min():.1f}, {coords_3d[:,1].max():.1f}]")
    print(f"    z: [{coords_3d[:,2].min():.1f}, {coords_3d[:,2].max():.1f}]")
    
    unique_cts = sorted(np.unique(cell_types))
    print(f"  Cell types ({len(unique_cts)}): {unique_cts}")
    
    return coords_3d, cell_types, z_values, unique_cts, adata


def build_density_volume(coords_3d, cell_types, target_ct,
                         grid_res, n_z_interp, sigma_xy, sigma_z,
                         x_range, y_range, z_range):
    
    # Filter to target cell type
    ct_mask = cell_types == target_ct
    ct_coords = coords_3d[ct_mask]
    n_cells = ct_mask.sum()
    
    if n_cells == 0:
        return None, None
    
    # Map coordinates to grid indices
    # cell_x → axis 0 (vertical), cell_y → axis 1 (horizontal), z → axis 2
    x_idx = ((ct_coords[:, 0] - x_range[0]) / (x_range[1] - x_range[0]) * (grid_res - 1)).astype(int)
    y_idx = ((ct_coords[:, 1] - y_range[0]) / (y_range[1] - y_range[0]) * (grid_res - 1)).astype(int)
    z_idx = ((ct_coords[:, 2] - z_range[0]) / (z_range[1] - z_range[0]) * (n_z_interp - 1)).astype(int)
    
    # Clip to grid bounds
    x_idx = np.clip(x_idx, 0, grid_res - 1)
    y_idx = np.clip(y_idx, 0, grid_res - 1)
    z_idx = np.clip(z_idx, 0, n_z_interp - 1)
    
    # Accumulate density
    density = np.zeros((grid_res, grid_res, n_z_interp), dtype=np.float32)
    np.add.at(density, (x_idx, y_idx, z_idx), 1)
    
    # Gaussian smoothing
    density = gaussian_filter(density, sigma=[sigma_xy, sigma_xy, sigma_z])
    
    grid_info = {
        'x_range': x_range,
        'y_range': y_range,
        'z_range': z_range,
        'grid_res': grid_res,
        'n_z_interp': n_z_interp,
    }
    
    return density, grid_info


def extract_mesh(density, grid_info, level_frac):
    
    threshold = density.max() * level_frac
    
    if threshold <= 0:
        return None, None
    
    try:
        verts, faces, normals, values = marching_cubes(density, level=threshold)
    except ValueError:
        # No surface found at this threshold
        return None, None
    
    # Map grid coordinates back to physical coordinates
    x_range = grid_info['x_range']
    y_range = grid_info['y_range']
    z_range = grid_info['z_range']
    grid_res = grid_info['grid_res']
    n_z = grid_info['n_z_interp']
    
    verts_physical = np.zeros_like(verts)
    verts_physical[:, 0] = verts[:, 0] / (grid_res - 1) * (x_range[1] - x_range[0]) + x_range[0]
    verts_physical[:, 1] = verts[:, 1] / (grid_res - 1) * (y_range[1] - y_range[0]) + y_range[0]
    verts_physical[:, 2] = verts[:, 2] / (n_z - 1) * (z_range[1] - z_range[0]) + z_range[0]
    
    return verts_physical, faces


def save_obj(vertices, faces, filepath):
    
    with open(filepath, 'w') as f:
        f.write(f"# Cell type mesh\n")
        f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n")
        for v in vertices:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for face in faces:
            # OBJ is 1-indexed
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("3D Mesh Generation from STINR Interpolated Data")
    print("=" * 70)
    
    
    coords_3d, cell_types, z_values, unique_cts, adata = load_interpolated_data(INTERPOLATED_DIR)
    
    # Determine cell types to process
    if CELL_TYPES_TO_MESH is not None:
        cts_to_process = [ct for ct in CELL_TYPES_TO_MESH if ct in unique_cts]
    else:
        cts_to_process = unique_cts
    
    print(f"\n  Will generate meshes for {len(cts_to_process)} cell types")
    
    # Global coordinate ranges
    x_range = (coords_3d[:, 0].min(), coords_3d[:, 0].max())
    y_range = (coords_3d[:, 1].min(), coords_3d[:, 1].max())
    z_range = (coords_3d[:, 2].min(), coords_3d[:, 2].max())
    
    # Free the adata to save memory
    del adata
    gc.collect()
    
    
    mesh_summary = []
    
    for ct_idx, ct in enumerate(cts_to_process):
        print(f"\n{'─'*50}")
        print(f"[{ct_idx+1}/{len(cts_to_process)}] {ct}")
        print(f"{'─'*50}")
        
        n_cells = (cell_types == ct).sum()
        print(f"  Spots: {n_cells:,}")
        
        if n_cells < 100:
            print(f"  Skipping — too few spots for mesh generation")
            continue
        
        # Build density volume
        print(f"  Building density volume ({GRID_RESOLUTION}×{GRID_RESOLUTION}×{N_Z_INTERP})...")
        density, grid_info = build_density_volume(
            coords_3d, cell_types, ct,
            GRID_RESOLUTION, N_Z_INTERP,
            CELL_SIGMA_XY, CELL_SIGMA_Z,
            x_range, y_range, z_range
        )
        
        if density is None:
            print(f"  Skipping — no data")
            continue
        
        print(f"  Density range: [{density.min():.4f}, {density.max():.4f}]")
        
        # Extract mesh
        print(f"  Extracting isosurface (level_frac={LEVEL_FRAC})...")
        vertices, faces = extract_mesh(density, grid_info, LEVEL_FRAC)
        
        del density
        gc.collect()
        
        if vertices is None:
            print(f"  Skipping — no surface found at threshold")
            continue
        
        print(f"  Mesh: {len(vertices):,} vertices, {len(faces):,} faces")
        
        # Save OBJ
        ct_safe = ct.replace('-', '_').replace(' ', '_')
        obj_path = os.path.join(OUTPUT_DIR, f"{ct_safe}.obj")
        save_obj(vertices, faces, obj_path)
        print(f"  Saved: {obj_path}")
        
        mesh_summary.append({
            'cell_type': ct,
            'n_spots': n_cells,
            'n_vertices': len(vertices),
            'n_faces': len(faces),
            'file': obj_path
        })
    
    
    print(f"\n{'='*70}")
    print("MESH GENERATION COMPLETE!")
    print(f"{'='*70}")
    print(f"  Output directory: {OUTPUT_DIR}/")
    print(f"  Meshes generated: {len(mesh_summary)}")
    print(f"\n  {'Cell Type':<25s} {'Spots':>10s} {'Vertices':>10s} {'Faces':>10s}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10}")
    for m in mesh_summary:
        print(f"  {m['cell_type']:<25s} {m['n_spots']:>10,} {m['n_vertices']:>10,} {m['n_faces']:>10,}")
    
    # Save summary CSV
    summary_df = pd.DataFrame(mesh_summary)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "mesh_summary.csv"), index=False)
    
    
    coord_info = {
        'x_range': list(x_range),
        'y_range': list(y_range),
        'z_range': list(z_range),
        'grid_resolution': GRID_RESOLUTION,
        'n_z_interp': N_Z_INTERP,
        'sigma_xy': CELL_SIGMA_XY,
        'sigma_z': CELL_SIGMA_Z,
        'level_frac': LEVEL_FRAC,
        'n_cell_types': len(mesh_summary),
        'total_spots': len(cell_types),
    }
    import json
    with open(os.path.join(OUTPUT_DIR, "coord_info.json"), 'w') as f:
        json.dump(coord_info, f, indent=2)
    
    

if __name__ == "__main__":
    main()
