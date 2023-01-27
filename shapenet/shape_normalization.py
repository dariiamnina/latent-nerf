# Normalize 3d shapes
# 1. Compute centre of gravity
# 2. Compute orientation of the principal axis
# 3. Compute size of the bounding box
import trimesh
import numpy as np
from skimage.measure import marching_cubes

DELTA = 0.2
shape_path = '../shapenet_samples/shapenet_dim32_df/02691156/10155655850468db78d106ce0a280f87__0__.df'

shape = np.fromfile(shape_path, dtype=np.uint64, offset=0, count=3)
df = np.fromfile(shape_path, dtype=np.float32, offset=3*8)
df = df.reshape(shape)
input_mesh = marching_cubes(df, level=1)
v, f = input_mesh[0].copy(), input_mesh[1].copy()

mesh = trimesh.Trimesh(vertices=v,
                       faces=f)
print(mesh.is_watertight)
print(mesh.center_mass)
mesh.vertices -= mesh.center_mass

#PCA


def normalize_mesh(v, target_scale=0.5):
    verts = v
    # Compute center of bounding box
    # center = torch.mean(torch.column_stack([torch.max(verts, dim=0)[0], torch.min(verts, dim=0)[0]]))
    center = verts.mean(axis=0)
    verts = verts - center
    scale = np.max(np.linalg.norm(verts, axis=1))
    verts = (verts / scale) * target_scale
    return verts

normalized_vertices = normalize_mesh(mesh.vertices, 0.7)
#normalized_verices, mesh.faces

