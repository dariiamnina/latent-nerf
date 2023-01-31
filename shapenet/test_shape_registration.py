import numpy as np
import os
import trimesh
from shape_normalization import normalize_mesh


shapes_dir = 'C:/Users/darii/Downloads/nearest_neighbor_search_via_single_image/result/'
dir_filenames_dict = {dir:os.listdir(shapes_dir + dir) for dir in os.listdir(shapes_dir)}
costs = {}

total_filename_list = []
for key, value in dir_filenames_dict.items():
    class_dir = key
    filename_list = value
    total_filename_list += [class_dir + '/' + filename for filename in filename_list]

only_first = True
for key, value in dir_filenames_dict.items():
    if not only_first:
        break
    class_dir = key
    filename_list = value

    for filename in filename_list:
        if filename.endswith('.obj'):
            object_costs_dict = {class_dir + '/' + filename: {}}

            mesh = trimesh.load(shapes_dir + class_dir + '/' + filename, force='mesh')
            #mesh = MeshOBJ(v, f)
            v_normalized = normalize_mesh(mesh.vertices, 0.7)
            mesh = trimesh.Trimesh(vertices=v_normalized, faces=mesh.faces)

            for dir2_filename2 in total_filename_list:
                if dir2_filename2.endswith('.obj') and dir2_filename2 != filename:
                    other = trimesh.load(shapes_dir + dir2_filename2, force='mesh')
                    cost = trimesh.registration.mesh_other(mesh, other, samples=100)

                    object_costs_dict[class_dir + '/' + filename][shapes_dir + dir2_filename2] = cost
            costs[filename] = object_costs_dict
    only_first = False
import pickle
with open('costs.pickle', 'wb') as fo:
    pickle.dump(costs, fo)
print(costs)




