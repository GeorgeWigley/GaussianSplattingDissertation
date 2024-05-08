import open3d as o3d
import numpy as np

def convert_colmap_to_pcd(colmap_points_3d_fp, output_fp):
    points = []
    cols = []

    with open(colmap_points_3d_fp) as f:
        for line in f:
            tokens = line.split(' ')
            # check for comments
            if "#" in tokens[0]:
                continue

            points.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            cols.append(np.divide([float(tokens[4]), float(tokens[5]), float(tokens[6])], 255))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(cols))
    o3d.io.write_point_cloud(output_fp, pcd)
    o3d.visualization.draw_geometries([pcd])

def convert_pcd_to_colmap(pcd_fp, colmap_points_3d_fp):
    pcd_load = o3d.io.read_point_cloud(pcd_fp)
    o3d.visualization.draw_geometries([pcd_load])
    with open(colmap_points_3d_fp, 'w') as f:
        for i in range(len(pcd_load.points)):
            point = pcd_load.points[i]
            col = pcd_load.colors[i]

            line = f"{i + 1} {point[0]} {point[1]} {point[2]} {int(col[0] * 255)} {int(col[1] * 255)} {int(col[2] * 255)} 0"
            f.write(line + "\n")

def vis_pcd(fp):
    pcd_load = o3d.io.read_point_cloud(fp)
    o3d.visualization.draw_geometries([pcd_load])


colmap_fp = r"C:\Users\georg\Documents\diss_final_experiments\execution\final-pipeline\sfm\sparse\0\points3D.txt"
out_fp = r"C:\Users\georg\Documents\diss_final_experiments\execution\final-pipeline\combined\sfm_pcd_it.ply"
adjusted_pcd = r"C:\Users\georg\Documents\diss_final_experiments\execution\final-pipeline\combined\adjusted.ply"
adjusted_colmap_fp = r"C:\Users\georg\Documents\diss_final_experiments\execution\final-pipeline\combined\sparse\0\points3D.txt"

# convert_colmap_to_pcd(adjusted_colmap_fp, out_fp)
convert_pcd_to_colmap(adjusted_pcd, adjusted_colmap_fp)
# vis_pcd(adjusted_pcd)