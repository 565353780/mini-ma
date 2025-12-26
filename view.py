import numpy as np
import open3d as o3d

source_mesh_file_path = "/Users/chli/Downloads/matched_source_vertices.ply"
target_mesh_file_path = "/Users/chli/Downloads/matched_target_vertices.ply"
deformed_mesh_file_path = "/Users/chli/Downloads/deformed_mesh.ply"

# 读取两个点云文件
source_pcd = o3d.io.read_point_cloud(source_mesh_file_path)
target_pcd = o3d.io.read_point_cloud(target_mesh_file_path)
deformed_mesh = o3d.io.read_triangle_mesh(deformed_mesh_file_path)
deformed_mesh.compute_vertex_normals()
deformed_mesh.paint_uniform_color([0, 0, 1])

source_points = np.asarray(source_pcd.points)
target_points = np.asarray(target_pcd.points)

assert source_points.shape == target_points.shape, "源与目标点数量不一致"

max_deform_ratio = float('inf')

# 计算每对点的L2距离
deform_vectors = target_points - source_points
deform_lengths = np.linalg.norm(deform_vectors, axis=1)

# 计算source pcd的bbox最大边长
bbox_min = source_points.min(axis=0)
bbox_max = source_points.max(axis=0)
bbox_size = bbox_max - bbox_min
max_bbox_len = np.max(bbox_size)

# 超过阈值的过滤掉
deform_threshold = max_bbox_len * max_deform_ratio
valid_mask = deform_lengths <= deform_threshold

# 只保留符合要求的点
filtered_source_points = source_points[valid_mask]
filtered_target_points = target_points[valid_mask]
n_points = filtered_source_points.shape[0]

# 构建连线
all_points = np.vstack([filtered_source_points, filtered_target_points])
lines = [[i, i + n_points] for i in range(n_points)]
colors = [[0, 1, 0] for _ in range(len(lines))]

line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(all_points)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(colors)

# 源点和目标点着色
filtered_source_pcd = o3d.geometry.PointCloud()
filtered_source_pcd.points = o3d.utility.Vector3dVector(filtered_source_points)
filtered_source_pcd.paint_uniform_color([1, 0, 0])   # 红色

filtered_target_pcd = o3d.geometry.PointCloud()
filtered_target_pcd.points = o3d.utility.Vector3dVector(filtered_target_points)
filtered_target_pcd.paint_uniform_color([0, 0, 1])   # 蓝色

o3d.visualization.draw_geometries(
    [filtered_source_pcd, filtered_target_pcd, line_set, deformed_mesh],
    window_name='点匹配连线',
    point_show_normal=False
)
