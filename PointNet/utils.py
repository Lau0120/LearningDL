import open3d as o3d
import numpy as np
import torch


def load_point_cloud(filename, format="auto"):
    pc_entity = o3d.io.read_point_cloud(filename, format)
    pts = np.asarray(pc_entity.points)
    return pc_entity, pts


def visl_point_cloud(pc_entity):
    o3d.visualization.draw_geometries([pc_entity])


def save_point_cloud(filename, pc_entity):
    o3d.io.write_point_cloud(filename, pc_entity)


def calc_squre_distance(fst, snd):
    B, N, _ = fst.shape
    _, M, _ = snd.shape
    dist = -2 * torch.matmul(fst, snd.permute(0, 2, 1))
    dist += torch.sum(fst ** 2, -1).view(B, N, 1)
    dist += torch.sum(snd ** 2, -1).view(B, 1, M)
    return dist


def get_points_from_indices(points, indices):
    device = points.device
    B = points.shape[0]
    view_shape = list(indices.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(indices.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, indices, :]
    return new_points


def farthest_point_sample(xyz, n_group):
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, n_group, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(n_group):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(b_radius, n_sample, xyz, new_xyz):
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = calc_squre_distance(new_xyz, xyz)
    group_idx[sqrdists > b_radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :n_sample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, n_sample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_group(n_group, b_radius, n_sample, xyz, points):
    B, _, C = xyz.shape
    S = n_group
    fps_idx = farthest_point_sample(xyz, n_group)
    new_xyz = get_points_from_indices(xyz, fps_idx)
    idx = query_ball_point(b_radius, n_sample, xyz, new_xyz)
    grouped_xyz = get_points_from_indices(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = get_points_from_indices(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points


def sample_group_all(xyz, points):
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points
