import open3d as o3d
import numpy as np
import torch


def load_point_cloud(filename, format="auto"):
    pc_entity = o3d.io.read_point_cloud(filename, format)
    pts = np.asarray(pc_entity.points)
    return pc_entity, pts


def visl_point_cloud(pc_entity):
    o3d.visualization.draw_geometries([pc_entity])


def visl_point_cloud_from_tensor(pts):
    pts = pts.squeeze(0).transpose(0, 1).cpu().detach().numpy()
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    visl_point_cloud(pc)


def save_point_cloud(filename, pc_entity):
    o3d.io.write_point_cloud(filename, pc_entity)


def calc_prior_matrix(g_prior, dim):
    b, _, n = g_prior.shape
    l_prior = torch.randn((b, dim, n), dtype=torch.float, device="cuda")
    return torch.concat((g_prior, l_prior), dim=1)


def adain(global_feats, styles):
    gamma, beta = styles.chunk(chunks=2, dim=1)
    return gamma * global_feats + beta


def get_edge_features(x, k, idx=None):
    b, dims, n = x.shape
    # * batched pair-wise distance
    if idx is None:
        xt = x.permute(0, 2, 1)
        xi = -2 * torch.bmm(xt, x)
        xs = torch.sum(xt ** 2, dim=2, keepdim=True)
        xst = xs.permute(0, 2, 1)
        dist = xi + xs + xst
        # * get knn id
        _, idx_o = torch.sort(dist, dim=2)
        idx = idx_o[:, :, 1:k+1]
        idx = idx.contiguous().view(b, n * k)
    # * gather
    neighbors = []
    for b in range(b):
        tmp = torch.index_select(x[b], 1, idx[b])
        tmp = tmp.view(dims, n, k)
        neighbors.append(tmp)
    neighbors = torch.stack(neighbors)
    # * centralize
    central = x.unsqueeze(3)
    central = central.repeat(1, 1, 1, k)
    return torch.cat([central, neighbors-central], dim=1)
