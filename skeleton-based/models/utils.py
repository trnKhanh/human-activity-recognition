import torch


def build_A(graph_args):
    type = graph_args["partition_type"]
    if type == "spatial":
        num_joints = graph_args["num_joints"]
        edges = graph_args["edges"]
        center = graph_args["center"]
        A = torch.zeros((3, num_joints, num_joints))
        adj = torch.zeros((num_joints, num_joints))
        for e in edges:
            adj[e[0] - 1, e[1] - 1] = 1
            adj[e[1] - 1, e[0] - 1] = 1
        r = torch.tensor([-1 for _ in range(num_joints)])
        r[center - 1] = 0
        dfs(center - 1, adj, r)

        for u in range(num_joints):
            A[1][u][u] = 1
            for v in range(num_joints):
                if adj[u][v]:
                    if r[u] > r[v]:
                        A[0][u][v] = 1
                    elif r[u] < r[v]:
                        A[2][u][v] = 1
        return A
    else:
        raise NotImplemented()


def dfs(u, adj, r):
    for v in range(adj.size()[1]):
        if adj[u][v] and r[v] == -1:
            r[v] = r[u] + 1
            dfs(v, adj, r)
