from re import A
import torch


class NTUGraph(object):
    def __init__(self):
        self.edges = [
            (1, 2),
            (2, 21),
            (3, 21),
            (4, 3),
            (5, 21),
            (6, 5),
            (7, 6),
            (8, 7),
            (9, 21),
            (10, 9),
            (11, 10),
            (12, 11),
            (13, 1),
            (14, 13),
            (15, 14),
            (16, 15),
            (17, 1),
            (18, 17),
            (19, 18),
            (20, 19),
            (22, 23),
            (23, 8),
            (24, 25),
            (25, 12),
        ]
        self.center = 20
        self.A = torch.eye(25)
        self.adj = dict()
        for u, v in self.edges:
            self.A[u - 1][v - 1] = 1
            self.A[v - 1][u - 1] = 1

            if u - 1 not in self.adj:
                self.adj[u - 1] = []
            if v - 1 not in self.adj:
                self.adj[v - 1] = []

            self.adj[u - 1].append(v - 1)
            self.adj[v - 1].append(u - 1)

        self.depth = [-1 for _ in range(25)]
        self.depth[self.center] = 0
        self._compute_depth(self.center, self.depth, self.adj)
        self.max_depth = max(self.depth)

        self.DA = [[self.A.clone(), self.A.clone(), self.center]]
        while True:
            MA, DA, newc = self.get_stride_A(
                self.DA[-1][1],
                self.DA[-1][2],
                stride=2,
                normalize=False,
            )
            if DA.size(0) == 1:
                break
            self.DA.append([MA, DA, newc])

    def _compute_depth(self, u, depth, adj):
        for v in adj[u]:
            if depth[v] != -1:
                continue
            depth[v] = depth[u] + 1
            self._compute_depth(v, depth, adj)

    def _dfs_stride(self, u, depth, adj, newadj, stride=1):
        if depth[u] % stride != 0:
            joints = []
            for v in adj[u]:
                if depth[v] <= depth[u]:
                    continue
                joints.extend(self._dfs_stride(v, depth, adj, newadj, stride))
            return joints
        joints = []
        for v in adj[u]:
            if depth[v] <= depth[u]:
                continue
            joints.extend(self._dfs_stride(v, depth, adj, newadj, stride))
        for v in joints:
            if u not in newadj:
                newadj[u] = []
            newadj[u].append(v)

            if v not in newadj:
                newadj[v] = []
            newadj[v].append(u)

        return [u]

    def get_compose(self, index):
        if index >= len(self.DA):
            raise ValueError(f"{index} is out of bound for NTUGraph.DA")
        return self.normalize_adj(self.DA[index][0]).unsqueeze(
            0
        ), self.get_khop(self.DA[index][1], True)

    def get_stride_A(self, A, center, stride=1, normalize=True):
        assert A.size(0) == A.size(1)
        adj = dict()
        for u in range(A.size(0)):
            for v in range(A.size(1)):
                if A[u][v] == 1:
                    if u not in adj:
                        adj[u] = []
                    adj[u].append(v)
        depth = [-1 for _ in range(A.size(0))]
        depth[center] = 0
        self._compute_depth(center, depth, adj)

        newadj = dict()
        newadj[center] = []
        self._dfs_stride(center, depth, adj, newadj, stride)

        MA = torch.zeros((len(adj), len(newadj)))
        DA = torch.eye(len(newadj))
        id_map = dict()
        for id, u in enumerate(newadj.keys()):
            id_map[u] = id
        for u in newadj.keys():
            for v in adj[u]:
                MA[v][id_map[u]] = 1
            MA[u][id_map[u]] = 1

        for u in newadj.keys():
            for v in newadj[u]:
                DA[id_map[u]][id_map[v]] = 1

        if normalize:
            MA = self.normalize_adj(MA)
            DA = self.normalize_adj(DA)

        return MA, DA, id_map[center]

    def get_num_joints(self):
        return self.A.size(0)

    def get_adj(self, normalize=True):
        if normalize:
            return self.normalize_adj(self.A.clone()).unsqueeze(0)
        else:
            return self.A.clone()

    def get_partial_adj(self):
        EA = self.get_eccentric_adj(True)
        CA = self.get_eccentric_adj(True)
        A = torch.eye(self.get_num_joints()).unsqueeze(0)
        return torch.cat([EA, A, CA], dim=0)

    def get_khop(self, A, normalize=True):
        A_k = []
        A_k.append(torch.eye(A.size(0)))
        for k in range(1, self.max_depth + 1):
            tmp = (
                torch.min(torch.matrix_power(A, k), torch.ones(1))
                - torch.min(torch.matrix_power(A, k - 1), torch.ones(1))
                + torch.eye(A.size(0))
            )
            if normalize:
                tmp = self.normalize_adj(tmp)
            A_k.append(tmp)

        A_k = torch.stack(A_k)
        return A_k

    def get_concentric_adj(self, normalize=True):
        A = self.A.clone()
        for i in range(A.size(0)):
            for j in range(A.size(1)):
                if A[i][j] == 1 and self.depth[j] > self.depth[i]:
                    A[i][j] = 0
        if normalize:
            return self.normalize_adj(A).unsqueeze(0)
        else:
            return A

    def get_eccentric_adj(self, normalize=True):
        A = self.A.clone()
        for i in range(A.size(0)):
            for j in range(A.size(1)):
                if A[i][j] == 1 and self.depth[j] < self.depth[i]:
                    A[i][j] = 0
        if normalize:
            return self.normalize_adj(A).unsqueeze(0)
        else:
            return A

    def get_khop_concentric_adj(self, normalize=True):
        A = self.get_concentric_adj(normalize=False)
        A = self.get_khop(A, normalize)
        return A

    def get_khop_eccentric_adj(self, normalize=True):
        A = self.get_eccentric_adj(normalize=False)
        A = self.get_khop(A, normalize)
        return A

    def normalize_adj(self, A):
        return A / torch.sum(A, dim=1, keepdim=True)
