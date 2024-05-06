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
        self.depth = [-1 for _ in range(25)]
        self.depth[self.center] = 0
        self._compute_depth(self.center)
        self.max_depth = max(self.depth)
        self.A = torch.eye(25)
        for u, v in self.edges:
            self.A[u - 1][v - 1] = 1
            self.A[v - 1][u - 1] = 1

    def _compute_depth(self, u):
        for e in self.edges:
            if e[0] - 1 == u:
                v = e[1] - 1
            elif e[1] - 1 == u:
                v = e[0] - 1
            else:
                continue

            if self.depth[v] != -1:
                continue
            self.depth[v] = self.depth[u] + 1

            self._compute_depth(v)

    def get_num_joints(self):
        return self.A.size(0)

    def get_adj(self, normalize=True):
        if normalize:
            return self.normalize_adj(self.A.clone()).unsqueeze(0)
        else:
            return self.A.clone()

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
