import os
import numpy as np


class Graph(object):
    graph_path = "../../resources/ntu_graph.txt"

    def __init__(self, center: int = 20, graph_path=None):
        self.center = center
        self.depth = []
        if graph_path is not None:
            self.graph_path = graph_path
        else:
            self.graph_path = os.path.join(
                os.path.dirname(__file__), self.graph_path
            )

        self.__prepare_graph(self.graph_path)

    def __prepare_graph(self, graph_path: str):
        # Load graph from file
        self.A = self.__load_graph(graph_path)
        self.num_joints = self.A.shape[0]

        # Compute the depth of all joints from center
        self.depth = dict()
        self.depth[self.center] = 0
        self.__compute_depth(self.center, self.A, self.depth)

    def __load_graph(self, graph_path):
        with open(graph_path, "r") as f:
            lines = f.readlines()
            num_joints = int(lines[0])
            A = np.eye(num_joints, dtype=np.float32)
            for line in lines[1:]:
                u, v = [int(s) - 1 for s in line.split(" ")]
                A[u, v] = 1
                A[v, u] = 1
        return A

    def __compute_depth(self, u, A, depth):
        num_joints = A.shape[0]
        for v in range(num_joints):
            if A[u, v] == 0 or v in depth:
                continue
            depth[v] = depth[u] + 1
            self.__compute_depth(v, A, depth)

    def __normalize_adj(self, A: np.ndarray):
        Dl = np.sum(A, 0)
        h, w = A.shape
        Dn = np.zeros((w, w))
        for i in range(w):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)
        return AD

    def __spatial_partition(self, A, depth):
        self_link = np.zeros_like(A)
        inward = np.zeros_like(A)
        outward = np.zeros_like(A)

        num_joints = A.shape[0]
        for u in range(num_joints):
            for v in range(num_joints):
                if A[u, v] == 0:
                    continue

                if depth[u] == depth[v]:
                    self_link[u, v] = A[u, v]
                elif depth[u] > depth[v]:
                    inward[u, v] = A[u, v]
                else:
                    outward[u, v] = A[u, v]

        return self_link, inward, outward

    def get_A(self, normalize=True):
        if normalize:
            return self.__normalize_adj(self.A)
        else:
            return self.A

    def get_spatial_A(self, normalize=True):
        self_link, inward, outward = self.__spatial_partition(
            self.A, self.depth
        )

        SA = np.stack([self_link, inward, outward], axis=0)

        if normalize:
            for i in range(SA.shape[0]):
                SA[i] = self.__normalize_adj(SA[i])

        return SA
