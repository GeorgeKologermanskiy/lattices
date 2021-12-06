import numpy as np

class CloseByOneAlgorithm:

    def __init__(self):
        self.G = None
        self.M = None
        self.I = None
        self.L = None

    def __call__(self, context):
        G, M = context.shape
        I = context
        # ({0, ... , G-1}, {0, ... , M-1}, I) - context
        self.G = set(np.arange(G))
        self.M = set(np.arange(M))
        self.I = (I == 1)
        self.L = [
            (set(), self.M)
        ]
        for g in range(G):
            D = self.__close_g_once(g)
            C = self.__close_M(D)
            self.__process(set([g]), g, (C, D))
        return self.L

    def __close_g_once(self, g):
        return set(np.where(self.I[g])[0])

    def __close_M(self, Y: set):
        return set(np.where(np.all(self.I[:, list(Y)], axis=1))[0])

    def __process(self, A, g, P):
        C, D = P
        dist = C - A
        if len(dist) > 0 and min(dist) < g:
            return
        self.L.append(P)
        for f in (self.G - C):
            if f <= g:
                continue
            Z = set.union(C, set([f]))
            Y = set.intersection(D, self.__close_g_once(f))
            X = self.__close_M(Y)
            self.__process(Z, f, (X, Y))


if __name__ == '__main__':
    # DEBUG
    context = np.array([
        [0, 1, 0, 0, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 1, 1, 0, 0, 1]
    ])

    alg = CloseByOneAlgorithm()
    L = alg(context)
    print(len(L))
    print(L)