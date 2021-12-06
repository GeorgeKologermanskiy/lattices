from close_by_one import CloseByOneAlgorithm
import numpy as np

class modelGALOIS():

    def __init__(self):
        self.concepts = None
        self.Y = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        alg = CloseByOneAlgorithm()
        L = alg(X)
        M = len(X[0])
        self.concepts = []
        self.marks = []
        for A, B in L:
            if len(A) == 0:
                continue
            mark = Y[A.__iter__().__next__()]
            if (mark == Y[list(A)]).all():
                new_row = np.zeros(M)
                new_row[list(B)] = 1
                self.concepts.append(new_row)
                self.marks.append(mark)
        self.concepts = np.array(self.concepts)
        self.marks = np.array(self.marks)

    def predict(self, X: np.ndarray):
        X_t = (1 - X).T
        res = self.concepts @ X_t
        idx = np.argmin(res, axis=0)
        return self.marks[idx]

if __name__ == '__main__':
    X = np.array([
        [0, 1, 0, 0, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 1, 1, 0]
    ])
    Y = np.array(
        [1, 1, 1, 1, 0, 0, 0]
    )

    model = modelGALOIS()
    model.fit(X, Y)

    print(model.predict(X))

    X = np.array([
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 1], # egg
        [0, 1, 0, 0, 0, 1, 1, 0, 0, 1] # mango
    ])
    print(model.predict(X))
