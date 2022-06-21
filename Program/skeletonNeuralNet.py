import numpy as np
import hashlib

seed = hashlib.sha256(b"Ignore mimes").hexdigest()
np.random.seed(int(seed, 36) & 0XFFFFFFFF)

class skeletonNet:
    def __init__(self, *args):
        self.shape = args
        self.skeleton = self.neuralNetArray()

    def neuralNetArray(self):
        out = []
        for i in self.shape:
            out.append([np.random.uniform(-1,1) for i in range(i)])
        return out