import numpy as np


def inverse(q: list[float]) -> list[float]:
    """

    :type q: list[float] quaternion
    """
    for i in range(1, 4):
        q[i] = -1 * q[i]
    return q


def rotate(x, q, mode: str = 'active'):
    """

        :param q: quaternion object of class Q
        :param mode: passive or active rotation mode
        :param x: quaternion to be rotated by q
        """
    assert mode in ['active', 'passive'], 'mode should be active or passive'
    if mode == 'active':
        return q.inverse.prod(x.x).prod(q.x)


class Q:
    def __init__(self, x: list[float]):
        self.x = x
        self.res = [0, 0, 0, 0]

    def prod(self, y):
        # t0 = (r0s0 − r1s1 − r2s2 − r3s3)
        self.res[0] = self.x[0] * y[0] - self.x[1] * y[1] - self.x[2] * y[2] - self.x[3] * y[3]
        # t1 = (r0s1 + r1s0 − r2s3 + r3s2)
        self.res[1] = self.x[0] * y[1] + self.x[1] * y[0] - self.x[2] * y[3] + self.x[3] * y[2]
        # t2 = (r0s2 + r1s3 + r2s0 − r3s1)
        self.res[2] = self.x[0] * y[2] + self.x[1] * y[3] + self.x[2] * y[0] - self.x[3] * y[1]
        # t3 = (r0s3 − r1s2 + r2s1 + r3s0)
        self.res[3] = self.x[0] * y[3] - self.x[1] * y[2] + self.x[2] * y[1] + self.x[3] * y[0]

        return self

    @property
    def inverse(self):
        for i in range(1, 4):
            self.x[i] = -1 * self.x[i]
        return self

    def rotate(self, q, mode='active'):
        assert mode in ['active', 'passive'], 'mode must be \'active\' or \'passive\' rotation'
        if mode == 'active':
            q.inverse.prod(self.x).prod(q.x)
        else:
            assert isinstance(q.inverse, object)
            q.prod(self.x).prod(q.inverse)
        return self
