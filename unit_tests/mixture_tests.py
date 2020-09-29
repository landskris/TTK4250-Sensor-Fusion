import unittest

import numpy as np


class TestCVmodel(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.ts = 0.5
        cls.x = np.array([0, 0, 1, 1])
        cls.P_init = np.eye(len(cls.x))  # Identity matrix
        # cls.wna = WhitenoiseAccelleration(sigma=0.5)
        # cls.meas_models = CartesianPosition(sigma=0.5)
        # cls.ekf = EKF(DynamicModel(), MeasurementModel())
        cls.N = 3
        cls.n = 3
        cls.random_weights = np.ones(cls.n) / cls.n
        cls.means = np.random.randint(0, 2, size=(cls.N, cls.n))
        cls.covs = np.reshape(np.eye(cls.n), newshape=(cls.N, cls.n, cls.n)

    def test_gaussian_reduction(self):
        f_dot_xk_last = self.wna.f(self.x, self.ts)
        self.assertEqual(f_dot_xk_last.shape, (4,))  # [p_01, p_11, u_01, u_11]

    def test_wna_q(self):
        pass

    def test_meas_model_H(self):
        pass


if __name__ == "__main__":
    unittest.main()