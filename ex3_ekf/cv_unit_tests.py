import os
import unittest

import numpy as np
import scipy

from ex3_ekf.dynamicmodels import WhitenoiseAccelleration, DynamicModel
from ex3_ekf.ekf import EKF
from ex3_ekf.gaussparams import GaussParams
from ex3_ekf.measurmentmodels import CartesianPosition, MeasurementModel


class DataHandler(object):
    def __init__(self, K: int, Ts: float, Xgt: np.ndarray, z: np.ndarray):
        self.K = K
        self.Ts = Ts
        self.xgt = Xgt
        self.z = z

class TestCVmodel(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.ts = 0.5
        cls.x = np.array([0, 0, 1, 1])
        cls.P_init = np.eye(len(cls.x))  # Identity matrix
        cls.wna = WhitenoiseAccelleration(sigma=0.5)
        cls.meas_models = CartesianPosition(sigma=0.5)
        cls.ekf = EKF(DynamicModel(), MeasurementModel())

        def set_up_ek_pretester():
            data_path = os.path.join('ex3_ekf','data_for_ekf.mat')
            loadData: dict = scipy.io.loadmat(data_path)
            K: int = int(loadData['K'])  # The number of time steps
            Ts: float = float(loadData['Ts'])  # The sampling time
            Xgt: np.ndarray = loadData['Xgt'].T  # ground truth
            Z: np.ndarray = loadData['Z'].T  # the measurements
            return DataHandler(K, Ts, Xgt, Z)

        cls.pre_test_data_handler = set_up_ek_pretester()

    def test_wna_f(self):
        f_dot_xk_last = self.wna.f(self.x, self.ts)
        self.assertEqual(f_dot_xk_last.shape, (4,))  # [p_01, p_11, u_01, u_11]

    def test_wna_q(self):
        f_dot_xk_last = self.wna.Q(self.x, self.ts)
        self.assertEqual(f_dot_xk_last.shape, (4,4))  # [p_01, p_11, u_01, u_11]

    def test_meas_model_H(self):
        H = self.meas_models.H(self.x)

    def test_ekf_log_likelihood_shapes(self):
        ekfstate = GaussParams(self.x, self.P_init)



if __name__ == "__main__":
    unittest.main()