#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic models to be used with eg. EKF.

"""
# %%
import math
from typing import Optional, Sequence
from typing_extensions import Final, Protocol
from dataclasses import dataclass, field

import numpy as np
import numdifftools as nd

# %% the dynamic models interface declaration


class DynamicModel(Protocol):
    n: int
    def f(self, x: np.ndarray, Ts: float) -> np.ndarray: ...
    def F(self, x: np.ndarray, Ts: float) -> np.ndarray: ...
    def Q(self, x: np.ndarray, Ts: float) -> np.ndarray: ...

# %%


@dataclass
class WhitenoiseAccelleration:
    """
    A white noise accelereation model also known as CV, states are position and speed.

    The model includes the discrete prediction equation f, its Jacobian F, and
    the process noise covariance Q as methods.
    """
    # noise standard deviation
    sigma: float
    # number of dimensions
    dim: int = 2
    # number of states
    n: int = 4

    def f(self,
            x: np.ndarray,
            Ts: float,
          ) -> np.ndarray:
        """
        Calculate the zero noise Ts time units transition from x.

        x[:2] is position, x[2:4] is velocity -> x = [p0, p1, u0, u1]
        """
        return self.F(x, Ts)@x

    def F(self,
            x: np.ndarray,
            Ts: float,
          ) -> np.ndarray:
        """ Calculate the transition function jacobian for Ts time units at x.
        F_jac = [f0/dp0  f0/dp1   f0/du0   f0/du1,
                f1/dp0  f1/dp1   f1/du0   f1/du1,
                f2/dp0  f2/dp1   f2/du0   f2/du1,
                f3/dp0  f3/dp1   f3/du0   f3/du1]
        funcs = [ f0: p0 + Ts*u0
                f1: p1 + Ts*u1
                f2: u0
                f3: u1
        ]
            F_jac = [1  0   Ts  0
                    0   1   0  Ts
                    0   0   1   0
                    0   0   0   1]
        """

        return np.array([[1, 0, Ts, 0],
                          [0, 1, 0, Ts],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

    def Q(self,
            x: np.ndarray,
            Ts: float,
          ) -> np.ndarray:
        """
        Calculate the Ts time units transition Covariance.
        """
        # TODO
        # Hint: sigma can be found as self.sigma, see variable declarations
        # Note the @dataclass decorates this class to create an init function that takes
        # sigma as a parameter, among other things.
        q_matr = np.array([[pow(Ts, 3)/3, 0, pow(Ts, 2)/2, 0],
                          [0, pow(Ts, 3)/3, 0, pow(Ts, 2)/2],
                          [pow(Ts, 2)/2, 0, Ts, 0],
                          [0, pow(Ts, 2)/2, 0, Ts]])
        return q_matr*self.sigma**2  # Q*sigma_a^2
