import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io
import scipy.stats

from ex4_imm import dynamicmodels, measurementmodels, ekf
from ex4_imm.estimationstatistics import mahalanobis_distance_squared
from exc5 import pda


# %% plot config check and style setup

# to see your plot config
print(f"matplotlib backend: {matplotlib.get_backend()}")
print(f"matplotlib config file: {matplotlib.matplotlib_fname()}")
print(f"matplotlib config dir: {matplotlib.get_configdir()}")
plt.close("all")

# set styles
try:
    # installed with "pip install SciencePLots" (https://github.com/garrettj403/SciencePlots.git)
    # gives quite nice plots
    plt_styles = ["science", "grid", "bright", "no-latex"]
    plt.style.use(plt_styles)
    print(f"pyplot using style set {plt_styles}")
except Exception as e:
    print(e)
    print("setting grid and only grid and legend manually")
    plt.rcParams.update(
        {
            # setgrid
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.color": "k",
            "grid.alpha": 0.5,
            "grid.linewidth": 0.5,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": True,
            "legend.numpoints": 1,
        }
    )

# %%
use_pregen = True
data_file_name = "data_for_pda.mat"
if use_pregen:
    loaded_data = scipy.io.loadmat(data_file_name)
    K = loaded_data["K"].item()
    Ts = loaded_data["Ts"].item()
    Xgt = loaded_data["Xgt"].T
    Z = [zk.T for zk in loaded_data["Z"].ravel()]
    true_association = loaded_data["a"].ravel()
else:
    x0 = np.array([0, 0, 1, 1, 0])
    P0 = np.diag([50, 50, 10, 10, np.pi / 4]) ** 2
    # model parameters
    sigma_a_true = 0.25
    sigma_omega_true = np.pi / 15
    sigma_z = 3
    # sampling interval a length
    K = 1000
    Ts = 0.1
    # detection and false alarm
    PDtrue = 0.9
    lambdatrue = 3e-4
    np.rando.rng(10)
    # [Xgt, Z, a] = sampleCTtrack(K, Ts, x0, P0, qtrue, rtrue,PDtrue, lambdatrue);
    raise NotImplementedError
# %%

# plot measurements close to the trajectory
fig1, ax1 = plt.subplots(num=1, clear=True)

Z_plot_data = np.empty((0, 2), dtype=float)
plot_measurement_distance = 45
for Zk, xgtk in zip(Z, Xgt):
    to_plot = np.linalg.norm(Zk - xgtk[None:2], axis=1) <= plot_measurement_distance
    Z_plot_data = np.append(Z_plot_data, Zk[to_plot], axis=0)

ax1.scatter(*Z_plot_data.T, color="C1")
ax1.plot(*Xgt.T[:2], color="C0", linewidth=1.5)
ax1.set_title("True trajectory and the nearby measurements")

# Model parameters EKF
sigma_a = 6  # From EKF exc 3
sigma_z = 4  # From EKF exc 3

# PDA relevant
PD = 0.6
clutter_intensity = 10e-10
gate_size = 2

dynamic_model = dynamicmodels.WhitenoiseAccelleration(sigma_a)
measurement_model = measurementmodels.CartesianPosition(sigma_z)
ekf_filter = ekf.EKF(dynamic_model, measurement_model)


tracker = pda.PDA(ekf_filter, clutter_intensity, PD, gate_size)

# allocate
NEES = np.zeros(K)
NEESpos = np.zeros(K)
NEESvel = np.zeros(K)

# initialize
x_bar_init = np.array([*Z[0][true_association[0] - 1], 0, 0])

P_bar_init = np.zeros((4, 4))
P_bar_init[[0, 1], [0, 1]] = 2 * sigma_z ** 2
P_bar_init[[2, 3], [2, 3]] = 10 ** 2

init_state = tracker.init_filter_state({"mean": x_bar_init, "cov": P_bar_init})

tracker_update = init_state
tracker_update_list = []
tracker_predict_list = []
# estimate
for k, (Zk, x_true_k) in enumerate(zip(Z, Xgt)):
    tracker_predict = tracker.predict(tracker_update, Ts=Ts)
    tracker_update = tracker.update(Zk, tracker_predict)
    x_est, P_est = tracker_update.mean, tracker_predict.cov
    NEES[k] = tracker.state_filter.NEES_from_gt(x_est, x_true_k[:4], P_est)  #  mahalanobis_distance_squared(x_true_k, x_bar, P_bar)
    NEESpos[k] = tracker.state_filter.NEES_from_gt(x_est[:2], x_true_k[:2], P_est[:2, :2])
    NEESvel[k] = tracker.state_filter.NEES_from_gt(x_est[2:4], x_true_k[2:4], P_est[2:, 2:])

    tracker_predict_list.append(tracker_predict)
    tracker_update_list.append(tracker_update)

