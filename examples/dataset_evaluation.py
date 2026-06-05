# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Evaluating RIANN on a recorded dataset
#
# This notebook loads IMU data from an HDF5 file (e.g. the
# [BROAD dataset](https://github.com/dlaidig/broad)), estimates the attitude
# with `RIANN`, and — if reference orientations are available — plots the
# attitude error. If the data file is not present, it falls back to dummy data
# so the notebook still runs end-to-end.

# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np

from riann import RIANN

# %% [markdown]
# ## Load the data
#
# Adapt the path and field names to your own dataset.

# %%
try:
    with h5py.File("data_hdf5/01_undisturbed_slow_rotation_A.hdf5", "r") as f:
        acc = f["imu_acc"][:]  # Accelerometer data [m/s²]
        gyr = f["imu_gyr"][:]  # Gyroscope data [rad/s]
        ref_quat = f["opt_quat"][:]  # Reference quaternions (w, x, y, z)
        fs = float(f.attrs["sampling_rate"])  # Sampling rate [Hz]
except (FileNotFoundError, OSError):
    # Dummy data so the example runs without the dataset present.
    sequence_length = 1000
    acc = np.ones((sequence_length, 3))
    gyr = np.zeros((sequence_length, 3))
    ref_quat = None
    fs = 200.0

# %% [markdown]
# ## Estimate the attitude

# %%
riann = RIANN()
est_quat = riann.predict(acc, gyr, fs)
print(f"Estimated {est_quat.shape[0]} quaternions at {fs:g} Hz")

# %% [markdown]
# ## Compare against the reference (if available)

# %%
def attitude_error_deg(q1, q2):
    """Attitude error in degrees between two quaternion sequences."""
    dots = np.clip(np.sum(q1 * q2, axis=1), -1.0, 1.0)
    return np.degrees(2 * np.arccos(np.abs(dots)))


if ref_quat is not None:
    errors = attitude_error_deg(est_quat, ref_quat)

    plt.figure(figsize=(10, 4))
    plt.plot(errors)
    plt.xlabel("Sample")
    plt.ylabel("Attitude error (deg)")
    plt.title("Orientation estimation error")
    plt.grid(True)
    plt.show()

    print(f"Mean error: {errors.mean():.2f}°, max error: {errors.max():.2f}°")
else:
    print("No reference data available; skipping the error plot.")
