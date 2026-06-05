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
# # RIANN quickstart
#
# This notebook demonstrates the three inference modes of `RIANN`:
#
# 1. **Batch** — `predict` processes a whole sequence at once.
# 2. **Real-time** — `predict_step` consumes one measurement at a time,
#    keeping the hidden state between calls.
# 3. **Step-by-step with state tracking** — `predict_sequence` runs
#    `predict_step` internally and also returns the hidden states.

# %%
import numpy as np

from riann import RIANN

# %% [markdown]
# ## Example 1 — batch processing

# %%
sequence_length = 100
acc = np.ones((sequence_length, 3))  # Accelerometer data [m/s²]
gyr = np.zeros((sequence_length, 3))  # Gyroscope data [rad/s]
fs = 200  # Sampling rate [Hz]

riann = RIANN()
quaternions = riann.predict(acc, gyr, fs)
print(f"Quaternions shape: {quaternions.shape}")  # (100, 4)

# %% [markdown]
# ## Example 2 — real-time, step-by-step processing
#
# Feed one noisy measurement at a time; the hidden state carries between calls.

# %%
riann = RIANN()
riann.set_sampling_rate(100)  # 100 Hz
riann.reset_state()


def simulate_imu_reading():
    """Simulate a single noisy IMU reading for a sensor at rest."""
    acc = np.array([0.0, 0.0, 9.81]) + np.random.normal(0, 0.1, 3)
    gyr = np.array([0.0, 0.0, 0.0]) + np.random.normal(0, 0.01, 3)
    return acc, gyr


orientations = []
for _ in range(100):
    acc_step, gyr_step = simulate_imu_reading()
    orientations.append(riann.predict_step(acc_step, gyr_step))

orientations = np.array(orientations)
print(f"Collected {len(orientations)} orientation estimates")

# %% [markdown]
# ## Example 3 — step-by-step with hidden-state tracking
#
# Generate a synthetic rotation around the y-axis and estimate the attitude
# while also collecting the hidden states.

# %%
def generate_synthetic_data(fs=100, duration=5):
    """Generate synthetic IMU data for a rotation around the y-axis."""
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples)
    acc = np.zeros((n_samples, 3))
    gyr = np.zeros((n_samples, 3))
    for i, time_s in enumerate(t):
        if 1.0 <= time_s <= 3.0:
            gyr[i, 1] = 0.5  # 0.5 rad/s around the y-axis
        if time_s < 1.0:
            acc[i] = [0, 0, 9.81]
        elif time_s < 3.0:
            angle = 0.5 * (time_s - 1.0)
            acc[i] = [9.81 * np.sin(angle), 0, 9.81 * np.cos(angle)]
        else:
            acc[i] = [9.81 * np.sin(1.0), 0, 9.81 * np.cos(1.0)]
    acc += np.random.normal(0, 0.1, acc.shape)
    gyr += np.random.normal(0, 0.01, gyr.shape)
    return t, acc, gyr


t, acc, gyr = generate_synthetic_data(fs=100)

riann = RIANN()
quaternions, states = riann.predict_sequence(acc, gyr, fs=100)
print(f"quaternions: {quaternions.shape}, hidden states: {states.shape}")
