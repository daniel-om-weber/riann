# RIANN (Robust IMU-based Attitude Neural Network) 
> Summary description here.


This file will become your README and also the index of your documentation.

## Install

`pip install .`

## How to use

```python
import numpy as np
##prepare dummy imu signals
sequence_length=100
signal_shape = (sequence_length,3)

acc = np.ones(signal_shape)
gyr = np.zeros(signal_shape)
fs = 200
```

```python
from riann.riann import RIANN
riann = RIANN()
attitude = riann.predict(acc,gyr,fs)
attitude.shape
```




    (100, 4)


