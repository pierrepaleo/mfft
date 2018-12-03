# mFFT

## About 

mFFT (Multi-backend FFT) is a wrapper around various FFT solutions, offering a unified interface.

## Features

* Backends: numpy, FFTW, CUDA, OpenCL
* Unified interface - mFFT takes care of all the boring work
* 1D-ND transforms, R2C, C2C
* Batched transforms

## Examples

### Simple FFT with numpy
```python
import numpy as np
from scipy.misc import ascent
from mfft.fft import FFT
img = ascent().astype(np.float32)

F = FFT(data=img, backend="numpy") # automatically chooses R2C transform
img_f = F.fft(img)
```

### Using FFTW
```python
F = FFT(data=img, backend="fftw", num_threads=4) 
img_f = F.fft(img)
# do some operation of img_f ...
F.ifft(img_f)
```

### Using OpenCL
```python
F = FFT(data=img, backend="opencl")
# All the Host <-> Device copies are handled under the hood
img_f = F.fft(img) # by default, result is a numpy array
# Input and/or output can be device array as well
d_in = parray.to_device(F.queue, img)
d_out = parray.zeros(F.queue, F.shape_out, dtype=F.dtype_out)
F.fft(d_in, output=d_out)
```


### Using CUDA
```python
import pycuda.autoinit
import pycuda.gpuarray as gpuarray 
F = FFT(data=img, backend="cuda")
d_in = gpuarray.to_gpu(img)
d_out = gpuarray.zeros(F.shape_out, F.dtype_out)
F.fft(d_in, output=d_out) # CUFFT is twice faster than clfft for R2C transforms
```
