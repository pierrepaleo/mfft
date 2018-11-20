import numpy as np

# gpyFFT (clfft/openCL)
try:
    from gpyfft.fft import FFT as cl_fft
    __have_gpyfft__ = True
except ImportError:
    __have_gpyfft__ = False
# skcuda (CUDA)
try:
    import skcuda.fft as cu_fft
    __have_cufft__ = True
except ImportError:
    __have_cufft__ = False
# pyfftw (FFTW3)
try:
    import pyfftw
    __have_fftw__ = True
except ImportError:
    __have_fftw__ = False



class FFT(object):
    def __init__(self, shape=None, dtype=None, data=None, shape_out=None, double_precision=False):
        """
        Initialize a FFT plan.

        Parameters
        ----------
        shape: tuple
            Shape of the input data.
        dtype: type
            Data type of the input data.
        data: numpy.ndarray, optional
            Input data. If provided, the arguments "shape" and "dtype" are ignored,
            and are instead infered from "data".
        shape_out: tuple, optional
            Shape of output data. By default, the data has the same shape as the input
            data (in case of C2C transform), or a shape with the last dimension halved
            (in case of R2C transform). If shape_out is provided, it must be greater
            or equal than the shape of input data. In this case, FFT is performed
            with zero-padding.
        double_precision: bool, optional
            If set to True, computations will be done in double precision regardless
            of the input data type.
        """
        if shape is None and dtype is None and data is None:
            raise ValueError("Please provide either (shape and dtype) or data")
        if data is not None:
            self.shape = data.shape
            self.dtype = data.dtype
        else:
            self.shape = shape
            self.dtype = dtype
        self.user_data = data
        self.double_precision = double_precision
        self.set_dtypes()
        self.calc_shape()

    def set_dtypes(self):
        dtypes_mapping = {
            np.dtype("float32"): np.complex64,
            np.dtype("float64"): np.complex128,
            np.dtype("complex64"): np.complex64,
            np.dtype("complex128"): np.complex128
        }
        dp = {
            np.dtype("float32"): np.float64,
            np.dtype("complex64"): np.complex128
        }
        self.dtype_in = np.dtype(self.dtype)
        if self.double_precision and self.dtype in dp:
            self.dtype_in = dp[self.dtype_in]
        self.dtype_out = dtypes_mapping[self.dtype_in]


    def calc_shape(self):
        # TODO allow for C2C even for real input data (?)
        if self.dtype_in in [np.float32, np.float64]:
            last_dim = self.shape[-1]//2 + 1
            # FFTW convention
            self.shape_out = self.shape[:-1] + (self.shape[-1]//2 + 1,)
        else:
            self.shape_out = self.shape

