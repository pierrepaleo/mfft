#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/
import numpy as np

from .basefft import BaseFFT


class NPFFT(BaseFFT):
    def __init__(
        self,
        shape=None,
        dtype=None,
        data=None,
        shape_out=None,
        double_precision=False,
        axes=None,
        normalize="rescale",
    ):
        """
        Initialize a FFTW plan.
        Please see FFT class for parameters help.
        """

        super().__init__(
            shape=shape,
            dtype=dtype,
            data=data,
            shape_out=shape_out,
            double_precision=double_precision,
            axes=axes,
            normalize=normalize,
        )
        self.backend = "numpy"
        self.real_transform = np.isrealobj(self.data_in)
        self.set_fft_functions()
        #~ self.allocate_arrays() # not needed for this backend
        self.compute_plans()


    def set_fft_functions(self):
        # (fwd, inv) = _fft_functions[is_real][ndim]
        self._fft_functions = {
            True: {
                1: (np.fft.rfft, np.fft.irfft),
                2: np.fft.rfft2, np.fft.irfft2),
                3: (np.fft.rfftn, np.fft.irfftn),
            }
            False: {
                1: (np.fft.fft, np.fft.ifft),
                2: (np.fft.fft2, np.fft.ifft2),
                3: (np.fft.fftn, np.fft.ifftn),
            }
        }


    def _allocate(self, shape, dtype):
        return np.zeros(self.queue, shape, dtype=dtype)


    def check_array(self, array, shape, dtype, copy=True):
        if array.shape != shape:
            raise ValueError("Invalid data shape: expected %s, got %s" %
                (shape, array.shape)
            )
        if array.dtype != dtype:
            raise ValueError("Invalid data type: expected %s, got %s" %
                (dtype, array.dtype)
            )


    def set_data(self, dst, src, shape, dtype, copy=True, name=None):
        """
        dst is a device array owned by the current instance
        (either self.data_in or self.data_out).

        copy is ignored for device<-> arrays.
        """
        self.check_array(src, shape, dtype)
        if isinstance(src, np.ndarray):
            if not(src.flags["C_CONTIGUOUS"]):
                src = np.ascontiguousarray(src, dtype=dtype)
            # working on underlying buffer is notably faster
            #~ dst[:] = src[:]
            evt = cl.enqueue_copy(self.queue, dst.data, src)
            evt.wait()
        elif isinstance(src, parray.Array):
            # No copy, use the data as self.d_input or self.d_output
            # (this prevents the use of in-place transforms, however).
            # We have to keep their old references.
            if name is None:
                # This should not happen
                raise ValueError("Please provide either copy=True or name != None")
            assert id(self.refs[name]) == id(dst) # DEBUG
            setattr(self, name, src)
        else:
            raise ValueError(
                "Invalid array type %s, expected numpy.ndarray or pyopencl.array" %
                type(src)
            )
        return dst


    def compute_plans(self):
        ndim = len(self.shape)
        forward_func = self._fft_function[self.real_transform][np.minimum(ndim, 3)]
        self.forward_func = forward_func
        # Batched transform
        if (self.user_axes is not None) and len(self.user_axes) < ndim:
            forward_func = self._fft_function[self.real_transform][np.minimum(ndim-1, 3)]
            kwargs = {"axes": self.user_axes}
            if ndim == 2:
                assert len(self.user_axes) == 0
                kwargs = {"axis": self.user_axes[0]}
            self.forward_func = lambda x : forward_func(x, **kwargs)





    def fft(self, array):
        """
        Perform a
        (forward) Fast Fourier Transform.

        Parameters
        ----------
        array: numpy.ndarray or pyopencl.array
            Input data. Must be consistent with the current context.

        """


    def ifft(self, array):
        """
        Perform a
        (inverse) Fast Fourier Transform.

        Parameters
        ----------
        array: numpy.ndarray or pyopencl.array
            Input data. Must be consistent with the current context.
        """



