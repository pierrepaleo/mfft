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
try:
    import pyopencl as cl
    import pyopencl.array as parray
except ImportError:
    __have_pyopencl__ = False
try:
    from gpyfft.fft import FFT as cl_fft
    __have_clfft__ = True
except ImportError:
    __have_clfft__ = False

class CLFFT(BaseFFT):
    def __init__(
        self,
        shape=None,
        dtype=None,
        data=None,
        shape_out=None,
        double_precision=False,
        axes=None,
        normalize="rescale",
        ctx=None,
        fast_math=False,
    ):
        """
        Initialize a FFTW plan.
        Please see FFT class for parameters help.

        CLFFT-specific parameters:
        ctx: pyopencl.Context
            If set to other than None, an existing pyopencl context is used.
        fast_math: bool
            If set to True, computations will be done with "fast math" mode,
            i.e more speed but less accuracy.
        """
        if not(__have_clfft__) or not(__have_pyopencl__):
            raise ImportError("Please install pyopencl and gpyfft to use the OpenCL back-end")

        super().__init__(
            shape=shape,
            dtype=dtype,
            data=data,
            shape_out=shape_out,
            double_precision=double_precision,
            axes=axes,
            normalize=normalize,
        )
        self.ctx = ctx
        self.fast_math = fast_math
        self.init_context_queue()
        self.allocate_arrays()
        self.compute_forward_plan()
        self.compute_inverse_plan()

    def _allocate(self, shape, dtype):
        return parray.zeros(self.queue, shape, dtype=dtype)



    def check_array(self, array, shape, dtype, copy=True):
        if array.shape != shape:
            raise ValueError("Invalid data shape: expected %s, got %s" %
                (shape, array.shape)
            )
        if array.dtype != dtype:
            raise ValueError("Invalid data type: expected %s, got %s" %
                (dtype, array.dtype)
            )
        if isinstance(array, np.ndarray):
            # numpy stuff

            if not(arr.flags["C_CONTIGUOUS"] and arr.dtype == dtype):
                array2 = np.ascontiguousarray(array, dtype=dtype)
            else:
                array2 = array






        #~ if not(arr.flags["C_CONTIGUOUS"] and arr.dtype == dtype):
            #~ return np.ascontiguousarray(arr, dtype=dtype)
        #~ else:
            #~ return arr
        elif isinstance(array, parray.Array):
            # parray styff
        else:
            raise ValueError(
                "Invalid array type %s, expected numpy.ndarray or pyopencl.array" %
                type(array)
            )



    def init_context_queue(self):
        if self.ctx is None:
            self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(ctx)

    def compute_forward_plan(self):
        self.plan_forward = cl_fft(
            self.ctx,
            self.queue,
            self.data_in,
            self.data_out,
            axes=self.axes
        )

    def compute_inverse_plan(self):
        self.plan_forward = cl_fft(
            self.ctx,
            self.queue,
            self.data_out,
            self.data_in,
            axes=self.axes
        )


    def fft(self, array, output=None, async=False):
        """
        Perform a
        (forward) Fast Fourier Transform.

        Parameters
        ----------
        array: numpy.ndarray or pyopencl.array
            Input data. Must be consistent with the current context.
        output: numpy.ndarray or pyopencl.array, optional
            Output data. By default, output is a numpy.ndarray.
        async: bool, optional
            Whether to perform operation in asynchronous mode. Default is False,
            meaning that we wait for transform to complete.
        """
        data_in = self.set_input_data(data=array, copy=True)
        data_out = self.set_output_data(data=output, copy=False)
        event, = self.plan_forward.enqueue()
        if not(async):
            event.wait()

        #~ assert id(self.plan_forward.output_array) == id(self.data_out) == id(data_out) # DEBUG









