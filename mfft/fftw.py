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
    import pyfftw
    __have_fftw__ = True
except ImportError:
    __have_fftw__ = False

# TODO support in-place ? In this case, pyfftw.builders cannot be used
class FFTW(BaseFFT):
    """
    TODO docstring
    """
    def __init__(
        self,
        shape=None,
        dtype=None,
        data=None,
        shape_out=None,
        double_precision=False,
        axes=(-1,),
        normalize="rescale",
        check_alignment=False,
        num_threads=1,
    ):
        """
        Initialize a FFTW plan.
        Please see FFT class for parameters help.

        FFTW-specific parameters:
        check_alignment: bool
            If set to True and "data" is provided, this will enforce the input data
            to be "byte aligned", which might imply extra memory usage.
        num_threads: int
            Number of threads for computing FFT.
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
        self.axes = axes
        self.normalize = normalize
        self.check_alignment = check_alignment
        self.num_threads = num_threads

        self.set_input_data()
        self.set_output_data()
        self.set_fftw_flags()
        self.compute_forward_plan()
        self.compute_inverse_plan()

    def set_fftw_flags(self):
        self.fftw_flags = ('FFTW_MEASURE', ) # TODO
        self.fftw_planning_timelimit = None # TODO
        self.fftw_norm_modes = {
            "rescale": {"ortho": False, "normalize": True},
            "ortho": {"ortho": True, "normalize": False},
            "none": {"ortho": False, "normalize": False},
        }
        if self.normalize not in self.fftw_norm_modes:
            raise ValueError("Unknown normalization mode %s. Possible values are %s" %
                (self.normalize, self.fftw_norm_modes.keys())
            )
        self.fftw_norm_mode = self.fftw_norm_modes[self.normalize]

    def check_array(self, array, dtype, copy=True):
        """
        Check that a given array is compatible with the FFTW plans,
        in terms of alignment and data type.
        If the provided array does not meet any of the checks, a new array
        is returned.
        """
        if array.dtype != dtype:
            raise ValueError("Invalid data type: expected %s, got %s" %
                (dtype, array.dtype)
            )
        if self.check_alignment and not(pyfftw.is_byte_aligned(array)):
            array2 = pyfftw.zeros_aligned(self.shape, dtype=self.dtype_in)
            np.copyto(array2, array)
        else:
            if copy:
                array2 = np.copy(array)
            else:
                array2 = array
        return array2

    def set_input_data(self, data=None, copy=True):
        if data is not None:
            self.data_in = self.check_array(data, self.dtype_in, copy=copy)
        else:
            if not(self.data_in_allocated):
                self.data_in = pyfftw.zeros_aligned(self.shape, dtype=self.dtype_in)
                self.data_in_allocated = True
        return self.data_in

    # TODO padding (or in BaseFFT.calc_shape)
    def set_output_data(self, data=None, copy=True):
        if data is not None:
            self.data_out = self.check_array(data, self.dtype_out, copy=copy)
        else:
            if not(self.data_out_allocated):
                self.data_out = pyfftw.zeros_aligned(self.shape_out, dtype=self.dtype_out)
                self.data_out_allocated = True
        return self.data_out

    def compute_forward_plan(self):
        self.plan_forward = pyfftw.FFTW(
            self.data_in,
            self.data_out,
            axes=self.axes,
            direction='FFTW_FORWARD',
            flags=self.fftw_flags,
            threads=self.num_threads,
            planning_timelimit=self.fftw_planning_timelimit,
            # the following seems to be taken into account only when using __call__
            ortho=self.fftw_norm_mode["ortho"],
        )

    def compute_inverse_plan(self):
        self.plan_inverse = pyfftw.FFTW(
            self.data_out,
            self.data_in,
            axes=self.axes,
            direction='FFTW_BACKWARD',
            flags=self.fftw_flags,
            threads=self.num_threads,
            planning_timelimit=self.fftw_planning_timelimit,
            # the following seem to be taken into account only when using __call__
            ortho=self.fftw_norm_mode["ortho"],
            normalise_idft=self.fftw_norm_mode["normalize"],
        )

    def fft(self, array, output=None):
        data_in = self.set_input_data(data=array, copy=True)
        data_out = self.set_output_data(data=output, copy=False)
        # execute.__call__ does both update_arrays() and normalization
        self.plan_forward(
            input_array=data_in,
            output_array=data_out,
            ortho=self.fftw_norm_mode["ortho"],
        )
        assert id(self.plan_forward.output_array) == id(self.data_out) == id(data_out) # DEBUG
        return data_out

    def ifft(self, array, output=None):
        data_in = self.set_output_data(data=array, copy=True)
        data_out = self.set_input_data(data=output, copy=False)
        # execute.__call__ does both update_arrays() and normalization
        self.plan_inverse(
            input_array=data_in,
            output_array=data_out,
            ortho=self.fftw_norm_mode["ortho"],
            normalise_idft=self.fftw_norm_mode["normalize"]
        )
        assert id(self.plan_inverse.output_array) == id(self.data_in) == id(data_out) # DEBUG
        return data_out







