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

class BaseFFT(object):
    def __init__(self, **kwargs):
        """
        Base class for all FFT backends.
        """
        self.get_args(**kwargs)

        if self.shape is None and self.dtype is None and self.data is None:
            raise ValueError("Please provide either (shape and dtype) or data")
        if self.data is not None:
            self.shape = self.data.shape
            self.dtype = self.data.dtype
        self.user_data = self.data
        self.data_in_allocated = False
        self.data_out_allocated = False
        self.set_dtypes()
        self.calc_shape()


    def get_args(self, **kwargs):
        expected_args = {
            "shape": None,
            "dtype": None,
            "data": None,
            "double_precision": False,
            "shape_out": None,
            "axes": (-1,),
            "normalize": "rescale",
        }
        for arg_name, default_val in expected_args.items():
            if arg_name not in kwargs:
                # Base class was not instantiated properly
                raise ValueError("Please provide argument %s" % arg_name)
            setattr(self, arg_name, default_val)
        for arg_name, arg_val in kwargs.items():
            setattr(self, arg_name, arg_val)

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
        if self.double_precision:
            if self.dtype_in in dp:
                self.dtype_in = np.dtype(dp[self.dtype_in])
            else:
                raise ValueError(
                    "Invalid input data type for double precision: got %s" %
                    self.dtype_in
                )
        if self.dtype_in not in dtypes_mapping:
            raise ValueError("Invalid input data type: got %s" %
                self.dtype_in
            )
        self.dtype_out = dtypes_mapping[self.dtype_in]


    def calc_shape(self):
        # TODO allow for C2C even for real input data (?)
        if self.dtype_in in [np.float32, np.float64]:
            last_dim = self.shape[-1]//2 + 1
            # FFTW convention
            self.shape_out = self.shape[:-1] + (self.shape[-1]//2 + 1,)
        else:
            self.shape_out = self.shape
