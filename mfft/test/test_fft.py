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
"""Test of the MFFT module"""

import numpy as np
import unittest
from scipy.misc import ascent
#~ from silx.opencl import ocl

from mfft.fft import FFT




class TransformInfos(object):
    def __init__(self):
        self.dimensions = [
            "1D",
            "batched_1D",
            "2D",
            #~ "batched_2D",
            #~ "3D",
        ]
        self.modes = {
            "R2C": np.float32,
            "R2C_double": np.float64,
            "C2C": np.complex64,
            "C2C_double": np.complex128,
        }
        self.sizes = {
            "1D": [(512,), (511,)],
            "2D": [(512, 512), (512, 511), (511, 512), (511, 511)],
            "3D": [(128, 128, 128), (128, 128, 127), (128, 127, 128), (127, 128, 128),
                 (128, 127, 127), (127, 128, 127), (127, 127, 128), (127, 127, 127)]
        }
        self.backends = [
            #~ "numpy",
            #~ "fftw",
            "opencl",
            #~ "cuda"
        ]
        self.axes = {
            "1D": None,
            "batched_1D": (1,),
            "2D": None,
            "batched_2D": (2, 1),
            "3D": None,
        }
        self.sizes["batched_1D"] = self.sizes["2D"]
        self.sizes["batched_2D"] = self.sizes["3D"]


class TestFFT(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestFFT, cls).setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    #~ def __init__(self, methodName='runTest', name=None, params=None):
        #~ unittest.TestCase.__init__(self, methodName)
        #~ self.name = name
        #~ self.params = params

    def setUp(self):
        self.tol = {
            "float32": 1e-4,
            "float64": 1e-9
        }
        self.data = ascent().astype("float32")
        self.data1d = self.data[:, 0] # non-contiguous data
        self.data3d = np.tile(self.data[:128, :128], (128, 1, 1))
        self.transform_infos = TransformInfos()
        self.data_refs = {
            1: self.data1d,
            2: self.data,
            3: self.data3d,
        }


    def tearDown(self):
        self.data = None

    def calc_mae(self, arr1, arr2):
        """
        Compute the Max Absolute Error between two arrays
        """
        return np.max(np.abs(arr1 - arr2))


    def test_fft(self, backend, trdim, mode, size):
        """
        Test FFT with a given configuration.

        Parameters
        ----------
        backend: str
            FFT backend. Can be numpy, opencl, cuda, fftw.
        trdim: str
            transform dimensions. Can be 1D, 2D, 3D, batched_1D, batched_2D
        mode: str
            transform mode. Can be R2C, C2R, R2C_double, C2C_double
        size: tuple
            transform input data shape.
        """
        ndim = len(size)
        input_data = self.data_refs[ndim].astype(self.transform_infos.modes[mode])

        F = FFT(data=input_data, axes=self.transform_infos.axes[trdim], backend=backend)
        self.assertTrue(F is not None)






    def test_dummy(self):
        for backend in self.transform_infos.backends:
            print("Testing backend: %s" % backend)
            for trdim in self.transform_infos.dimensions:
                print("   testing %s" % trdim)
                for mode in self.transform_infos.modes:
                    print("   testing %s:%s" % (trdim, mode))
                    for size in self.transform_infos.sizes[trdim]:
                        print("      size: %s" % str(size))
                        self.test_fft(backend, trdim, mode, size)













    def test_plan_creation(self):
        #~ plan_numpy = FFT(data=self.data[:, 0], backend="numpy")
        plan_fftw = FFT(data=self.data[:, 0], backend="fftw", check_alignment=True)
        plan_opencl = FFT(data=self.data[:, 0], backend="opencl")
        #~ plan_cuda = FFT(data=self.data[:, 0], backend="cuda")


    def test_forward_FFT(self):
        data1d = self.data[:, 0]
        F = FFT(data=self.data1d, backend="fftw", check_alignment=True)
        res = F.fft(self.data1d)

    def test_fft_modes(self):
        data1d = self.data1d
        N = data1d.size
        res_np = np.fft.rfft(self.data1d)

        # rescale
        F = FFT(data=data1d, backend="fftw", check_alignment=True, normalize="rescale")
        res = F.fft(data1d)
        self.assertLess(self.calc_mae(res, res_np), self.tol["float32"] * data1d.max())
        res2 = F.ifft(res)
        self.assertLess(self.calc_mae(res2, data1d), self.tol["float32"])
        # ortho
        F = FFT(data=data1d, backend="fftw", check_alignment=True, normalize="ortho")
        res = F.fft(data1d)
        self.assertLess(self.calc_mae(res, res_np/np.sqrt(N)), self.tol["float32"] * data1d.max())
        res2 = F.ifft(res)
        self.assertLess(self.calc_mae(res2, data1d), self.tol["float32"])
        # none
        F = FFT(data=data1d, backend="fftw", check_alignment=True, normalize="none")
        res = F.fft(data1d)
        mae = np.max(np.abs(res - res_np))
        self.assertLess(mae, self.tol["float32"] * data1d.max())
        res2 = F.ifft(res)
        self.assertLess(self.calc_mae(res2, data1d * N), self.tol["float32"] * data1d.max())


    def test_device_input(self):
        """
        Test FFT where input is on device (OpenCL).
        """
        F = FFT(data=self.data, backend="opencl")
        # input: host, output: host






def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestFFT("test_plan_creation"))
    testSuite.addTest(TestFFT("test_fft_modes"))
    testSuite.addTest(TestFFT("test_dummy"))

    #~ testSuite.addTest(TestFFT("test_forward_FFT"))
    #~ for test_name, test_params in test_cases.items():
        #~ testSuite.addTest(parameterize(TestFFT, name=test_name, params=test_params))
    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")



"""
Test plan
----------

- "Odd" sized transform
- 1D, batched 1D, 2D, batched 2D, 3D
- with/without double precision
- Forward and inverse

Cuda/OpenCL:
  - host input, host output
  - device input, host output
  - host input, device output
  - device input, device output

for backend in backends:
    for trdim in ["1d", "batched_1d", "2d", ...]:
        for mode in ["R2C", "C2C", R2C_double", "C2C_double"]:
            for size in sizes[trdim]: # odd sizes, pow 2, ...
                test_forward()
                test_inverse()


1d:     [(512,), (511,)]
b1d/2d: [(512, 512), (512, 511), (511, 512), (511, 511)]
b2d/3d: [(128, 128, 128), (128, 128, 127), (128, 127, 128), (127, 128, 128),
         (128, 127, 127), (127, 128, 127), (127, 127, 128), (127, 127, 127)]






"""
