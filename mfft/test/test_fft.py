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
from mfft.fft import FFT

# http://eli.thegreenplace.net/2011/08/02/python-unit-testing-parametrized-test-cases/
class ParametrizedTestCase(unittest.TestCase):
    """ TestCase classes that want to be parametrized should
        inherit from this class.
    """
    def __init__(self, methodName='runTest', param=None):
        super(ParametrizedTestCase, self).__init__(methodName)
        self.param = param

    @staticmethod
    def parametrize(testcase_klass, param=None):
        """ Create a suite containing all tests taken from the given
            subclass, passing them the parameter 'param'.
        """
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(testcase_klass)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(testcase_klass(name, param=param))
        return suite


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
            "batched_1D": (-1,),
            "2D": None,
            "batched_2D": (-2, -1),
            "3D": None,
        }
        self.sizes["batched_1D"] = self.sizes["2D"]
        self.sizes["batched_2D"] = self.sizes["3D"]


class TestData(object):
    def __init__(self):
        self.data = ascent().astype("float32")
        self.data1d = self.data[:, 0] # non-contiguous data
        self.data3d = np.tile(self.data[:128, :128], (128, 1, 1))
        self.data_refs = {
            1: self.data1d,
            2: self.data,
            3: self.data3d,
        }



class TestFFT(ParametrizedTestCase):

    """
    @classmethod
    def setUpClass(cls):
        super(TestFFT, cls).setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
    """

    def setUp(self):
        self.tol = {
            "float32": 1e-4,
            "float64": 1e-9
        }
        self.backend = self.param["backend"]
        self.trdim = self.param["trdim"]
        self.mode = self.param["mode"]
        self.size = self.param["size"]
        self.transform_infos = self.param["transform_infos"]
        self.test_data = self.param["test_data"]

    def tearDown(self):
        pass


    def calc_mae(self, arr1, arr2):
        """
        Compute the Max Absolute Error between two arrays
        """
        return np.max(np.abs(arr1 - arr2))


    def test_fft(self):
        ndim = len(self.size)
        input_data = self.test_data.data_refs[ndim].astype(self.transform_infos.modes[self.mode])

        F = FFT(
            data=input_data,
            axes=self.transform_infos.axes[self.trdim],
            backend=self.backend
        )

        F.fft(input_data)








    '''


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

    '''






class TestNumpyFFT(ParametrizedTestCase):
    """
    Test the Numpy backend individually.
    """


    def setUp(self):
        transforms = {
            "1D": {
                True: (np.fft.rfft, np.fft.irfft),
                False: (np.fft.fft, np.fft.ifft),
            },
            "2D": {
                True: (np.fft.rfft2, np.fft.irfft2),
                False: (np.fft.fft2, np.fft.ifft2),
            },
            "3D": {
                True: (np.fft.rfftn, np.fft.irfftn),
                False: (np.fft.fftn, np.fft.ifftn),
            },
        }
        transforms["batched_1D"] = transforms["1D"]
        transforms["batched_2D"] = transforms["2D"]
        self.transforms = transforms



    def test_numpy_fft(self):
        """
        Test the numpy backend against native fft.
        Results should be exactly the same.
        """
        trinfos = self.param["transform_infos"]
        trdim = self.param["trdim"]
        ndim = len(self.param["size"])
        input_data = self.param["test_data"].data_refs[ndim].astype(trinfos.modes[self.param["mode"]])
        np_fft, np_ifft = self.transforms[trdim][np.isrealobj(input_data)]

        F = FFT(
            data=input_data,
            axes=trinfos.axes[trdim],
            backend="numpy"
        )
        res = F.fft(input_data)
        ref = np_fft(input_data)
        self.assertTrue(np.allclose(res, ref))



def test_numpy_backend(dimensions=None):
    testSuite = unittest.TestSuite()
    transform_infos = TransformInfos()
    test_data = TestData()
    dimensions = dimensions or transform_infos.dimensions

    for trdim in dimensions:
        print("   testing %s" % trdim)
        for mode in transform_infos.modes:
            print("   testing %s:%s" % (trdim, mode))
            for size in transform_infos.sizes[trdim]:
                print("      size: %s" % str(size))
                testcase = ParametrizedTestCase.parametrize(
                    TestNumpyFFT,
                    param={
                        "transform_infos": transform_infos,
                        "test_data": test_data,
                        "trdim": trdim,
                        "mode": mode,
                        "size": size,
                    }
                )
                testSuite.addTest(testcase)
    return testSuite


def test_fft(backend, dimensions=None):
    testSuite = unittest.TestSuite()
    transform_infos = TransformInfos()
    test_data = TestData()
    dimensions = dimensions or transform_infos.dimensions

    print("Testing backend: %s" % backend)
    for trdim in dimensions:
        print("   testing %s" % trdim)
        for mode in transform_infos.modes:
            print("   testing %s:%s" % (trdim, mode))
            for size in transform_infos.sizes[trdim]:
                print("      size: %s" % str(size))
                testcase = ParametrizedTestCase.parametrize(
                    TestFFT,
                    param={
                        "transform_infos": transform_infos,
                        "test_data": test_data,
                        "backend": backend,
                        "trdim": trdim,
                        "mode": mode,
                        "size": size,
                    }
                )
                testSuite.addTest(testcase)
    return testSuite


def test_all():
    suite = unittest.TestSuite()

    suite.addTest(test_numpy_backend())

    #~ suite.addTest(test_fft("fftw"))
    suite.addTest(test_fft("opencl"))
    #~ suite.addTest(test_fft("cuda"))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest="test_all")


