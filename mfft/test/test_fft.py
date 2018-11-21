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
        self.data = ascent().astype("float32")
        #~ self.F = FFT(data=self.data[:, 1], check_alignment=True)

    def tearDown(self):
        self.data = None


    def test_plan_creation(self):
        #~ plan_numpy = FFT(data=self.data[:, 0], backend="numpy")
        plan_fftw = FFT(data=self.data[:, 0], backend="fftw", check_alignment=True)
        #~ plan_opencl = FFT(data=self.data[:, 0], backend="opencl")
        #~ plan_cuda = FFT(data=self.data[:, 0], backend="cuda")


    def test_forward_FFT(self):
        data1d = self.data[:, 0]
        F = FFT(data=data1d, backend="fftw", check_alignment=True)
        res = F.fft(data1d)





def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestFFT("test_plan_creation"))
    #~ testSuite.addTest(TestFFT("test_forward_FFT"))
    #~ for test_name, test_params in test_cases.items():
        #~ testSuite.addTest(parameterize(TestFFT, name=test_name, params=test_params))
    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
