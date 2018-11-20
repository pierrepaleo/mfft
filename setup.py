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

from setuptools import setup
import os

def get_version():
    with open("mfft/__init__.py", "r") as fid:
        lines = fid.readlines()
    version = None
    status = None
    for line in lines:
        if "version" in line:
            version = line.rstrip().split("=")[-1].lstrip()
        if "status" in line:
            status = line.rstrip().split("=")[-1].lstrip()
    if version is None:
        raise RuntimeError("Could not find version from __init__.py")
    version = version.strip("'").strip('"')
    status = status.strip("'").strip('"')
    return version, status


def setup_package():

    version, status = get_version()
    version = ".".join([version, status])

    packages = ["mfft"]
    package_dir = {
        "mfft": "mfft",
        "mfft.test": "mfft/test",
    }
    setup(
        name='mfft',
        author='Pierre Paleo',
        version=version,
        author_email = "pierre.paleo@esrf.fr",
        maintainer = "Pierre Paleo",
        maintainer_email = "pierre.paleo@esrf.fr",

        packages=packages,
        package_dir = package_dir,
        #~ package_data = {'': ['data/testdata.npz']},

        install_requires = [
          'numpy',
        ],
        #~ dependency_links = [ ... ],

        #~ entry_points = {
            #~ 'console_scripts': ['test_mfft=mfft.test:main'],
        #~ },

        long_description = """
        Multi Back-end FFT
        """,

        zip_safe=True
    )


if __name__ == "__main__":
    setup_package()
