import numpy as np

from .fft import FFT, __have_fftw__

if __have_fftw__:
    import pyfftw


# TODO support in-place ? In this case, pyfftw.builders cannot be used
class FFTW(FFT):
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
        check_alignment=False,
    ):
        """
        Initialize a FFTW plan.
        Please see FFT class for parameters help.

        FFTW-specific parameters:
        check_alignment: bool, optional
            If set to True and "data" is provided, this will enforce the input data
            to be "byte aligned", which might imply extra memory usage.
        """
        super().__init__(
            shape=shape,
            dtype=dtype,
            data=data,
            shape_out=shape_out,
            double_precision=double_precision,
        )
        self.check_alignment = check_alignment
        self.set_input_data()


    def set_input_data(self):
        """
        FFTW-specific data formatting
        """
        if self.user_data is not None:
            if self.check_alignment and not(pyfftw.is_byte_aligned(self.user_data)):
                self.data_in = pyfftw.zeros_aligned(self.shape, dtype=self.dtype_in)
                np.copyto(self.data_in, self.user_data)
            else:
                self.data_in = self.user_data
        else:
            self.data_in = pyfftw.zeros_aligned(self.shape, dtype=self.dtype_in)


    def compute_forward_plan(self):
        pass


    def compute_inverse_plan(self):
        pass





"""
Workflow
---------

L'utilisateur définit un plan :
  (data_shape, data_type, force_complex)
  (data) => infer


commun à toutes les classes (trivial pour numpy):
set_input_data
set_output_data


"""

