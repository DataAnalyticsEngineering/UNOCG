"""
Fourier-based Transforms
"""
import torch
import torch.nn.functional as F
from unocg.transforms import Transform
from unocg import Backend
from collections.abc import Iterable
import numpy as np
import math

try:
    import scipy as sp
except:
    pass

try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
except:
    pass


class DiscreteFourierTransform(Transform):
    """
    Discrete Fourier Transform
    """
    def __init__(self, dim=-1, norm="backward", real=True, backend=Backend.TORCH):
        """
        Constructor

        :param n_dim:
        :param norm:
        :param real:
        :param backend:
        """
        super().__init__()
        if isinstance(dim, Iterable):
            self.dim = dim
        else:
            self.dim = [dim]
        self.norm = norm
        self.real = real
        self.backend = backend

        if self.backend == Backend.TORCH:
            if self.real:
                self.fftn = torch.fft.rfftn
                self.ifftn = torch.fft.irfftn
            else:
                self.fftn = torch.fft.fftn
                self.ifftn = torch.fft.ifftn
        elif self.backend == Backend.SCIPY:
            try:
                if self.real:
                    self.fftn = sp.fft.rfftn
                    self.ifftn = sp.fft.irfftn
                else:
                    self.fftn = sp.fft.fftn
                    self.ifftn = sp.fft.ifftn
            except ImportError:
                raise ImportError("SciPy installation not found, but was specified as backend")
        elif self.backend == Backend.FFTW:
            try:
                self.pyfftw_threads = 16
                if self.real:

                    def fftn(x, dim, *args, **kwargs):
                        if isinstance(x, torch.Tensor):
                            x = x.detach().cpu().numpy()
                        return pyfftw.interfaces.numpy_fft.rfftn(x, axes=dim, *args, **kwargs, threads=self.pyfftw_threads)
                    
                    self.fftn = fftn

                    def ifftn(x, dim, *args, **kwargs):
                        if isinstance(x, torch.Tensor):
                            x = x.detach().cpu().numpy()
                        return pyfftw.interfaces.numpy_fft.irfftn(x, axes=dim, *args, **kwargs, threads=self.pyfftw_threads)
                    
                    self.ifftn = ifftn
                else:

                    def fftn(x, dim, *args, **kwargs):
                        if isinstance(x, torch.Tensor):
                            x = x.detach().cpu().numpy()
                        return pyfftw.interfaces.numpy_fft.fftn(x, axes=dim, *args, **kwargs, threads=self.pyfftw_threads)
                    
                    self.fftn = fftn

                    def ifftn(x, dim, *args, **kwargs):
                        if isinstance(x, torch.Tensor):
                            x = x.detach().cpu().numpy()
                        return pyfftw.interfaces.numpy_fft.ifftn(x, axes=dim, *args, **kwargs, threads=self.pyfftw_threads)
                    
                    self.ifftn = ifftn
            except ImportError:
                raise ImportError("PyFFTW installation not found, but was specified as backend")
        else:
            raise NotImplementedError(f"Backend {self.backend} is not implemented for this operation")

    def transform(self, x, s=None):
        s_dim = None if s is None else tuple(np.atleast_1d(np.atleast_1d(s)[self.dim]))
        return self.fftn(x, dim=self.dim, s=s_dim)

    def inverse_transform(self, x, s=None):
        s_dim = None if s is None else tuple(np.atleast_1d(np.atleast_1d(s)[self.dim]))
        return self.ifftn(x, dim=self.dim, s=s_dim)
    
    def adjoint_transform(self, x, s=None):
        s_dim = None if s is None else np.atleast_1d(np.atleast_1d(s)[self.dim])
        return self.inverse_transform(x, s=s_dim)


class DiscreteSineTransform(Transform):
    """
    Discrete Sine Transform
    """
    def __init__(self, dim=-1, norm="ortho", type=1, backend=Backend.TORCH):
        """
        Constructor

        :param n_dim:
        :param norm:
        :param type:
        :param backend:
        """
        super().__init__()
        if isinstance(dim, Iterable):
            self.dim = dim
        else:
            self.dim = [dim]
        self.type = type
        self.backend = backend
        self.norm = norm
        self.rfft_transform = DiscreteFourierTransform(real=True, norm=self.norm, dim=-1, backend=self.backend)
        self.pyfftw_threads = 16

        if self.type != 1:
            raise NotImplementedError(f"Discrete Sine Transform of type {self.type} not implemented")

    def transform(self, x, s=None):
        s = None if s is None else np.atleast_1d(s)
        for dim in self.dim:
            s_dim = None if s is None else np.atleast_1d(s[dim])
            x = self.dst_1d(x, dim=dim, s=s_dim)
        return x

    def inverse_transform(self, x, s=None):
        s = None if s is None else np.atleast_1d(s)
        for dim in self.dim:
            s_dim = None if s is None else np.atleast_1d(s[dim])
            x = self.idst_1d(x, dim=dim, s=s_dim)
        return x
    
    def adjoint_transform(self, x, s=None):
        s = None if s is None else np.atleast_1d(s)
        return self.transform(x)

    def dst_1d(self, x, s=None, dim=-1):
        if self.backend == Backend.TORCH:
            n = x.shape[dim]
            padded_x = torch.tensor(-1j, device=x.device) * F.pad(x.swapaxes(dim, -1), (1, 1))
            padded_x_hat = self.rfft_transform.inverse_transform(padded_x)
            x_hat = padded_x_hat[...,1:n+1].real.swapaxes(-1, dim) * math.sqrt(2 * (n + 1))
        elif self.backend == Backend.FFTW:
            x_hat = pyfftw.interfaces.scipy_fft.dstn(x, axes=dim, type=self.type, norm=self.norm, workers=self.pyfftw_threads)
        elif self.backend == Backend.SCIPY:
            x_hat = sp.fft.dstn(x, axes=dim, type=self.type, norm=self.norm)
        else:
            raise NotImplementedError(f"Backend {self.backend} is not supported for this operation")
        return x_hat

    def idst_1d(self, x, s=None, dim=-1):
        if self.backend == Backend.TORCH:
            n = x.shape[dim]
            x_hat = self.dst_1d(x, s=s, dim=dim)
        elif self.backend == Backend.FFTW:
            x_hat = pyfftw.interfaces.scipy_fft.idstn(x, axes=dim, type=self.type, norm=self.norm, workers=self.pyfftw_threads)
        elif self.backend == Backend.SCIPY:
            x_hat = sp.fft.idstn(x, axes=dim, type=self.type, norm=self.norm, workers=self.pyfftw_threads)
        else:
            raise NotImplementedError(f"Backend {self.backend} is not supported for this operation")
        return x_hat


class DiscreteCosineTransform(Transform):
    """
    Discrete Cosine Transform
    """
    def __init__(self, dim=-1, norm="ortho", type=1, backend=Backend.TORCH):
        """
        Constructor

        :param n_dim:
        :param norm:
        :param type:
        """
        super().__init__()
        if isinstance(dim, Iterable):
            self.dim = dim
        else:
            self.dim = [dim]
        self.type = type
        self.norm = norm
        self.backend = backend
        self.pyfftw_threads = 16

        if self.backend == Backend.TORCH:
            self.rfft_transform = DiscreteFourierTransform(real=True, norm=self.norm, dim=-1, backend=self.backend)

        if self.type != 1:
            raise NotImplementedError(f"Discrete Cosine Transform of type {self.type} not implemented")

    def transform(self, x, s=None):
        for dim in self.dim:
            x = self.dct_1d(x, dim=dim, s=s)
        return x

    def inverse_transform(self, x, s=None):
        for dim in self.dim:
            x = self.idct_1d(x, dim=dim, s=s)
        return x

    def adjoint_transform(self, x, s=None):
        # TODO: check if this is correct!
        return self.transform(x, s=s)

    def dct_1d(self, x, s=None, dim=-1):
        """
        Discrete Cosine Transform, Type I in 1D

        :param x: the input signal
        :return: the DCT-I of the signal over the last dimension
        """
        if self.backend == Backend.TORCH:
            if self.norm != "ortho":
                raise NotImplementedError(f"Norm {norm} not implemented")
            xt = x.swapaxes(dim, -1)
            xt[..., 0] *= math.sqrt(2)
            xt[..., -1] *= math.sqrt(2)
            padded_xt = torch.cat([xt, xt.flip([-1])[..., 1:-1]], dim=-1)
            xt_hat = self.rfft_transform.transform(padded_xt).real
            xt_hat[..., 0] /= math.sqrt(2)
            xt_hat[..., -1] /= math.sqrt(2)
            xt_hat /= math.sqrt(2 * (xt.shape[-1] - 1))
            x_hat = xt_hat.swapaxes(dim, -1)
        elif self.backend == Backend.FFTW:
            x_hat = pyfftw.interfaces.scipy_fft.dctn(x, axes=dim, type=self.type, norm=self.norm, workers=self.pyfftw_threads)
        elif self.backend == Backend.SCIPY:
            x_hat = torch.tensor(sp.fft.dctn(x.cpu().numpy(), axes=dim, type=self.type, norm=self.norm))
        else:
            raise NotImplementedError(f"Backend {self.backend} is not supported for this operation")
        return x_hat

    def idct_1d(self, x, s=None, dim=-1):
        """
        The inverse of DCT-I in 1D, which is just a scaled DCT-I

        :param X: the input signal
        :return: the inverse DCT-I of the signal over the last dimension
        """
        if self.backend == Backend.TORCH:
            if self.norm != "ortho":
                raise NotImplementedError(f"Norm {norm} not implemented")
            x_hat = self.dct_1d(x, s=s, dim=dim)
        elif self.backend == Backend.FFTW:
            x_hat = pyfftw.interfaces.scipy_fft.idctn(x, axes=dim, type=self.type, norm=self.norm, workers=self.pyfftw_threads)
        elif self.backend == Backend.SCIPY:
            x_hat = torch.tensor(sp.fft.idctn(x.cpu().numpy(), axes=dim, type=self.type, norm=self.norm))
        else:
            raise NotImplementedError(f"Backend {self.backend} is not supported for this operation")
        return x_hat


class DiscreteMixedTransform(Transform):
    """
    Discrete Mixed Transform
    """
    def __init__(self, dim_fourier=None, dim_sine=None, dim_cosine=None, real=True, norm="backward", type=1, backend=Backend.TORCH):
        """
        Constructor

        :param n_dim:
        :param norm:
        :param type:
        """
        super().__init__()
        self.type = type
        self.norm = norm

        if dim_fourier is None:
            self.dim_fourier = []
        else:
            self.dim_fourier = dim_fourier

        if dim_sine is None:
            self.dim_sine = []
        else:
            self.dim_sine = dim_sine

        if dim_cosine is None:
            self.dim_cosine = []
        else:
            self.dim_cosine = dim_cosine
        
        self.rfft_transform = DiscreteFourierTransform(real=real, norm=norm, dim=self.dim_fourier, backend=backend)
        self.dst_transform = DiscreteSineTransform(norm=norm, dim=self.dim_sine, backend=backend)
        self.dct_transform = DiscreteCosineTransform(norm=norm, dim=self.dim_cosine, backend=backend)

        if self.type != 1:
            raise NotImplementedError(f"Discrete Mixed Transform of type {self.type} not implemented")

    def transform(self, x, s=None):
        s = None if s is None else np.atleast_1d(s)
        x = self.dst_transform.transform(x, s=s)
        x = self.dct_transform.transform(x, s=s)
        x = self.rfft_transform.transform(x, s=s)
        return x

    def inverse_transform(self, x, s=None):
        s = None if s is None else np.atleast_1d(s)
        x = self.rfft_transform.inverse_transform(x, s=s)
        x = self.dct_transform.inverse_transform(x, s=s)
        x = self.dst_transform.inverse_transform(x, s=s)
        return x

    def adjoint_transform(self, x, s=None):
        s = None if s is None else np.atleast_1d(s)
        x = self.rfft_transform.adjoint_transform(x, s=s)
        x = self.dct_transform.adjoint_transform(x, s=s)
        x = self.dst_transform.adjoint_transform(x, s=s)
        return x
