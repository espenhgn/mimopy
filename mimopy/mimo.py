#!/usr/bin/env python
'''
mimo.py - Prediction of transfer functions assuming a linear
multiple-input multiple-output system
'''
from __future__ import division
import numpy as np
import scipy.linalg as sl
from matplotlib.mlab import csd
import unittest


class MIMO(object):
    def __init__(self, NFFT=256, noverlap=128, Fs=2):
        '''
        Method of Perreault et al. (1999) Biol. Cybern. 80, 327-337 in frequency
        domain using Welch's method to estimate transfer functions of a linear
        system of X measured inputs and Y measured outputs, where Y <= X
            
        Arguments
        ---------
        *NFFT*: integer
            The number of data points used in each block for the FFT.
            A power 2 is most efficient.  The default value is 256.
        *noverlap*: integer
            The number of points of overlap between segments.
            The default value is 0 (no overlap).
        *Fs*: scalar
            The sampling frequency (samples per time unit).  It is used
            to calculate the Fourier frequencies, freqs, in cycles per time
            unit. The default value is 2.
        '''        
        self.NFFT = NFFT
        self.noverlap = noverlap
        self.Fs = Fs
        
    
    def get_cross_spectra(self, x, y):
        '''
        Compute cross-spectra between x and y using the method implemented in
        matplotlib.mlab.csd
        
        Arguments
        ---------
        *x*, *y* : np.ndarray
            signals, shape (n_x/n_y, T), n_x/n_y is number of signals,
            T the number of samples
        
        Returns
        -------
        *C_xy* : np.ndarray
            complex array if shape (n_x, n_y, NFFT) containing the cross-spectra
        '''
        
        n_x = x.shape[0]
        n_y = y.shape[0]
        
        try:
            assert(n_x >= n_y)
        except AssertionError as ae:
            raise ae, 'n_x = {} < n_y = {}'.format(n_x, n_y)
        
        #compute cross-spectra
        C_xy = np.empty((n_x, n_y, self.NFFT), dtype=complex)
        for i in range(n_x):
            for j in range(n_y):
                G, f = csd(x[i, ], y[j, ], NFFT=self.NFFT, Fs=self.Fs,
                           noverlap=self.noverlap,
                           sides='twosided')
                C_xy[i, j, ] = G
        
        return C_xy


    def get_complex_transferfunctions(self, x, y):
        '''
        Compute the complex transfer functions between the
        inputs x and outputs y
        
        Arguments
        ---------
        *x*: np.ndarray
            input signals, shape (n_x, T), n_x is number of signals,
            T the number of samples
        *y* : np.ndarray
            input signals, shape (n_y, T), n_y is number of signals,
            T the number of samples
        
        Returns
        -------
        *H_xy* : np.ndarray
            complex array of shape (n_x, n_y, NFFT) containing the complex
            transfer functions
        '''
        #input-output cross-spectra
        C_xy = self.get_cross_spectra(x, y)
        
        #input-input cross-spectra
        C_xx = self.get_cross_spectra(x, x)
        
        #frequency domain solution of transfer functions H_yx, given that
        #C_xy(f) = C_xx(f) x H_yx(f)
        H_yx = np.zeros_like(C_xy)
        for i in range(H_yx.shape[-1]):
            H_yx[:, :, i] = sl.solve(C_xx[:, :, i], C_xy[:, :, i])

        #swap indices so that the IRF can be obtained by iFFT
        H_yx = np.c_[H_yx[:, :, self.NFFT//2:], H_yx[:, :, :self.NFFT//2]]
        
        return H_yx


    def get_real_transferfunctions(self, x, y):
        '''
        Compute the complex transfer functions between the
        inputs x and outputs y
        
        Arguments
        ---------
        *x*: np.ndarray
            input signals, shape (n_x, T), n_x is number of signals,
            T the number of samples
        *y* : np.ndarray
            input signals, shape (n_y, T), n_y is number of signals,
            T the number of samples
        
        Returns
        -------
        *h_xy* : np.ndarray
            real-valued array of shape (n_x, n_y, NFFT) containing the
            transfer functions as function of lag
        '''
        #get complex transfer function
        H_yx = self.get_complex_transferfunctions(x, y)
        
        #backtransform H to time domain, keep real part
        return np.fft.ifft(H_yx).real


