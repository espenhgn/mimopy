#!/usr/bin/env python
'''test implementation of methods in Perreault et al 1999'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mimo import MIMO

np.random.seed(123456)

#parameters
n_x = 3     #number of inputs
n_y = 3     #number of outputs
n = 10000   #signal length
n_lag = 128     #full length of transfer function coefficients is 2*n_lag

#PSD parameters
NFFT = 512
noverlap = NFFT*3/4
Fs = 2


################################################################################
# Generate test input and output data including noise and correlation in input
################################################################################
#correlation in input
correlation = 0.1

#noise variance in output
noisevar = 0

#time constant of each kernel
tau = np.random.rand(n_x*n_y)*15

#amplitude of each kernel
scaling = (np.random.rand(n_x*n_y)-0.5)*10

#generate input signals (white noise)
x = np.random.randn(n_x, n)

#make input correlated
x = np.dot(x.T, np.eye(n_x)*(1-correlation/2)+correlation/2).T

#generate transfer functions (time domain), convert to frequency domain for
#validation
lag = np.arange(-n_lag, n_lag)
h_tau = np.zeros((n_x, n_y, n_lag*2))
H_tau = np.zeros((n_x, n_y, NFFT), dtype='complex')
k = 0
for i in range(n_x):
    for j in range(n_y):
        h_tau[i, j, n_lag:] = scaling[k]*np.exp(-np.arange(n_lag) / tau[k]
                                                  )*np.arange(n_lag)*tau[k]**-2
        H_tau[i, j] = np.fft.fft(h_tau[i, j], NFFT)
        k += 1
        

#generate output signals by convolving each respective input with transfer fun 
y = np.zeros((n_y, n))
for i in range(n_x):
    for j in range(n_y):
        y[j] += np.convolve(x[i], h_tau[i, j, ], 'same')
for j in range(n_y):
    #add uncorrelated noise to output
    y[j] += np.random.randn(n)*noisevar

        

mimo = MIMO(NFFT=NFFT, Fs=Fs, noverlap=noverlap)
H_yx = mimo.get_complex_transferfunctions(x, y)
h_yx = mimo.get_real_transferfunctions(x, y)

#fix indices so that the onset of IRF is at tau=0, and slice to the length of
#the actual convolution kernels
h_yx = np.c_[h_yx[:, :, NFFT-n_lag+1:], h_yx[:, :, :n_lag+1]]


#test plot
fig, axes = plt.subplots(4,2, figsize=(8,8))
axes[0, 0].plot(lag, h_tau.reshape(n_x*n_y, -1).T)
axes[0, 0].set_ylabel(r'$h(\tau)$')
axes[0, 0].set_title('ground truth kernels')
axes[1, 0].semilogy(np.abs(H_tau.reshape(n_x*n_y, -1)).T)
axes[1, 0].set_ylabel(r'$|H(f)|$')
axes[2, 0].plot(H_tau.reshape(n_x*n_y, -1).imag.T)
axes[2, 0].set_ylabel(r'$\Re H(f)$')
axes[3, 0].plot(H_tau.reshape(n_x*n_y, -1).real.T)
axes[3, 0].set_ylabel(r'$\Im H(f)$')

axes[0, 1].plot(lag, h_yx.reshape(n_x*n_y, -1).T)
axes[0, 1].set_title('reconstructed kernels')
axes[1, 1].semilogy(np.abs(H_yx.reshape(n_x*n_y, -1)).T)
axes[2, 1].plot(H_yx.reshape(n_x*n_y, -1).imag.T)
axes[3, 1].plot(H_yx.reshape(n_x*n_y, -1).real.T)

for ax in np.array(axes).flatten():
    ax.axis(ax.axis('tight'))


#test plot
fig, axes = plt.subplots(n_x, 2, figsize=(8,8))
axes = axes.T.flatten()
k = 0
for i in range(n_x):
    if i == 0:
        axes[k].set_title('inputs $x_i(t)$')
    axes[k].plot(x[i])
    axes[k].set_ylabel(r'$x_{%i}$' % i, labelpad=0)
    k += 1
for j in range(n_y):
    if j == 0:
        axes[k].set_title('outputs $y_j(t)$')
    axes[k].plot(y[j])
    axes[k].set_ylabel(r'$y_{%i}$' % j, labelpad=0)
    k += 1


plt.show()

