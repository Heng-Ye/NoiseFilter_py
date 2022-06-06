import csv
#import math
import numpy as np
import matplotlib.pyplot as plt

import operator
import sys

from pylab import *
from math import *

import pywt
import seaborn

#import Timfunctions as Tf
from statsmodels import robust
from matplotlib2tikz import save as tikz_save
import copy as copy

#from statsmodels.robust import mad

def decomposite(signal, coef_type='d', wname='db6', level=9):
    w = pywt.Wavelet(wname)
    a = raw
    ca = []
    cd = []
    for i in range(level):
        (a, d) = pywt.dwt(a, w, mode='soft')
        ca.append(a)
        cd.append(d)
    rec_a = []
    rec_d = []
    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))
    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w))
    if coef_type == 'd':
        return rec_d
    return rec_a

#fin='wf_4696_358_1_1_261.root.txt'
fin='wf_4696_358_1_0_188.root.txt'

#file = open(fin,"r") 
#print(file.readlines()) 
#file.close() 

raw = []
with open(fin, 'r') as f:
    f.readline()  # skip first line
    for number in f:
        raw.append(int(number))

#De-noising
#(ca, cd) = pywt.dwt(raw,'haar')
#rec_d = decomposite(raw, 'd', 'haar', level=1)

# compute the wavelet coefficients of raw waveform
#fw = pywt.wavedec(raw, 'haar')

#Plot the level 2 detail
#plt.plot(fw[-2], linestyle='steps')
#plt.show()

#f_prime = pywt.waverec(fw, 'haar') # reconstruct the signal

wavelet="haar"
decomp=pywt.wavedec(raw,wavelet)

# Universal threshold
T_U=sqrt(2*np.log(len(raw)))*robust. mad(decomp[ -1])/0.6745

Thres_hard = copy . deepcopy ( decomp )
Thres_soft = copy . deepcopy ( decomp )
Thres_scale = copy . deepcopy ( decomp )

for i in range (len(decomp))[1:] :
  Thres_hard[i][:]=pywt.threshold(Thres_hard[i][:],T_U ,'hard')
  Thres_soft[i][:]=pywt.threshold(Thres_soft[i][:],T_U/2,'soft')
  Thres_scale[i][:]=pywt.threshold(Thres_scale[i][:],1.3*std(Thres_scale[i][:]),'soft')

x_filt_hard=pywt.waverec(Thres_hard,wavelet)
x_filt_soft=pywt.waverec(Thres_soft,wavelet)
x_filt_scale=pywt.waverec(Thres_scale,wavelet)



ax = plt.subplot(1,3,1)
plt.plot(x_filt_hard)

ax = plt.subplot(1,3,2)
plt.plot(x_filt_soft)

ax = plt.subplot(1,3,3)
plt.plot(x_filt_scale)

plt.show()

#ax = plt.subplot(3,1,3)
#plt.plot(time_ref[:], chofchs[k][:], label='{}'.format(leg_title[k:]), color=color_ch[k], linewidth=0.6)












#cat = pywt.thresholding.soft(ca, np.std(ca)/2)
#cdt = pywt.thresholding.soft(cd, np.std(cd)/2)

#cat = pywt.threshold(ca, np.std(ca)/2, mode='soft')
#cdt = pywt.threshold(cd, np.std(cd)/2, mode='soft')

#cat = pywt.threshold(ca, np.std(ca)/1, mode='soft')
#cat = pywt.threshold(ca, 0, mode='soft')
#cdt = pywt.threshold(cd, 10, mode='soft')

#cat = pywt.threshold(ca, value=4., mode='hard')
#cdt = pywt.threshold(cd, value=100, mode='hard')

#cat = pywt.threshold(ca, value=0.5, mode='garotte')
#cdt = pywt.threshold(cd, value=0.5, mode='garotte')

'''
ts_rec = pywt.idwt(cat, cdt, 'haar')
plt.close('all')
plt.subplot(211)
# Original coefficients
plt.plot(ca, '--*b')
plt.plot(cd, '--*r')
# Thresholded coefficients
plt.plot(cat, '--*c')
plt.plot(cdt, '--*m')
plt.legend(['ca','cd','ca_thresh', 'cd_thresh'], loc=0)
plt.grid('on')

plt.subplot(212)
plt.plot(raw)
plt.hold('on')
plt.plot(ts_rec, 'r')
#plt.legend(['original signal', 'reconstructed signal'])
plt.legend(['original signal', 'reconstructed signal'])
plt.grid('on')
plt.show()



plt.figure(figsize=(20,10))
plt.rcParams['axes.grid'] = True
ax = plt.subplot(1,1,1)
plt.plot(raw[:], label='{}'.format('raw waveform'), color='black', linewidth=0.6)
plt.show()
'''