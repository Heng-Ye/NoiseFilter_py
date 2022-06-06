import pywt
import pylab
import matplotlib.pyplot as plt
import math
import statistics

import numpy as np
import operator
import seaborn
from statsmodels.robust import mad
 
def waveletSmooth( x, wavelet="db4", level=1, title=None ):
    # calculate the wavelet coefficients
    coeff = pywt.wavedec( x, wavelet, mode="per" )
    # calculate a threshold
    sigma = (1./0.6745) * mad( coeff[-level] )
    # changing this threshold also changes the behavior,
    # but I have not played with this very much
    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )
    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode="soft" ) for i in coeff[1:] )
    # reconstruct the signal using the thresholded coefficients
    y = pywt.waverec( coeff, wavelet, mode="per" )
    f, ax = plt.subplots()
    plt.plot( x, color="b", alpha=0.5 )
    plt.plot( y, color="b" )
    if title:
        ax.set_title(title)
    ax.set_xlim((0,len(y)))
    plt.show()

#def snr(x, imin, imax):
    #standard deviation
    #std=statistics.stdev(x[imin,imax])
    #print("std:{:.2f}".format(std))
    #av=sum(x[imin:imax])/len(x[imin:imax])
    #av=sum([i for i in x if isinstance(i, float)])/len(imin:imax)
    #print("baseline average:{:.2f}".format(av))

def mean(data):
    """Return the sample arithmetic mean of data."""
    n = len(data)
    if n < 1:
        raise ValueError('mean requires at least one data point')
    return sum(data)/n # in Python 2 use sum(data)/float(n)

def _ss(data):
    """Return sum of square deviations of sequence data."""
    c = mean(data)
    ss = sum((x-c)**2 for x in data)
    return ss

def stddev(data, ddof=0):
    """Calculates the population standard deviation
    by default; specify ddof=1 to compute the sample
    standard deviation."""
    n = len(data)
    if n < 2:
        raise ValueError('variance requires at least two data points')
    ss = _ss(data)
    pvar = ss/(n-ddof)
    return pvar**0.5

def savitzky_golay(y, window_size, order, deriv=0, rate=1):

    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    #except ValueError, msg:
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

#read data
#low-freq signal
#fin='wf_4696_358_1_0_188.root.txt'
#noise
#fin='wf_4696_358_1_1_261.root.txt'
#high-freq signal
fin='wf_4696_358_8_0_254.root.txt'
#coherent-noise
#fin='wf_4696_1_0_109.root.txt'

raw = []
with open(fin, 'r') as f:
    f.readline()  # skip first line
    for number in f:
        raw.append(float(number))


#Wavelet decomposition #########################################################################
#wavelet basis
#wavelet_basis="db5"
#wavelet_basis="db9"
wavelet_basis="bior3.3"
#wavelet_basis="haar"
#wavelet_basis="bior1.3"
#wavelet_basis="bior3.1"
#wavelet_basis="db3"

#number of decomposition
#n_decom=9
n_decom=10

#get wavelet coefficients
coeff = pywt.wavedec(raw, wavelet_basis, mode="per" )
len_map = list(map(len, coeff))
#print(list(map(len, coeff)))
#print(len(coeff))

coeff_after_cut = coeff.copy()
#print(coeff)
#print(coeff[:][1])
#print(coeff[0:11][0])
#print(coeff[:][0])

#define adaptive global thresholding
sigma = (1./0.6745) * mad( coeff[-n_decom] )
uthresh = sigma * np.sqrt( 2*np.log( len( raw ) ) )
print('sigma:{} uth:{}'.format(sigma,uthresh))

#apply thresholding
#[1:] NOT apply threshold on cA1, apply other coefficients
coeff_after_cut[1:] = ( pywt.threshold( i, value=uthresh, mode="soft" ) for i in coeff_after_cut[1:] )

#reconstruct the signal using the thresholded coefficients
reco = pywt.waverec( coeff_after_cut, wavelet_basis, mode="per" )

#Apply low-pass filter in advance
#win_size=301
win_size=51
pol_order=3
smooth = savitzky_golay(reco, win_size, pol_order) 

#calculate SNR
imin=0
imax=1000
rms_raw=stddev(raw[imin:imax], ddof=1)
rms_reco=stddev(reco[imin:imax], ddof=1)
rms_smooth=stddev(smooth[imin:imax], ddof=1)

tmax_raw, amax_raw = max(enumerate(raw), key=operator.itemgetter(1))
tmax_reco, amax_reco = max(enumerate(reco), key=operator.itemgetter(1))
tmax_smooth, amax_smooth = max(enumerate(smooth), key=operator.itemgetter(1))

mean_raw = mean(raw[imin:imax])
mean_reco = mean(reco[imin:imax])
mean_smooth = mean(smooth[imin:imax])

if (mean_raw<0): mean_raw = -1.*mean_raw 
if (mean_reco<0): mean_reco = -1.*mean_reco
if (mean_smooth<0): mean_smooth = -1.*mean_smooth
snr_raw=amax_raw/rms_raw
snr_reco=amax_reco/rms_reco
#snr_raw=mean_raw/rms_raw
#snr_reco=mean_reco/rms_reco
snr_smooth=mean_smooth/rms_smooth

print("std_dev(raw):{:.2f}".format(rms_raw))
print("std_dev(reco):{:.2f}".format(rms_reco))
print("std_dev(smooth):{:.2f}".format(rms_smooth))

print("\namax(raw):{:.2f}".format(amax_raw))
print("amax(reco):{:.2f}".format(amax_reco))
print("amax(smooth):{:.2f}".format(amax_smooth))

print("\nsnr(raw):{:.2f}".format(snr_raw))
print("snr(reco):{:.2f}".format(snr_reco))
print("snr(smooth):{:.2f}".format(snr_smooth))

#draw reco signals
plt.figure(figsize=(4,3))
s_plt = plt.figure(1)
f, ax = plt.subplots()
#f, ax = plt.subplots()
plt.plot(raw, color="b", alpha=0.5 )
plt.plot(reco, color="r" )
#plt.plot(smooth, color="black" )
ax.set_title('ProtoDUNE Raw Trace')
ax.set_xlim((0,len(reco)))

plt.legend(['original signal (SNR:{:.2f})'.format(snr_raw), 'wavelet-denoising (SNR:{:.2f})'.format(snr_reco)])
#plt.legend(['original signal (SNR:{:.2f})'.format(snr_raw), 'wavelet-denoising (SNR:{:.2f})'.format(snr_reco), 'wavelet+SG Filter (SNR:{:.2f})'.format(snr_smooth)])

#waveletSmooth(raw, wavelet=wavelet_basis, level=n_decom, title=None )
#################################################################################################

decom_plt = plt.figure(2)
plt.figure(figsize=(10,16))

index_plt=0
sum=0
for j in range(n_decom):
    k=j+1
    #print('k:',k)
    #print('j:',j)
    index_plt+=1
    
    index_1=sum
    index_2=sum+len_map[j]-1
    sum+=len_map[j]
    #print('{}:{}'.format(index_1,index_2))

    #cA
    plt.rcParams['axes.grid'] = True
    plt.subplot(n_decom,1,index_plt)

    #plt.plot(coeff[index_1:index_2][0], color='blue')
    plt.plot(coeff[:][j], color='blue')
    plt.plot(coeff_after_cut[:][j], 'r', color='red')
 
    #cD
    #plt.rcParams['axes.grid'] = True
    #index_plt=index_plt+1
    #plt.subplot(n_decom,2,index_plt)
    #plt.plot(coeff[index_1:index_2][1], color='black')
    #plt.plot(coeff_after_cut[n_decom-k][1], 'r', color='red')

#plt.set_title('Wavelet Coefficients')

#waveletSmooth(raw, wavelet=wavelet_basis, level=n_decom, title=None )

plt.show()