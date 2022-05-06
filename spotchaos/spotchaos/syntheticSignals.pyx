# cython: profile=True
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
from astropy.timeseries import LombScargle
#from scipy.integrate import RK45
from scipy.stats import iqr
from scipy.signal import argrelextrema
from scipy.spatial.distance import chebyshev

__all__ = ['plotTimeSeries', 'KB88', 'Rossler_FPs', 'Rossler_vel', 'rotated_Rossler_vel','Lorenz_FPs','Lorenz_vel','bin2D','calc_MI','shan_entropy','optimal_Nbins','moving_average','FS86','delayMatrix','nearestNeighborIndices','cao97','localDensity','Cq','direct_C2']


def plotTimeSeries(t, y, min_freq=None, max_freq=None, nq=None, spp=10, true_freq=None, LS_xlim=None, plot_harmonics=False, title=None):
    """
    plot time series, histogram of time series, and Lomb-Scargle periodogram of time series
    
    inputs:
    t : np.array, t values of time series
    y : np.array, y values of time series
    min_freq : float, passed to astropy.timeseries.LombScargle minimum_frequency
    max_freq : float, passed to astropy.timeseries.LombScargle maximum_frequency
    nq : float, passed to astropy.timeseries.LombScargle nyquist_factor
    spp : float, passed to astropy.timeseries.LombScargle samples_per_peak
    true_freq : array-like; if specified, plot the true frequenc(ies) as vertical line(s) on the LS periodogram
    LS_xlim : tuple, x limits for LS periodogram subplot
    plot_harmonics : bool; if True, plot vertical lines at true_freq/2. and true_freq/3.
    title : string, plot title
    
    outputs:
    None
    """
    
    frequency, power = LombScargle(t,y).autopower(minimum_frequency=min_freq, maximum_frequency=max_freq, nyquist_factor=nq, samples_per_peak=spp)
    
    fig = plt.figure(figsize=(16,8))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    ax00 = fig.add_subplot(gs[0, :])
    ax00.plot(t, y)
    ax00.set_xlabel('time')
    ax00.set_ylabel('signal')
    if title is not None:
        ax00.set_title('{0} time series'.format(title))
    else:
        ax00.set_title('time series')

    ax10 = fig.add_subplot(gs[1, 0])
    ax10.hist(y,bins=30)
    ax10.set_xlabel('signal')
    ax10.set_ylabel('count')
    ax10.set_title('Histogram')


    ax11 = fig.add_subplot(gs[1, 1])
    ax11.plot(frequency, power)
    ax11.set_xlabel('frequency [cycles/time unit]')
    ax11.set_ylabel('power [dimensionless]')
    ax11.set_title('Lomb-Scargle periodogram')
    if LS_xlim is not None:
        ax11.set_xlim(LS_xlim)
    if true_freq is not None:
        [ax11.axvline(x, color='r',lw=0.5) for x in true_freq]
        if plot_harmonics is True:
            [ax11.axvline(x, color='b',lw=0.5) for x in true_freq/2.]
            [ax11.axvline(x, color='g',lw=0.5) for x in true_freq/3.]
        
    plt.show()
    return 



# multi-periodic signal from Kovacs & Buchler 1988
def KB88(t,a1=1.0,a2=0.6,a3=0.2,v1=0.1,v2=0.048,v3=0.0252):
    w1 = 2*np.pi*v1
    w2 = 2*np.pi*v2
    w3 = 2*np.pi*v3
    
    # length t + 1
    t2 = np.append(t, t[-1] + np.mean(t[1:] - t[0:-1]))
    
    r = a1 * np.cos(w1 * t2) + a2 * np.cos(w2 * t2) + a3 * np.cos(w3 * t2)         + 0.5 * a1 * a3 * (np.cos( (w1 + w3) * t2) + np.cos( (w1 - w3) * t2))         + 0.1 * a2 * a3 * (np.cos( (w2 + w3) * t2) + np.cos( (w2 - w3) * t2))         + 0.1 * a1 * a2 * (np.cos( (w1 + w2) * t2) + np.cos( (w1 - w2) * t2))         + 0.1 * a1**2 * np.cos(2. * w1 * t2)         + 0.1 * a2**2 * np.cos(2. * w2 * t2)
    
    v = (r[1:] - r[0:-1])/np.mean(t[1:] - t[0:-1])
        
    return r[:-1], v


def Rossler_FPs(a,b,c):
    """
    For given Rossler parameters a, b, c, return fixed points of the system
    
    Inputs:
    a, b, c : floats, parameters of Rossler system of equations
    
    Outputs:
    fp1 : np.array, 3D coordinates of first FP
    fp2 : np.array, 3D coordinates of second FP
    
    """
    x1 = (c + np.sqrt(c**2 - 4*a*b))/2.
    y1 = (-c - np.sqrt(c**2 - 4*a*b))/(2.*a)
    z1 = (c + np.sqrt(c**2 - 4*a*b))/(2.*a)
    
    x2 = (c - np.sqrt(c**2 - 4*a*b))/2.
    y2 = (-c + np.sqrt(c**2 - 4*a*b))/(2.*a)
    z2 = (c - np.sqrt(c**2 - 4*a*b))/(2.*a)
    
    fp1 = np.array((x1,y1,z1))
    fp2 = np.array((x2,y2,z2))
    
    return fp1, fp2


def Rossler_vel(t,r):
    """
    For array r = (x, y, z), return array (xdot, ydot, zdot) for the Rossler system. This function
    is of the form expected by scipy's RK45 integrator.
    
    Inputs: 
    t : np.array 
    r : np.array
    
    Outputs:
    rdot : np.array
    
    """
    a=0.2
    b=0.2
    c=4.8
    
    x = r[0]
    y = r[1]
    z = r[2]
    
    v_x = -y - z
    v_y = x + a*y
    v_z = b + z*(x - c)
    
    rdot = np.array((v_x, v_y, v_z))
    
    return rdot


def rotated_Rossler_vel(t,r):
    """
    For array r = (x, y, z), return array (xdot, ydot, zdot) for the  Rossler system. This function
    is of the form expected by scipy's RK45 integrator.
    
    Inputs: 
    t : np.array 
    r : np.array
    
    Outputs:
    rdot : np.array
    
    """
    a=0.2
    b=0.2
    c=4.8
    
    x = r[0]
    y = r[1]
    z = r[2]
    
    # Note: equations (18) of Letellier & Aguirre 2002 are WRONG---these are right.
    #v_x = -0.5*((1 - a + c)*x + (a - 1 + c)*y - (1 + a + c)*z - 2*b) + 0.25*(x + y - z)*(z + y - x)
    #v_y = b + 0.25*(x + y - z)*(z + y - x - 2*c) - x
    #v_z = -x + 0.5*(x*(a - 1) + y*(1 - a) + z*(1 + a))
    
    #or, renormalizing by 1/sqrt(2): (i.e., x' = (1/sqrt(2))*(y+z), etc.
    v_x = 0.5*(x*(a - 1) + y*(1 - a) + z*(1 + a) + np.sqrt(2)*b + (x + y - z)*((np.sqrt(2)/2)*(y + z - x) - c))
    v_y = (1./np.sqrt(2))*b + 0.5*(x + y - z)*((np.sqrt(2)/2)*(z + y - x) - c) - x
    v_z = -x + 0.5*(x*(a - 1) + y*(1 - a) + z*(1 + a))
    
    return np.array((v_x, v_y, v_z))



def Lorenz_FPs(sigma, beta, rho):
    """
    For given Lorenz parameters sigma, beta, rho, return fixed points of the system
    
    Inputs:
    sigma, beta, rho: floats, parameters of Rossler system of equations
    
    Outputs:
    fp1 : np.array, 3D coordinates of first FP
    fp2 : np.array, 3D coordinates of second FP
    
    """
    x1 = np.sqrt( beta * (rho - 1))
    y1 = x1
    z1 = rho - 1
    
    x2 = -x1
    y2 = -y1
    z2 = z1
    
    if rho < sigma*((sigma + beta + 3.)/(sigma - beta - 1.)):
        print("stable")
    else:
        print("unstable")
    
    return np.array((x1,y1,z1)), np.array((x2,y2,z2))


def Lorenz_vel(t,r):
    """
    For array r = (x, y, z), return array (xdot, ydot, zdot) for the  Lorenz system. This function
    is of the form expected by scipy's RK45 integrator.
    
    Inputs: 
    t : np.array 
    r : np.array
    
    Outputs:
    rdot : np.array
    
    """
    sigma=10.
    beta=8./3.
    rho=28.
    
    x = r[0]
    y = r[1]
    z = r[2]
    
    v_x = -sigma*(x-y)
    v_y = x*(rho-z) - y
    v_z = x*y - beta*z
    
    return np.array((v_x, v_y, v_z))


# first, need a way of discretizing the data (X(t), X(t+T)) into 2D bins
def bin2D(timeSeries, tauIdx,plotTitle=None):
    """
    Plot 2D histograms in rectangular and hexagonal bins of delayed timeSeries vs. starting timeSeries
    
    Inputs:
    timeSeries: np.array
    tauIdx : int, delay time in units of time series cadence
    plotTitle : str
    
    Outputs:
    None
    """
    
    x = timeSeries[:-tauIdx]
    y = timeSeries[tauIdx:]
    #print(timeSeries)
    #print(x)
    #print(y)
    fig, axes = plt.subplots(1,2,figsize=(13,6))
    axes[0].hist2d(x,y,bins=(100,100),cmap="Blues")
    axes[1].hexbin(x,y,gridsize=100,cmap="Blues",extent=(np.min(x)-0.01*np.ptp(x),np.max(x)+0.01*np.ptp(x),np.min(y)-0.01*np.ptp(y),np.max(y)+0.01*np.ptp(y)))
    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlabel("x[i]",fontsize=14)
        ax.set_ylabel("x[i + tauIdx]",fontsize=14)
        ax.set_xlim(np.min(x)-0.01*np.ptp(x), np.max(x)+0.01*np.ptp(x))
        ax.set_ylim(np.min(y)-0.01*np.ptp(y), np.max(y)+0.01*np.ptp(y))
    plt.suptitle(plotTitle,y=0.93,fontsize=16)
    
    return


def calc_MI(X,Y,Xbins,Ybins):
    """
    Estimate mutual information of arrays X and Y, divided into pre-specified number of bins Xbins and Ybins
    
    Inputs:
    X : np.array
    Y : np.array
    Xbins : int
    Ybins : int
    
    Outputs:
    MI : float
    """
    c_XY = np.histogram2d(X,Y,[Xbins,Ybins])[0]
    c_X = np.histogram(X,Xbins)[0]
    c_Y = np.histogram(Y,Ybins)[0]

    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)

    MI = H_X + H_Y - H_XY
    return MI

def shan_entropy(c):
    """
    Calculate Shannon entropy -sum ( c * log_2(c) ) for array c
    
    Inputs:
    c : np.array
    
    Outputs:
    H : float, Shannon entropy
    """
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H

def optimal_Nbins(X):
    """
    Implement Freedman-Diaconis rule for choosing optimal bin size / bin number:
    bin width = 2 * interquartile range * N**(-1/3)
    
    Inputs:
    X : np.array
    
    Outputs:
    n_bins : int
    """
    bin_width = 2. * iqr(X) * (len(X)**(-1./3.))
    n_bins = int(np.ptp(X)/bin_width)
    return n_bins


def moving_average(a, n=3) :
    """
    calculate moving average of array a over window size n
    
    Inputs:
    a : np.array
    n : int
    
    Outputs:
    smooth : np.array
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    smooth = ret[n - 1:] / n
    return smooth

def FS86(timeSeries, trialDelayIndices, plot=False):
    """
    Calculate the first local minimum of the mutual information as a function of delay time (in units of the cadence
    of the time series).

    Inputs:
    timeSeries (array-like): the time series on which to calculate the mutual information between pairs of delayed points
    trialDelayIndices (array-like): delay times (in units of time series cadence) to test
    
    Returns:
    MI (array-like, same shape as trialDelayIndices): each entry is the mutual information calculated at the respective
    choice of tau from trialDelayIndices
    firstLocalMinIdx (integer): index into MI of the first local minimum.
    
    """
    MI = np.zeros_like(trialDelayIndices,dtype=float)
    for i,tau in enumerate(trialDelayIndices):
        MI[i] = calc_MI(timeSeries[:-(tau+1)], timeSeries[(tau+1):],Xbins=optimal_Nbins(timeSeries[:-(tau+1)]), Ybins = optimal_Nbins(timeSeries[(tau+1):]))

    smooth_MI = moving_average(MI)
    localMinima = argrelextrema(smooth_MI, np.less)
    firstLocalMinIdx = localMinima[0][0]
    
    #check that the smoothing hasn't moved the index of the local minimum 
    for j in np.arange(-5,5):
        if MI[firstLocalMinIdx + j] < MI[firstLocalMinIdx]:
            firstLocalMinIdx = firstLocalMinIdx + j

    if plot is True:
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        ax.plot(MI,'b-')
        ax.plot(smooth_MI,'g-')
        ax.axvline(x=firstLocalMinIdx,color='r')
        plt.show()
        
        bin2D(timeSeries=timeSeries, tauIdx = firstLocalMinIdx, plotTitle="tauIdx = {0}".format(firstLocalMinIdx))
    
    return MI, firstLocalMinIdx


def delayMatrix(timeSeries, tau, m):
    """
    Organize timeSeries into a delay matrix, where each row is a delay vector of dimension m where the entries are separated by 
    delay time tau (in units of the time series cadence)
    
    Inputs:
    timeSeries : np.array
    tau : int
    m : int
    
    Returns:
    delayMat : np.array, shape ( N - (m-1)*tau , m )
    
    """
    if m==1:
        return timeSeries
    else:
        N = len(timeSeries)
        firstDim = N - (m-1)*tau
        delayMat = np.zeros((firstDim, m),dtype=float)
        for i in range(m):
            delayMat[:,i] = timeSeries[(i*tau): (N - (m - 1 - i)*tau)]
        return delayMat

def nearestNeighborIndices(delayMatrix_m, delayMatrix_mp1):
    """
    Get the nearest neighbor indices  n(i, m) as defined in Cao 1997 equation 1
    
    Inputs:
    delayMatrix_m : delay matrix of a time series in embedding dimension m
    delayMatrix_mp1 : delay matrix of a time series in embedding dimension m + 1
    
    Returns:
    nnI : 1D np.array, length ( N - m*tau )
    """
    if delayMatrix_m.ndim == 1:
        nEntries = len(delayMatrix_m)
        m = 1
    else:
        nEntries, m = np.shape(np.atleast_2d(delayMatrix_m))
    
    nEntries_mp1 = np.shape(delayMatrix_mp1)[0]
    
    nnI = np.zeros((nEntries_mp1),dtype=int)
    
    for i in range(nEntries_mp1):
        nn = (i+1) % nEntries_mp1
        compDist = chebyshev(delayMatrix_m[i], delayMatrix_m[nn])
        for j in range(nEntries_mp1):
            if j == i:
                pass
            else:
                chebyshevDist = chebyshev(delayMatrix_m[i],delayMatrix_m[j])
                if chebyshevDist < compDist:
                    compDist = chebyshevDist
                    nn = j
                    
        nnI[i] = nn
    return nnI


def cao97(timeSeries, tau, mMax):
    """
    Calculate arrays E1(m), E2(m) as defined in Cao 1997 equations 3 and 5, respectively. E1 should saturate if signal is
    coming from an attractor. E2, meanwhile, should always be equal to 1 if the time series is stochastic, regardless of m; if
    there are values of m where E2 != 1, then the time series has "some determinism" in it, i.e. future values depend on past
    values.
    
    Inputs:
    timeSeries : np.array
    tau : int, time delay in units of time series cadence
    mMax : maximum embedding dimension to test
    
    Returns:
    E1 : np.array (1D, len = mMax - 1)
    E2 : np.array (1D, len = mMax - 1)
    
    """
    # mMax - 1 because we're not bothering with m = 1
    E = np.zeros((mMax), dtype=float)
    E1 = np.zeros((mMax - 1),dtype=float)
    #E1[0] = 0.
    Estar = np.zeros((mMax), dtype=float)
    E2 = np.zeros((mMax-1),dtype=float)
    #E2[0] = 0.
    
    # E must range up to mMax+1 because E1, E2 calculations require m + 1
    for m in range(1,mMax+1):
        delayMat_m = delayMatrix(timeSeries, tau, m)
        delayMat_mp1 = delayMatrix(timeSeries, tau, m+1)

        # find indices of nearest neighbors--this is the slowest step so far, scales as n_datapoints^2
        start = time.time()
        nnIndices = nearestNeighborIndices(delayMat_m,delayMat_mp1)
        end = time.time()
        
        print("time taken: {0}".format(end - start))
        
        # calculate a[i, m] and populate E[m]
        a = np.zeros_like(nnIndices, dtype=float)
        for i in range(len(a)):
            a[i] = chebyshev(delayMat_mp1[i], delayMat_mp1[nnIndices[i]])/chebyshev(delayMat_m[i], delayMat_m[nnIndices[i]])
        
        E[m-1] = np.mean(a)
        
        # calculate equation 4 and populate Estar[m]
        b = np.zeros_like(nnIndices, dtype=float)
        for i in range(len(a)):
            b[i] = np.abs(timeSeries[i + m*tau] - timeSeries[nnIndices[i] + m*tau])
        Estar[m-1] = np.mean(b)
    
    #print(E)
    #print(Estar)
    
    for m in range(0,mMax-1):
        E1[m] = E[m+1]/E[m]
        E2[m] = Estar[m+1]/Estar[m]
    #print(E1)
    #print(E2)
    return E1, E2


def localDensity(rArr, delayMat):
    """
    Local density estimation of Kurths & Herzel 1987 equation 5
    
    Inputs:
    rArr : 1D np.array , array of r values (distances between points) to test
    delayMat : delay matrix constructed from time series
    
    Outputs:
    nArr : np.array of shape (N, len(rArr)), where N is the first dimension of the delay matrix
    """
    print("Local density estimation")
    print(np.shape(delayMat))
    N = np.shape(delayMat)[0]

    nArr = np.zeros((N, len(rArr)))
    
    # this is so slow........
    for i in range(N):
        #print(i)
        for j in range(N):
            if j == i:
                pass
            else:
                for rIdx, r in enumerate(rArr):
                    # heaviside implementation that numba isn't mad about...
                    x = r - np.linalg.norm(delayMat[i] - delayMat[j])
                    if x < 0:
                        pass
                    else:
                        nArr[i, rIdx] += 1.
    
    nArr = (1./(N-1.)) * nArr

    return nArr

def Cq(rArr, timeSeries, tau, m):
    """
    Calculate C0, C1, and C2 as defined in Kurths & Herzel 1987.
    
    Inputs:
    rArr : 1D np.array , array of r values (distances between points) to test
    timeSeries : np.array
    tau : int, delay time in units of time series cadence
    m : int, embedding dimension
    
    Outputs:
    C0 : np.array (like rArr)
    C1 : np.array (like rArr)
    C2 : np.array (like rArr)
    """
    delayMat = delayMatrix(timeSeries, tau, m)

    N = np.shape(delayMat)[0]
    nArr = localDensity(rArr, delayMat) # shape ((N, len(rArr))
    C0 = np.zeros_like(rArr) # capacity or fractal dimension
    C1 = np.zeros_like(rArr) # information dimension or pointwise dimension
    C2 = np.zeros_like(rArr) # correlation exponent, same quantity as Grassberger & Procaccia 1983
    
    print("C array calculation")
                    
    for rIdx, r in enumerate(rArr):
        print(rIdx)
        C0sum = (1./N) * np.sum( (1./nArr[:,rIdx]) )
        C0[rIdx] = 1./C0sum
        
        C1prod = np.prod(nArr[:,rIdx])
        C1[rIdx] = C1prod**(1./N)
        
        C2sum = np.sum(nArr[:,rIdx])
        C2[rIdx] = (1./N)*C2sum
    
    return C0, C1, C2

def direct_C2(rArr, timeSeries, tau, m):
    """
    Calculate the correlation integral C2 directly according to Grassberger & Procaccia 1983. This really should
    not be different from the above, so hopefully should help with debugging.
    
    Inputs:
    rArr : 1D np.array , array of r values (distances between points) to test
    timeSeries : np.array
    tau : int, delay time in units of time series cadence
    m : int, embedding dimension
    
    Returns:
    C2 : np.array (like rArr)
    """
    delayMat = delayMatrix(timeSeries, tau, m)

    N = np.shape(delayMat)[0]
    
    C2 = np.zeros_like(rArr)
    
    for i in range(N):
        for j in range(N):
            if j == i:
                pass
            else:
                for rIdx, r in enumerate(rArr):
                    x = r - np.linalg.norm(delayMat[i] - delayMat[j])
                    if x < 0:
                        pass
                    else:
                        C2[rIdx] += 1.
    
    C2 = (1./N**2) * C2
    
    return C2


