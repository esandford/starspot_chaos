# cython: profile=True
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
from astropy.timeseries import LombScargle
#from scipy.integrate import RK45
from scipy.stats import iqr
from scipy.signal import argrelextrema, savgol_filter, correlate
from scipy.spatial.distance import chebyshev
from pynndescent import NNDescent

__all__ = ['plotTimeSeries', 'KB88', 'Rossler_FPs', 'Rossler_vel', 'rotated_Rossler_vel','Lorenz_FPs','Lorenz_vel','bin2D','calc_MI','shan_entropy','optimal_Nbins','moving_average','FS86','estimateQuasiPeriod','delayMatrix','nearestNeighborIndices','cao97','localDensity','Cq','direct_C2']


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
    c=5.7
    
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
    c=5.7
    
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
    sigma, beta, rho: floats, parameters of Lorenz system of equations
    
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

def estimateQuasiPeriod(time, timeSeries, plot=False):
    """
    Estimate the period or cycling time of a quasi-periodic signal.
    Make a first guess = the median interval between successive local maxima, in units of cadence
    Then find the nearest local maximum in mutual information to that guess.
    """
    cadence = np.median(time[1:] - time[0:-1])
    #print(cadence)
    localMinima = argrelextrema(timeSeries, np.less)
    localMaxima = argrelextrema(timeSeries, np.greater)
    
    localMinimaSep = time[localMinima[0][1:]] - time[localMinima[0][:-1]]
    localMaximaSep = time[localMaxima[0][1:]] - time[localMaxima[0][:-1]]

    qp_min = np.median(localMinimaSep)
    qp_max = np.median(localMaximaSep)
    
    #print(qp_min)
    #print(qp_max)
    """
    if plot is True:
        fig, axes = plt.subplots(1,2, figsize=(16,6))
        axes[0].plot(time,timeSeries,'b-')
        '''
        for i in range(len(localMinima[0])):
            axes[0].axvline(x=localMinima[0][i], color = 'b')
        
        for i in range(len(localMaxima[0])):
            axes[0].axvline(x=localMaxima[0][i], color = 'g')
        
        for i in range(int(len(timeSeries)/qp)):
            axes[0].axvline(x=timeSeries[0]+(qp*i), color='r')
        '''
        axes[1].hist(localMinimaSep, color='b', alpha=0.5)
        axes[1].hist(localMaximaSep, color='r', alpha=0.5)
        
        axes[1].axvline(qp_min, color='k')
        
        plt.show()
    """
    
    numTrials = int(0.2*(qp_max/cadence))
    
    MI = np.zeros(2*numTrials)
    trialDelayIndices = np.arange(int(qp_max/cadence) - numTrials, int(qp_max/cadence) + numTrials)
    
    for i,tau in enumerate(trialDelayIndices):
        MI[i] = calc_MI(timeSeries[:-(tau+1)], timeSeries[(tau+1):],Xbins=optimal_Nbins(timeSeries[:-(tau+1)]), Ybins = optimal_Nbins(timeSeries[(tau+1):]))

    qp = trialDelayIndices[np.argmax(MI)]
    #print(qp)
    """
    MI_long = np.zeros(1000)
    for i,tau in enumerate(np.arange(1000)):
        MI_long[i] = calc_MI(timeSeries[:-(tau+1)], timeSeries[(tau+1):],Xbins=optimal_Nbins(timeSeries[:-(tau+1)]), Ybins = optimal_Nbins(timeSeries[(tau+1):]))
    """
    #compute autocorrelation function
    #print(np.shape(timeSeries))
    autocorr = correlate(timeSeries, timeSeries, mode="same")
    autocorr = autocorr[int(len(autocorr)/2):int(len(autocorr)/2)+1000]
    autocorr = autocorr/np.max(autocorr)
    #print(np.shape(autocorr))

    #zeros of autocorrelation fn
    ac_zeros = (np.diff(np.sign(autocorr)) != 0)*1

    if plot is True:
        fig, ax = plt.subplots(1,1, figsize=(8,6))
        #ax.plot(np.arange(1000), MI_long)
        ax.plot(trialDelayIndices, MI)
        ax.plot(np.arange(1000), autocorr)
        ax.axvline(qp,color='r')

        if np.any(ac_zeros):
            for i in range(np.sum(ac_zeros)):
                ax.axvline(np.arange(len(ac_zeros))[ac_zeros==1][i])
        ax.axhline(0)
        plt.show()
    
    return qp



def FS86(time, timeSeries, method="first_or_second_local_min", level_off_criterion=0.05, plot=False):
    """
    Calculate the first local minimum of the mutual information as a function of delay time (in units of the cadence
    of the time series).

    Inputs:
    timeSeries (array-like): the time series on which to calculate the mutual information between pairs of delayed points
    trialDelayIndices (array-like): delay times (in units of time series cadence) to test
    method (str): global_min, 
                  first_local_min,
                  local_min_stop_decreasing, or 
                  level_off_local_min (which requires additional argument level_off_criterion)
    Returns:
    MI (array-like, same shape as trialDelayIndices): each entry is the mutual information calculated at the respective
    choice of tau from trialDelayIndices
    firstLocalMinIdx (integer): index into MI of the first local minimum.
    
    """
    qp = estimateQuasiPeriod(time, timeSeries, plot=False)
    
    trialDelayIndices = np.arange(2*qp)

    MI = np.zeros_like(trialDelayIndices,dtype=float)
    
    for i,tau in enumerate(trialDelayIndices):
        MI[i] = calc_MI(timeSeries[:-(tau+1)], timeSeries[(tau+1):],Xbins=optimal_Nbins(timeSeries[:-(tau+1)]), Ybins = optimal_Nbins(timeSeries[(tau+1):]))

    length = len(timeSeries)
    smoothing_length = int(qp/2)
    if smoothing_length % 2 == 0:
        smoothing_length = smoothing_length - 1
    #if smoothing_length < 11:
    #    smoothing_length = 11

    #print(smoothing_length)
    smooth_MI = savgol_filter(MI, smoothing_length, 2)
    localMinima = argrelextrema(smooth_MI, np.less)

    if method=="first_local_min":
        firstLocalMinIdx = localMinima[0][0]
        to_return = firstLocalMinIdx
        
        #check that the smoothing hasn't moved the index of the local minimum 
        lowIdx = firstLocalMinIdx - smoothing_length
        highIdx = firstLocalMinIdx + smoothing_length

        if lowIdx < 0:
            lowIdx = 0
        if highIdx > len(MI):
            highIdx = len(MI)

        for j in np.arange(lowIdx,highIdx):
            if MI[j] < MI[to_return]:
                to_return = j

    elif method=="first_or_second_local_min":
        #print(localMinima[0])
        first_local_min = localMinima[0][0]
        second_local_min = localMinima[0][1]
        if MI[first_local_min] < MI[second_local_min]:
            first_or_second = first_local_min
        else:
            first_or_second = second_local_min
        #print("first_or_second is {0}".format(first_or_second))
        to_return = first_or_second

        #check that the smoothing hasn't moved the index of the local minimum 
        lowIdx = first_or_second - smoothing_length
        highIdx = first_or_second + smoothing_length
        #print("lowIdx is {0}".format(lowIdx))
        #print("highIdx is {0}".format(highIdx))
        if lowIdx < 0:
            lowIdx = 0
        if highIdx > len(MI):
            highIdx = len(MI)

        for j in np.arange(lowIdx,highIdx):
            if MI[j] < MI[to_return]:
                #print(j)
                to_return = j

    elif method=="global_min":
        to_return = np.argmin(MI)
        
    elif method=="local_min_stop_decreasing":
        try:
            for j in range(len(localMinima[0])-1):
                if MI[localMinima[0][j+1]] > MI[localMinima[0][j]]:
                    break
            local_min_stop_decreasing_idx = localMinima[0][j]

            to_return = local_min_stop_decreasing_idx
            #check that the smoothing hasn't moved the index of the local minimum 
            lowIdx = local_min_stop_decreasing_idx - smoothing_length
            highIdx = local_min_stop_decreasing_idx + smoothing_length

            if lowIdx < 0:
                lowIdx = 0
            if highIdx > len(MI):
                highIdx = len(MI)

            for j in np.arange(lowIdx,highIdx):
                if MI[j] < MI[to_return]:
                    to_return = j
        
        except UnboundLocalError: #what happens if there are no local minima! just return the global
            to_return = np.argmin(MI)

    elif method=="level_off_local_min":
        for j in range(len(localMinima[0])-1):
            if np.abs(MI[localMinima[0][j+1]] - MI[localMinima[0][j]]) < level_off_criterion*MI[localMinima[0][j]]:
                break

        level_off_local_min = localMinima[0][j]
        to_return = level_off_local_min

        #check that the smoothing hasn't moved the index of the local minimum 
        lowIdx = level_off_local_min - smoothing_length
        highIdx = level_off_local_min + smoothing_length

        if lowIdx < 0:
            lowIdx = 0
        if highIdx > len(MI):
            highIdx = len(MI)

        for j in np.arange(lowIdx,highIdx):
            if MI[j] < MI[to_return]:
                to_return = j

    if plot is True:
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        ax.plot(MI,'b-')
        ax.plot(smooth_MI,'g-')
        ax.axvline(x=to_return,color='r')
        plt.show()
            
        bin2D(timeSeries=timeSeries, tauIdx = to_return, plotTitle="tauIdx = {0}".format(to_return))

    return MI, to_return



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

    Returns:
    nnI : 1D np.array, length ( N - m*tau )
    """
    if delayMatrix_m.ndim == 1:
        delayMatrix_m = np.atleast_2d(delayMatrix_m).T

    nEntries_mp1 = np.shape(delayMatrix_mp1)[0]

    index = NNDescent(delayMatrix_m[:nEntries_mp1], n_neighbors=10, metric="chebyshev")    
    nnI = index.neighbor_graph[0]
    nnD = index.neighbor_graph[1] 
    return nnI, nnD
'''
cpdef nearestNeighborIndices(delayMatrix_m, delayMatrix_mp1):
    """
    Get the nearest neighbor indices  n(i, m) as defined in Cao 1997 equation 1
    
    Inputs:
    delayMatrix_m : delay matrix of a time series in embedding dimension m
    delayMatrix_mp1 : delay matrix of a time series in embedding dimension m + 1
    
    Returns:
    nnI : 1D np.array, length ( N - m*tau )
    """
    cdef:
        int nEntries, nEntries_mp1, m, i, j, nn
        double compDist, chebyshevDist

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

'''
cpdef cao97(timeSeries, int tau, int mMax):
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
    cdef:
        int m, i
        double start, end

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
        nEntries_mp1 = np.shape(delayMat_mp1)[0]
    
        # find indices of nearest neighbors--this is the slowest step so far, scales as n_datapoints^2
        start = time.time()
        nnIndices, nnDistances = nearestNeighborIndices(delayMat_m, delayMat_mp1)
        #print(np.shape(nnIndices))
        #print(np.shape(nnDistances))
        end = time.time()
        
        #print("time taken: {0}".format(end - start))
        
        #print(np.shape(delayMat_m))
        #print(np.shape(delayMat_mp1))
        #print(nEntries_mp1)
        # calculate a[i, m] and populate E[m]
        a = np.zeros(nEntries_mp1, dtype=float)
        for i in range(nEntries_mp1):
        #for i in range(10):
            #print(nnIndices[i])
            #print(nnDistances[i])
            j = 0
            #print("j is 0")
            numerator = chebyshev(delayMat_mp1[i], delayMat_mp1[nnIndices[i,j]])
            denominator = chebyshev(delayMat_m[i], delayMat_m[nnIndices[i,j]])
            #print(denominator)

            #if the distance is zero, take the next nearest neighbor
            #if the "nearest neighbor" is the point itself, take the next nearest neighbor
            # (there can be multiple points with distance 0, and PyNNDescent does not
            # necessarily sort them so that the point itself is the first entry)
            while denominator == 0. or nnIndices[i,j] == i:
                j+=1
                #print("j is {0}".format(j))
                denominator = chebyshev(delayMat_m[i], delayMat_m[nnIndices[i,j]])
                #print(denominator)
            
            #print("numerator ingredients")
            #print(delayMat_mp1[i])
            #print(delayMat_mp1[nnIndices[i,j]])
            #print("numerator")
            #print(chebyshev(delayMat_mp1[i], delayMat_mp1[nnIndices[i,j]]))
            #print("denominator ingredients")
            #print(delayMat_m[i])
            #print(delayMat_m[nnIndices[i,j]])
            #print("denominator")
            #print(chebyshev(delayMat_m[i], delayMat_m[nnIndices[i,j]]))
            a[i] = chebyshev(delayMat_mp1[i], delayMat_mp1[nnIndices[i,j]])/denominator #chebyshev(delayMat_m[i], delayMat_m[nnIndices[i,j]])
            
            #print("a[i]")
            #print(a[i])
        #print(a)
        #fig, ax = plt.subplots(1,1,figsize=(4,3))
        #ax.hist(a)
        #plt.show()
        #print(np.mean(a))
        E[m-1] = np.mean(a)
        
        # calculate equation 4 and populate Estar[m]
        b = np.zeros(nEntries_mp1, dtype=float)
        for i in range(nEntries_mp1):
            b[i] = np.abs(timeSeries[i + m*tau] - timeSeries[nnIndices[i,j] + m*tau])
        Estar[m-1] = np.mean(b)
    
    #print(E)
    #print(Estar)
    
    for m in range(0,mMax-1):
        E1[m] = E[m+1]/E[m]
        E2[m] = Estar[m+1]/Estar[m]
    #print(E1)
    #print(E2)
    return E1, E2


def localDensity(rArr, delayMat, nNeighbors=200, divprob=1.0, pdm=1.5):
    """
    Local density estimation of Kurths & Herzel 1987 equation 5
    
    Inputs:
    rArr : 1D np.array , array of r values (distances between points) to test
    delayMat : delay matrix constructed from time series
    nNeighbors : int, number of nearest neighbors for PyNNDescent to index
    
    Outputs:
    nArr : np.array of shape (N, len(rArr)), where N is the first dimension of the delay matrix
    """
    #print("Local density estimation")
    #print(np.shape(delayMat))
    
    N = np.shape(delayMat)[0]

    if delayMat.ndim == 1:
        delayMat = np.atleast_2d(delayMat).T

    index = NNDescent(delayMat, metric="euclidean", n_neighbors=nNeighbors, diversify_prob=divprob, pruning_degree_multiplier=pdm)    
    neighborDistances = index.neighbor_graph[1]
    
    nArr = np.zeros((N, len(rArr)))

    for i in range(N):
        for rIdx, r in enumerate(rArr):
            withinSphere = len(neighborDistances[i][neighborDistances[i] < r])
            if int(withinSphere) == nNeighbors:
                #print("Warning! PyNNDescent hasn't indexed enough neighbors for accurate calculation at r = {0}".format(r))
                nArr[i, rIdx] = np.NaN
                #break
            else:
                nArr[i, rIdx] = withinSphere

    '''
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
    '''
    nArr = (1./(N)) * nArr

    return nArr, neighborDistances

def Cq(rArr, timeSeries, tau, m, divprob=1.0, pdm=1.5):
    """
    Calculate C0, C1, and C2 as defined in Kurths & Herzel 1987.
    
    Inputs:
    rArr : 1D np.array , array of r values (distances between points) to test
    timeSeries : np.array
    tau : int, delay time in units of time series cadence
    m : int, embedding dimension
    nNeighbors : int, number of nearest neighbors for PyNNDescent to index
    divprob : float between 0.0 and 1.1, the diversify_prob parameter for PyNNDescent
    pdm : float, the pruning_degree_multiplier parameter for PyNNDescent

    Outputs:
    C0 : np.array (like rArr)
    C1 : np.array (like rArr)
    C2 : np.array (like rArr)
    """
    
    delayMat = delayMatrix(timeSeries, tau, m)

    N = np.shape(delayMat)[0]

    # cap nNeighbors to eliminate memory pressure problem
    if N <= 10000:
        nNeighbors = N
    else:
        nNeighbors = 5000

    nArr, neighborDistances = localDensity(rArr, delayMat, nNeighbors=nNeighbors, divprob=divprob, pdm=pdm) # shape ((N, len(rArr))
    C0 = np.zeros_like(rArr) # capacity or fractal dimension
    C1 = np.zeros_like(rArr) # information dimension or pointwise dimension
    C2 = np.zeros_like(rArr) # correlation exponent, same quantity as Grassberger & Procaccia 1983
                    
    for rIdx, r in enumerate(rArr):
        #C0 is the harmonic mean of nArr, per Kurths & Herzel 1987
        C0sum = (1./N) * np.sum( (1./nArr[:,rIdx]) )
        C0[rIdx] = 1./C0sum
        
        #C1 is the geometric mean of nArr
        C1sum = np.sum(np.log(nArr[:,rIdx]))
        C1[rIdx] = np.exp((1./N)*C1sum)

        #C2 is the arithmetic mean of nArr
        C2sum = np.sum(nArr[:,rIdx])
        C2[rIdx] = (1./N)*C2sum
    
    return C0, C1, C2, nArr

def direct_C2(rArr, timeSeries, tau, m, nNeighbors):
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
    if delayMat.ndim == 1:
        delayMat = np.atleast_2d(delayMat).T

    index = NNDescent(delayMat, metric="euclidean", n_neighbors=nNeighbors)    
    neighborDistances = index.neighbor_graph[1]

    C2 = np.zeros_like(rArr)
    
    for i in range(N):
        for rIdx, r in enumerate(rArr):
            withinSphere = len(neighborDistances[i][neighborDistances[i] < r])
            if int(withinSphere) == nNeighbors:
                #print("Warning! PyNNDescent hasn't indexed enough neighbors for accurate calculation at r = {0}".format(r))
                C2[rIdx] = np.NaN
                #break
            else:
                C2[rIdx] += withinSphere
    
    C2 = (1./N**2) * C2
    
    return C2

def brokenLine(x, x_break, b, m):
    """
    Broken line model:
        y( x <= x_break) = b
        y( x >  x_break) = m*(x - x_break) + b
    """
    y = np.zeros_like(x)
    y[x <= x_break] = b
    y[x > x_break] = m*(x[x > x_break] - x_break) + b
    return y

def brokenLine2(x, x1, x2, b, m1, m2):
    """
    more complicated broken line model
        y( x <= x1) = b
        y( x1 < x <= x2) = m1*(x - x1) + b
        y( x > x2) = m2*(x - x2) + m1*(x2-x1) + b 
    """
    y = np.zeros_like(x)
    y[x <= x1] = b
    y[ (x > x1) & (x <= x2) ] = m1*(x[ (x > x1) & (x <= x2) ] - x1) + b
    y[x > x2] = m2*(x[x > x2] - x2) + m1*(x2-x1) + b
    return y

def brokenLine3(x, x1, x2, b, m1, m2, m3):
    """
    three-segment broken line model; no flat part
    """
    y = np.zeros_like(x)
    y[x <= x1] = m1*x[x <= x1] + b
    y[ (x > x1) & (x <= x2) ] = m2*(x[ (x > x1) & (x <= x2) ] - x1) + m1*x1 + b
    y[x > x2] = m3*(x[x > x2] - x2) + m2*(x2-x1) + m1*x1 + b
    return y

def normal_equation(x, y, yerr, order=2):
    """
    Solve the normal equation B = (X.T*C.inv*X).inv*X.T*C.inv*Y
    Inputs:
    x = matrix of x values
    y = vector of y values
    yerr = vector of yerr values
    order = integer polynomial order 
    
    Outputs:
    b = vector of model parameters that minimizes chi^2
    Bunc = covariance matrix of uncertainties on model parameters. diagonal entries are sigma**2 of the individual parameters, and off-diagonals are covariances.
    """
    
    X = np.vander(x, order)
    
    XTX = np.dot(X.T, X/yerr[:, None]**2)
    
    b = np.linalg.solve(XTX, np.dot(X.T, y/yerr**2))
    Bunc = np.linalg.solve(XTX, np.identity(order))
    return b, Bunc

# fit line to linear regime of data only
def fitLinearRegime(rArr, nArr, C):
    N = len(nArr[:,0])
    medians = np.percentile(nArr, 50, axis=0)
    
    # uncertainty on C at each value of r will be defined as the 84th-50th percentile or the 50th-16th percentile, whichever is larger
    Cunc_up = np.percentile(nArr, 84, axis=0) - np.percentile(nArr, 50, axis=0)
    Cunc_down =  np.percentile(nArr, 50, axis=0) - np.percentile(nArr, 16, axis=0)
    
    #element-wise maximum
    Cunc = np.maximum(Cunc_down,Cunc_up)
    
    # transform variables to get uncertainty in log10 space
    Cunc_log10 = (Cunc/(np.log(10) * C))
    
    # exclude values of r where the median of n(r) is <= 10./N . Cutoff is a little arbitrary but the idea is that these points don't have enough neighbors.
    enoughNeighborsIdxs = np.arange(len(rArr))[medians > 10./N]
    firstGood = enoughNeighborsIdxs[0]
    
    # exclude values of r where any n(r) are NaN. The time series is not long enough to populate all the neighbors of the points.
    anyNans = [np.any(~np.isfinite(nArr[:,i])) for i in range(len(rArr))]
    anyNans = np.array(anyNans)
    nansIdxs = np.arange(len(rArr))[anyNans]
    lastGood = nansIdxs[0]
    
    """
    fig, axes = plt.subplots(1,2,figsize=(16,6))
    axes[0].errorbar(rArr[firstGood:lastGood], C[firstGood:lastGood], Cunc[firstGood:lastGood])
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    
    axes[1].errorbar(np.log10(rArr[firstGood:lastGood]), np.log10(C[firstGood:lastGood]), Cunc_log10[firstGood:lastGood])
    
    axes[0].set_xlim(10**-1.5, 10**1.5)
    axes[1].set_xlim(-1.5,1.5)
    axes[0].set_ylim(10**-4, 10**0)
    axes[1].set_ylim(-4,0)

    plt.show()
    """
    
    params, params_unc = normal_equation(x=np.log10(rArr)[firstGood:lastGood], y=np.log10(C)[firstGood:lastGood], yerr=Cunc_log10[firstGood:lastGood], order=2)
    return params, params_unc
