def contains_digits(s):
    return any(char.isdigit() for char in s)


import time
from datetime import datetime
from segmentedLeastSq import *
from collections import Counter
import pylab as plt
from scipy.special import zeta
from scipy.stats import *
from scipy import stats 
from collections import Counter


def customaxis(ax, c_left='k', c_bottom='k', c_right='none', c_top='none',
               lw=3, size=12, pad=8):
    """
    Format the plots
    """

    for c_spine, spine in zip([c_left, c_bottom, c_right, c_top],
                              ['left', 'bottom', 'right', 'top']):
        if c_spine != 'none':
            ax.spines[spine].set_color(c_spine)
            ax.spines[spine].set_linewidth(lw)
        else:
            ax.spines[spine].set_color('none')
    if (c_bottom == 'none') & (c_top == 'none'): # no bottom and no top
        ax.xaxis.set_ticks_position('none')
    elif (c_bottom != 'none') & (c_top != 'none'): # bottom and top
        ax.tick_params(axis='x', direction='out', width=lw, length=7,
                      color=c_bottom, labelsize=size, pad=pad)
    elif (c_bottom != 'none') & (c_top == 'none'): # bottom but not top
        ax.xaxis.set_ticks_position('bottom')
        ax.tick_params(axis='x', direction='out', width=lw, length=7,
                       color=c_bottom, labelsize=size, pad=pad)
    elif (c_bottom == 'none') & (c_top != 'none'): # no bottom but top
        ax.xaxis.set_ticks_position('top')
        ax.tick_params(axis='x', direction='out', width=lw, length=7,
                       color=c_top, labelsize=size, pad=pad)
    if (c_left == 'none') & (c_right == 'none'): # no left and no right
        ax.yaxis.set_ticks_position('none')
    elif (c_left != 'none') & (c_right != 'none'): # left and right
        ax.tick_params(axis='y', direction='out', width=lw, length=7,
                       color=c_left, labelsize=size, pad=pad)
    elif (c_left != 'none') & (c_right == 'none'): # left but not right
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='y', direction='out', width=lw, length=7,
                       color=c_left, labelsize=size, pad=pad)
    elif (c_left == 'none') & (c_right != 'none'): # no left but right
        ax.yaxis.set_ticks_position('right')
        ax.tick_params(axis='y', direction='out', width=lw, length=7,
                       color=c_right, labelsize=size, pad=pad)



def get_CCDF(sizeEvent):
    """
    Get CCDF
    """
    #Frequency
    a = Counter(sizeEvent)
    bins,freqs = zip(*sorted(zip(a.keys(),a.values())))
    freqs = np.array(freqs)
    cdf1 = np.concatenate([[0],np.cumsum(freqs[:-1]/np.sum(freqs))])
    yCCDF = ((1.-cdf1))
    return [bins,yCCDF,freqs]



def findXmin(sizeEvent):
    """
    Find the xmin that gives the best power-law fit
    """
    best_chi = -1
    i = 0

    lenS = len(np.unique(sizeEvent))
    a = 1
    for xmin in sorted(np.unique(sizeEvent))[:-10]:
        i+=1
        sizeEvent2 = sizeEvent[sizeEvent>=xmin]
        [xCCDF,yCCDF,PDF] = get_CCDF(sizeEvent2)
        [chi,alpha,Theoretical_CCDF] = get_alpha(xCCDF,yCCDF,xmin,len(sizeEvent2))
        if chi > best_chi:
            best_chi = chi
            a = xmin

    return a

def get_alpha(xCCDF,yCCDF,xmin,N):
    """
    Fit power-law distribution
    """
    
    bins = xCCDF
    n = len(xCCDF)
    a = 0
    best_chi = 50000

    for alpha in np.linspace(1.5,6.,5001):
        Theoretical_CCDF = (zeta(alpha, bins) / zeta(alpha, xmin))

        chi  = np.sum((yCCDF*N-Theoretical_CCDF*N)**2/(Theoretical_CCDF*N))

        if chi < best_chi:
            best_chi = chi
            a = alpha

    best_chi =  stats.chi2.cdf(best_chi,n-1)
    alpha=a
    Theoretical_CCDF = (zeta(alpha, bins) / zeta(alpha, xmin))


    return [1-best_chi,alpha,Theoretical_CCDF]




def plotGlobalPL(sizeEvent,xmin=1,Theo = 0, alpa = 0,ax=None,color=(73./256, 142./256, 204./256)):
    """
    Plot the powerlaw
    """

    if not ax:
        fig = plt.figure()
        ax = plt.subplot(1,1,1)

    [xCCDFAll,yCCDFAll,PDFAll] = get_CCDF(sizeEvent)
    sizeEventAll = np.copy(sizeEvent)
    sizeEvent = np.sort(sizeEvent[sizeEvent>=xmin])
    [xCCDF,yCCDF,PDF] = get_CCDF(sizeEvent)

    plt.hold(True)

    ax.plot((xCCDF),(yCCDF),'.-',color=color,linewidth=2,markersize=15)


    if alpa == 0:
        [best_chi,alpha,Theoretical_CCDF] = get_alpha(xCCDF,yCCDF,xmin,len(sizeEvent))
    else:
        alpha = alpa
        Theoretical_CCDF = Theo
        best_chi = 0

    print("Best chi-sqare value for powerlaw: ", best_chi, " Degree Freedom: ",len(Theoretical_CCDF)-1, alpha)

    if xmin == 1:
        x = np.log10(xCCDF)
        y = np.log10(yCCDF)
        x = x[:,np.newaxis]
        a, intercept, _, _ = np.linalg.lstsq(x, y)

        #plot(np.log10(xCCDF),np.log10(xCCDF)*a,color='red')
        slope1 = a
    else:
        slope1, intercept, r_value, p_value, std_err = linregress(np.log10(xCCDF),np.log10(yCCDF))
        #plot(np.log10(xCCDF),intercept+np.log10(xCCDF)*slope1,color='red')

    ax.plot((xCCDF),(Theoretical_CCDF[-len(xCCDF):]),'-.',color='black',linewidth=2)


    plt.legend(['Data',''.join(['Best slope = ', str(alpha)]),''.join(['paramLR = ', str(-slope1+1)]),],prop={'size':11},loc=3,frameon = False)
    print( ['xmin = ',str(xmin),'Alpha = ', str(alpha)])

    plt.xlabel(r'Severity attack (s)')
    plt.ylabel(r'$P(X>s)$')
    plt.xlim([1,1.1*np.max(xCCDF)])

    #Get limits from lognormal
    #[mu,sigma,loc,best_chi,Theoretical_CCDF] = getMu(xCCDFAll,yCCDFAll,sizeEventAll,PDFAll)
           #ylim([np.min(np.concatenate([yCCDFAll,Theoretical_CCDF]))/1.5,0.2+np.max(np.concatenate([yCCDFAll,Theoretical_CCDF]))])
    plt.xscale('log')
    plt.yscale('log')

def plotGlobalLN(sizeEvent,figname=None,ax=None):

    if not ax:
        fig = plt.figure()
        ax = plt.subplot(1,1,1)

    sizeEvent = sizeEvent[np.isfinite(sizeEvent)]
    [xCCDF,yCCDF,PDF] = get_CCDF(sizeEvent)

    plt.hold(True)

    plt.plot((xCCDF),(yCCDF),'.-',color=(73./256, 142./256, 204./256),linewidth=2,markersize=10)


    [mu,sigma,loc,best_chi,Theoretical_CCDF] = getMu(xCCDF,yCCDF,sizeEvent,PDF)

    print("Best chi-sqare value for lognormal: ", best_chi, " Degree Freedom: ",len(Theoretical_CCDF)-2)
    #plt.plot((xCCDF),(Theoretical_CCDF),'--',color='black',linewidth=2)
    #Tho2 = 1-lognorm.cdf(xCCDF,1,0,5)
    #plt.legend(['Data',''.join(['Mu = ', str(np.round(np.log(mu),2)), ', Sigma = ', str(np.round(sigma,2))])],prop={'size':12},loc=3)

    print(''.join(['Mu = ',str(np.round(np.log(mu),2)),'Sigma = ', str(np.round(sigma,2))]))
    plt.xlabel(r'Severity of attack')
    plt.ylabel(r'$P(X>s)$')
    plt.xlim([1,1.05*np.max(xCCDF)])
    plt.ylim([np.min([yCCDF,Theoretical_CCDF])/1.5,0.2+np.max([yCCDF,Theoretical_CCDF])])
    plt.xscale('log')
    plt.yscale('log')
    return np.log(mu), sigma


def getMu(xCCDF,yCCDF,sizeEvent,PDF):
    from scipy.stats import lognorm
    from scipy.special import erf
    bins = np.sort(sizeEvent)
    best_chi = 50000
    N = len(sizeEvent)
    n = len(xCCDF)


    #MLE estimators
    mu = 1./N*np.sum(np.log(bins))
    sigma = np.sqrt(1./N*np.sum((np.log(bins)-mu)**2))
    scale = np.exp(mu)
    shape = sigma
    print( [shape,scale])
    bestb = 1
    bestc = 1
    bestd = 1
    loc = 0
    #[add,loc,bdd] = lognorm.fit(bins)

    d = 1

    Theoretical_CDF = lognorm.cdf(xCCDF,bestb*shape,bestd*loc,bestc*scale)

    Theoretical_CCDF = 1- Theoretical_CDF
    print( bestb,bestc,bestd)

    return [(bestc*scale),bestb*shape,bestd*loc,1-best_chi,Theoretical_CCDF]


#move to helpers
def studyTweets_v2(time=0):
    import os
    import pylab as plt
    import numpy as np
    from datetime import datetime
    import pandas as pd

    listFiles = os.listdir("./data/datav2/")

    d = dict()

    for fileT in listFiles:
        if ".tgz" in fileT:
            name = fileT[:-10]
            name = "20" + name[-2:] +"-"+ name[3:5] + "-" + name[0:2]
        elif ".gz" in fileT:
            name = fileT[:-15]
        else: name = 0

        if name:
            with open("./data/datav2/"+fileT) as f:
                t = np.asarray([int(_.strip()) for _ in f.readlines()])
                if len(t) > 0:
                    if d.get(name) is not None:
                        d[name] += t
                    else:
                        d[name] = t


    dates = []
    tweetsAll = []
    tweetsSS = []
    tweetsMS = []
    tweetsMM = []
    tweetsS = []
    for name in d:
        if time:
            if int(name[0:4]) > time:
                dates.append(datetime (year=int(name[0:4]),month=int(name[5:7]),day=int(name[-2:])))

                tweetsAll.append(float(d[name][0]))
                tweetsSS.append(float(d[name][1]))
                tweetsMS.append(float(d[name][2]))
                tweetsMM.append(float(d[name][3]))
                tweetsS.append(float(d[name][4]))
        else:
            dates.append(datetime (year=int(name[0:4]),month=int(name[5:7]),day=int(name[-2:])))

            tweetsAll.append(float(d[name][0]))
            tweetsSS.append(float(d[name][1]))
            tweetsMS.append(float(d[name][2]))
            tweetsMM.append(float(d[name][3]))
            tweetsS.append(float(d[name][4]))


    tweetsAll = np.asarray([tweet for (date,tweet) in sorted(zip(dates,tweetsAll))])
    tweetsSS = np.asarray([tweet for (date,tweet) in sorted(zip(dates,tweetsSS))])
    tweetsMS = np.asarray([tweet for (date,tweet) in sorted(zip(dates,tweetsMS))])
    tweetsMM = np.asarray([tweet for (date,tweet) in sorted(zip(dates,tweetsMM))])
    tweetsS = np.asarray([tweet for (date,tweet) in sorted(zip(dates,tweetsS))])
    datesFinal = sorted(dates)
    print(np.sum(tweetsAll))
    print(np.sum(tweetsSS))
    print(np.sum(tweetsMM))
    print(np.sum(tweetsMS))
    print(np.sum(tweetsS))
    return [datesFinal,tweetsS,tweetsSS,tweetsAll,tweetsMS,tweetsMM]
