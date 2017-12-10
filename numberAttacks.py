from __future__ import print_function,division



def contains_digits(s):
    return any(char.isdigit() for char in s)


import time
from datetime import datetime
from segmentedLeastSq import *
import matplotlib
font = {'family':'sans-serif','sans-serif':['Helvetica'],'size' : 12}

matplotlib.rc('font', **font)




from pylab import diff,show,figure,hold,subplot,plot,scatter,legend,vlines,ylim,axhline,errorbar,xlabel,ylabel,xscale,yscale,xlim,title
import numpy as np
#import powerlaw
from scipy.stats import linregress
from smooth import smooth
from scipy import stats

def customaxis(ax, c_left='k', c_bottom='k', c_right='none', c_top='none',
               lw=3, size=12, pad=8):

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

def getDataIndividual(fname):
    f1 = open(fname)
    #Add data to array
    z2=  datetime(1932,1,1)
    z1 = datetime(2015,1,1)

    j = []

    for line in f1:
        a =line.split(',')
        if a:

            if contains_digits(a[0]) and len(a)>3:

                dat = datetime.strptime(a[0],"%d %m %Y")


                if  dat< z1 and dat>z2:
                    j.append(dat)

    sizeEvent = []
    d = 1
    ddd =[]
    count = 0


    for a in range(1,len(j)):
        if j[a] == j[a-1]:
            d +=1
        else:
            count +=1
            sizeEvent.append(d)
            ddd.append(j[a])
            d = 1
    if d>1:
        sizeEvent.append(d)
        ddd.append(j[a])


    a = [n.total_seconds()/60/60/24 for n in diff(j)]
    daysBetweenKills = np.asarray(a,dtype='int')
    daysBetweenAttacks = daysBetweenKills[daysBetweenKills>0]

    return [ddd,np.asarray(sizeEvent),daysBetweenAttacks]

def getDataDays(fname):
    f1 = open(fname)
    #Add data to array
    z2=  datetime(2000,1,1)
    z1 = datetime(2016,1,1)

    j = []
    sizeEvent = []
    if 'SATP' in fname:
        for line in f1:
            a =line.split(',')

            if a:
                if contains_digits(a[0]) and len(a)>3 and int(a[3])>0:
                    dat = datetime.strptime(a[0],"%d %m %Y")
                    if  dat< z1 and dat>z2:
                        j.append(dat)
                        sizeEvent.append(int(a[3]))
    elif 'YemenNATSEC' in fname:
        for line in f1:
            a =line.split(',')
            if a:

                if contains_digits(a[0]) and len(a)>3 and int(eval(a[3])/2)>0:# and int(eval(a[2])/2)<35:
                    dat = datetime.strptime(a[0],"%d %m %Y")
                    if  dat< z1 and dat>z2:
                        j.append(dat)
                        sizeEvent.append(int(eval(a[3])/2))

    elif 'PakNATSEC' in fname:
        for line in f1:
            a =line.split(',')
            if a:

                if contains_digits(a[0]) and len(a)>3 and int(eval(a[3])/2)>0 :# and int(eval(a[2])/2)<35: ' and 'South Waziristan' in a[1]
                    dat = datetime.strptime(a[0],"%d %m %Y")
                    if  dat< z1 and dat>z2:
                        j.append(dat)
                        sizeEvent.append(int(eval(a[3])/2))

    ddd = j


    a = [n.total_seconds()/60/60/24 for n in diff(j)]
    daysBetweenKills = np.asarray(a)

    daysBetweenKills[daysBetweenKills==0] += 0.1

    #Separate attacks in the same day
    while(np.any(np.diff(daysBetweenKills)==0)):
        daysBetweenKills[np.concatenate([[False],np.diff(daysBetweenKills)==0])] += 0.1


    daysBetweenAttacks = daysBetweenKills

    return [ddd,np.asarray(sizeEvent),np.asarray(daysBetweenAttacks)]




def get_CCDF(mat):
    #Frequency

    mat = np.sort(mat)
    count = 1
    z = 0
    mat2 = np.empty(5000)
    for i in range(1,len(mat)):
        if mat[i] == mat[i-1]:
            count += 1
        else:
            mat2[z]= count
            count = 1
            z +=1
    mat2[z]= count
    mat2 = mat2[:z+1]

    cdf1 = np.concatenate([[0],np.cumsum(mat2[:-1]/np.sum(mat2))])

    yCCDF = ((1.-cdf1))

    xCCDF = (np.unique(mat))
    return [xCCDF,yCCDF,np.asarray(mat2)]



def get_alpha(xCCDF,yCCDF,xmin,N):
    from scipy.special import zeta
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

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')



def plotEvolutionAndAlpha(ddd,sizeEvent,dates,daysBetweenAttacks = [],xmin=1,ax1=0,suicide=[]):
    """
    figure(1,facecolor='white')

    ind = np.cumsum(np.diff(dates))-(np.diff(dates)/2)
    alp = np.zeros(len(ind))
    alpE = np.zeros(len(ind))

    axis1 = np.ceil(np.sqrt(len(dates))).astype(int)
    axis2 = np.ceil(np.sqrt(len(dates))).astype(int)

    for d in range(1,len(dates)):
        sizeEvent2 = np.sort(np.asarray(sizeEvent)[dates[d-1]:dates[d]])
        sizeEvent2 = sizeEvent2[sizeEvent2>=xmin]
        [xCCDF,yCCDF,PDF] = get_CCDF(sizeEvent2)
        [best_chi,alp[d-1],Theoretical_CCDF] = get_alpha(xCCDF,yCCDF,xmin,len(sizeEvent2))
        alpE[d-1]=(alp[d-1]-1)/np.sqrt(len(sizeEvent2))
        subplot(axis1,axis2,d)
        hold(True)
        plot(np.log10(xCCDF),np.log10(yCCDF),'blue',marker= 'o')
        plot(np.log10(xCCDF),np.log10(Theoretical_CCDF),'--',color='black')

    sizeEvent2 = sizeEvent[sizeEvent>=xmin]
    [xCCDF,yCCDF,PDF] = get_CCDF(np.sort(sizeEvent2))
    [best_chi,a,Theoretical_CCDF] = get_alpha(xCCDF,yCCDF,xmin,len(sizeEvent2))

    subplot(axis1,axis2,len(dates))
    hold(True)
    plot(np.log10(xCCDF),np.log10(yCCDF),'blue',marker= 'o')
    plot(np.log10(xCCDF),np.log10(Theoretical_CCDF),'--',color='black')
    """

    #fig =figure(2,)
    import matplotlib.pyplot as plt
    if ax1 == 0:
        fig, ax1 = plt.subplots(facecolor='white')
    #ax1 = fig.set_size_inches(6*0.9,3.7*0.9*0.906)


    ax1.vlines(ddd,np.zeros(len(sizeEvent)),sizeEvent,'orange',linewidth=1.5)
    #ax1.vlines(np.asarray(ddd)[dates],np.zeros(len(dates)),np.zeros(len(dates))+max(sizeEvent))
    print( suicide)
    if len(suicide) >0:
        print( 'sui')
        ax1.plot(ddd,suicide,'o',color='k')

    #ax1.set_ylim([0,max(sizeEvent)])
    #ax1.legend(['Size of attack'], 2)
    #t = 21
    #plot(ddd, 4*smooth(sizeEvent,t*2+1)[t:-t],"orange",linewidth=2)
    ax2 = ax1.twinx()

    if len(daysBetweenAttacks) == 0:
        t = 50
        ran = np.arange(0,len(sizeEvent)-t,3)
        alp2 = np.zeros(len(ran))
        alp3 = np.zeros(len(ran))
        count = 0
        for i in ran:
            sizeEvent2 = sizeEvent[i:i+t]
            [xCCDF,yCCDF,PDF] = get_CCDF(sizeEvent2)
            [best_chi,alp2[count],Theoretical_CCDF] = get_alpha(xCCDF,yCCDF,xmin,len(sizeEvent2))
            #print i,alp2[count] ,best_chi
            if best_chi <0.7: alp2[count] = 0
            count +=1
        n = np.asarray(ddd)[t+ran]

        ax2.plot(n[alp2>0],-alp2[alp2>0],'.-',color=(73./256, 142./256, 204./256),linewidth=2)


        #ax2.legend('Alpha', bbox_to_anchor=(1, 0.5))

        #errorbar(np.asarray(ddd)[ind],alp,yerr=alpE,fmt='bo')
        ax2.set_ylabel('Learning rate')
        ax1.set_ylabel('Casualties')
        #ax1.set_xlabel('Time')

        axhline(y=2.5,color=(73./256, 142./256, 204./256))
        axhline(y=a,color=(73./256, 142./256, 204./256))
    sizeEvent = np.asarray(sizeEvent)
    if len(daysBetweenAttacks) != 0:
        for t in [50]:
            ran = np.arange(0,len(daysBetweenAttacks)-t,1)
            alp2 = np.zeros(len(ran))
            days = []
            days2 = []
            count = 0
            for i in ran:
                aNT = (daysBetweenAttacks[i:i+t])
                x = np.log10(np.arange(1,len(aNT)+1))
                y = np.log10((aNT))
                #plot(x,y,'o-')
                #show()
                slope, intercept, r_value, p_value, std_err = linregress(x,y)
                #print i, slope, r_value
                #if r_value**2 > 0.04:
                    #alp2[count] = slope
                    #days.append(ddd[int((2.*i+t)/2.)])
                    #days2.append(ddd[int(i+t*i/max(ran))])
                #else: days.append(0); days2.append(0)
                z= 1./t*np.sum(np.log(np.sort(sizeEvent[i:i+t])))
                alp2[count] = np.sqrt(1./t*np.sum((np.log(np.sort(sizeEvent[i:i+t]))-z)**2))#


                days.append(ddd[int((2.*i+t)/2.)])
                count+=1

                #print i, slope,r_value
            #print t, alp2[alp2!=0]
            n = np.asarray(ddd)
            #ax2.plot(n[np.linspace(0,len(n)-1,(np.sum(alp2!=0))).astype(int)],-alp2[alp2!=0],'.-',color=(73./256, 142./256, 204./256),linewidth=2,markersize=10)
            #n = np.asarray(ddd)[t+ran]

            #ax2.plot(n[alp2!=0],-alp2[alp2!=0],'.-',color=(73./256, 142./256, 204./256),linewidth=2,markersize=10)

            days = np.asarray(days)
            #print days[alp2!=0]
            ax2.plot(days[alp2!=0],-alp2[alp2!=0],'.-',color=(73./256, 142./256, 204./256),linewidth=2,markersize=10)

            #ax2.plot(n[alp2!=0],smooth(alp2[alp2!=0],7[3:-3],'o-',linewidth=2)


        ax2.set_ylabel('Learning rate',rotation=270)
        ax1.set_ylabel('Casualties')
        #ax1.set_xlabel('Time')

        #ax2.legend(['Learning rate'],3)

        axhline(y=0,color=(73./256, 142./256, 204./256))
        ax2.set_ylim([-2,2])



def plotGlobalPL(sizeEvent,xmin=1,Theo = 0, alpa = 0,ax=None):

    if not ax:
        fig = figure()
        ax = subplot(1,1,1)

    [xCCDFAll,yCCDFAll,PDFAll] = get_CCDF(sizeEvent)
    sizeEventAll = np.copy(sizeEvent)
    sizeEvent = np.sort(sizeEvent[sizeEvent>=xmin])

    print(xmin)
    print(sizeEventAll)
    [xCCDF,yCCDF,PDF] = get_CCDF(sizeEvent)
    print(len(xCCDF))
    print(len(yCCDF))

    hold(True)

    #ax.plot((xCCDFAll),(yCCDFAll),'.-',color=(73./256, 142./256, 204./256),linewidth=2,markersize=15)
    ax.plot((xCCDF),(yCCDF),'.-',color=(73./256, 142./256, 204./256),linewidth=2,markersize=15)


    if alpa == 0:
        [best_chi,alpha,Theoretical_CCDF] = get_alpha(xCCDF,yCCDF,xmin,len(sizeEvent))
    else:
        alpha = alpa
        Theoretical_CCDF = Theo
        best_chi = 0

    #Theoretical_CCDF = (zeta(alpha, xCCDFAll) / zeta(alpha, xmin))

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


    legend(['Data',''.join(['Best slope = ', str(alpha)]),''.join(['paramLR = ', str(-slope1+1)]),],prop={'size':11},loc=3,frameon = False)
    print( ['xmin = ',str(xmin),'Alpha = ', str(alpha)])

    xlabel(r'Severity attack (s)')
    ylabel(r'$P(X>s)$')
    xlim([1,1.1*np.max(xCCDF)])

    #Get limits from lognormal
    #[mu,sigma,loc,best_chi,Theoretical_CCDF] = getMu(xCCDFAll,yCCDFAll,sizeEventAll,PDFAll)
    #ylim([np.min(np.concatenate([yCCDFAll,Theoretical_CCDF]))/1.5,0.2+np.max(np.concatenate([yCCDFAll,Theoretical_CCDF]))])
    xscale('log')
    yscale('log')


from scipy.special import erf

def lognormD(x,mu,sigma):
   return 0.5 + 0.5*erf(np.log(x) - mu)/np.sqrt(2*sigma**2)

def getMu(xCCDF,yCCDF,sizeEvent,PDF):
    from scipy.stats import lognorm
    from scipy.special import erf
    bins = np.sort(sizeEvent)
    best_chi = 50000
    N = len(sizeEvent)
    n = len(xCCDF)
    print( 'j')


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
    """
    for b in np.linspace(0.5,3,101):
        for c in np.linspace(0.5,3,101):
            Theoretical_CDF = lognorm.cdf(xCCDF,b*shape,d*loc,c*scale)
            Theoretical_PDF = lognorm.pdf(xCCDF,b*shape,d*loc,c*scale)
            Theoretical_CCDF = 1 - Theoretical_CDF

            chi  = np.sum((PDF-Theoretical_PDF*N)**2/(Theoretical_PDF*N))

            if chi < best_chi:
                bestb = b
                bestc = c
                bestd = d
                best_chi = chi

    best_chi =  stats.chi2.cdf(best_chi,n-2)
    """
    Theoretical_CDF = lognorm.cdf(xCCDF,bestb*shape,bestd*loc,bestc*scale)

    Theoretical_CCDF = 1- Theoretical_CDF


    print( bestb,bestc,bestd)

    return [(bestc*scale),bestb*shape,bestd*loc,1-best_chi,Theoretical_CCDF]

from scipy.misc import comb
from scipy.special import beta

def betaDist(x,a,b,n = 10):
    # k = x
    # n  = len(attacks)
    return comb(n,x) * beta(x+a, n-x+b) / beta(a,b)

def plotGlobalBeta(sizeEvent):
    from scipy.stats import beta
    from scipy.optimize import curve_fit
    import scipy


    sizeEvent = np.sort(sizeEvent[sizeEvent>0])
    [xCCDF,yCCDF,PDF] = get_CCDF(sizeEvent)

    popt, pcov = curve_fit(betaDist, xCCDF, PDF)
    print( "dasfaSDF")
    print( popt, pcov)


    hold(True)

    #plot((xCCDF),(yCCDF),'.-',color=(73./256, 142./256, 204./256),linewidth=2,markersize=10)

    dist = getattr(scipy.stats, "beta")
    param = dist.fit(sizeEvent,loc=0,scale=1)


    pdf_fitted = dist.pdf(xCCDF, *param[:-2], loc=param[-2], scale=param[-1])
    plot(xCCDF, 1-np.cumsum(pdf_fitted),'--',color='black',linewidth=2)

    print(param)





    ####Theoretical_CDF = beta.cdf(xCCDF,alpha1,beta1)
    ###Theoretical_CCDF = 1- Theoretical_CDF
    ###plot((xCCDF),(Theoretical_CCDF),'--',color='black',linewidth=2)


    from scipy.stats import lognorm
    Tho2 = 1-lognorm.cdf(xCCDF,1,0,5)
    #slope1, intercept, r_value, p_value, std_err = linregress(np.log10(xCCDF),np.log10(yCCDF))
    #plot(np.log10(xCCDF),intercept+np.log10(xCCDF)*slope1,color='red')
    #plot(np.log10(xCCDF),np.log10(Tho2),color='red')
    #legend(['Data',''.join(['Alpha = ', str(param[0]), ', Beta = ', str(param[1])])],prop={'size':12},loc=3)


    xlabel(r'Severity of attack')
    ylabel(r'$P(X>s)$')
    xlim([1,1.05*np.max(xCCDF)])
    ylim([np.min([yCCDF,1-np.cumsum(pdf_fitted)])/1.5,0.2+np.max([yCCDF,1-np.cumsum(pdf_fitted)])])
    xscale('log')
    yscale('log')


def plotGlobalLN(sizeEvent):
    sizeEvent = sizeEvent[sizeEvent>0]
    [xCCDF,yCCDF,PDF] = get_CCDF(sizeEvent)

    hold(True)

    plot((xCCDF),(yCCDF),'.-',color=(73./256, 142./256, 204./256),linewidth=2,markersize=10)


    [mu,sigma,loc,best_chi,Theoretical_CCDF] = getMu(xCCDF,yCCDF,sizeEvent,PDF)

    print("Best chi-sqare value for lognormal: ", best_chi, " Degree Freedom: ",len(Theoretical_CCDF)-2)
    plot((xCCDF),(Theoretical_CCDF),'--',color='black',linewidth=2)

    """
    x = np.log10(xCCDF)
    y = np.log10(yCCDF)
    x = x[:,np.newaxis]
    a, _, _, _ = np.linalg.lstsq(x, y)

    plot(np.log10(xCCDF),np.log10(xCCDF)*a,color='red')
    """
    from scipy.stats import lognorm
    Tho2 = 1-lognorm.cdf(xCCDF,1,0,5)
    #slope1, intercept, r_value, p_value, std_err = linregress(np.log10(xCCDF),np.log10(yCCDF))
    #plot(np.log10(xCCDF),intercept+np.log10(xCCDF)*slope1,color='red')
    #plot(np.log10(xCCDF),np.log10(Tho2),color='red')
    legend(['Data',''.join(['Mu = ', str(np.round(np.log(mu),2)), ', Sigma = ', str(np.round(sigma,2))])],prop={'size':12},loc=3)

    print(''.join(['Mu = ',str(np.round(np.log(mu),2)),'Sigma = ', str(np.round(sigma,2))]))
    xlabel(r'Severity of attack')
    ylabel(r'$P(X>s)$')
    xlim([1,1.05*np.max(xCCDF)])
    ylim([np.min([yCCDF,Theoretical_CCDF])/1.5,0.2+np.max([yCCDF,Theoretical_CCDF])])
    xscale('log')
    yscale('log')
    return np.log(mu), sigma

def plotD(a,b,c,aNT,vtm,ax = None,valImin = 0):

    if not ax: ax = subplot(a,b,c)
    hold(True)

    x = np.log10(valImin + np.arange(1,len(aNT)+1))
    y = np.log10((aNT))

    #scatter(x,y)
    ax.scatter(x,y,s=20,facecolor =(73./256, 142./256, 204./256),edgecolor=(73./256, 142./256, 204./256),lw=0)

    slope, intercept, r_value, p_value, std_err = linregress(x,y)

    plot(x,intercept+slope*x,'black')

    #plot(np.log10([vtm,vtm]),[min(y),max(y)],'red')
    #legend([''.join([r'$R^2$= ',str(r_value**2)]) ,''.join(['-b= ',str(slope)])])
    #legend(['b= ' + str(-slope)])
    #legend(['R2 = '+str(r_value**2)])
    ax.set_xlabel('n',fontsize=11)
    ax.set_ylabel('Tn',fontsize=11)
    #show()
    return [slope,intercept,r_value]


def plotFrequency(dates, daysBetweenAttacks,a1,a2,a3,label,vtm,ax=None,ddd=None,allX=True):
    import scipy.stats as stats

    import datetime
    
    import statsmodels.api as sm
    lowess = sm.nonparametric.lowess
    if not ax: ax = subplot(1,1,1)

    #figure(4,facecolor='white')
    """
    axis1 = np.ceil(np.sqrt(len(dates)))
    axis2 = np.ceil(np.sqrt(len(dates)))
    for i in range(len(dates)-1):
        cNT = (daysBetweenAttacks[dates[i]:dates[i+1]])
        plotD(axis1,axis2,i+1,cNT)

    """

    fNT = daysBetweenAttacks
    if len(fNT) > 0:

        y = np.log10((fNT))
        errMin = 0.
        valImin = 0
        """
        for i in range(0,0):
            #x = np.log10(i+np.arange(1,len(fNT)+1))
            #a = lowess(y,x,frac = 0.66,delta=0.0)
            [slope,intercept,r_value] = plotD(a1,a2,a3,fNT,vtm,ax=ax,valImin=i)
            print(i, slope,r_value)
            if r_value**2 > errMin:
                errMin = r_value**2
                valImin = i
        """
        [slope,intercept,r_value] = plotD(a1,a2,a3,fNT,vtm,ax=ax,valImin=valImin)

        x = np.log10(valImin+np.arange(1,len(fNT)+1))
        a = lowess(y,x,frac = 0.5,delta=0.0)
        ax.plot(a[:,0],a[:,1], linewidth=2,color='green',label=label)
        if ddd:
            ttt = [0]
            difA = diff(a[:,1])
            for elem in range(len(difA)-1):
                if difA[elem]*difA[elem+1] < 0:
                    ttt.append(elem)

            x2 = 0

            for elem in ttt:
                ax.annotate(str(ddd[elem+1].year),(a[elem,0],a[elem,1:]-0.2),fontsize=11,weight='bold')
                x2 += 1

        #title(label)

        col = ['red','blue','orange','green']
        i=-1
        listSlopes = []
        listX = []
        listStates = []
        lenT = 0
        """
        fitter = getCoeffLeastSquare #getCoeffLinear
        points  = np.transpose((x,y))
        linfit = multiFit(fitter, 2., degree=1)
        pointSplit = linfit.getPointSegments(points)

        for ps in pointSplit:
            i+=1
            trans = np.transpose(ps)
            ox = trans[0]
            oy = trans[1]

            pt = linfit.getCoeff(ox,oy)

            if len(ox) > 4:
                listSlopes.append(-pt[0])
                #listX.append(np.mean(y[np.max([0,lenT-1]):lenT+1]))
                listX.append(y[lenT])
                listStates.append(label)
                lenT += len(ox)

            ax.plot(ox,polyval(pt,ox) ,"-")

        ax.set_xlabel('Log10 Attack number')
        ax.set_ylabel('Log10 Time between attacks')
        xlim([0,max(x)])
        ylim([0,max(y)])
        #title('K-12',fontsize=11)
        customaxis(ax,size=10,lw=2)
        """
        customaxis(ax,size=10,lw=2)

        if allX == True:
            listSlopes = []
            tau = 0
            for x in range(np.size(fNT)-1):
                [_,intercept,r_value] = plotD(a1,a2,a3,fNT[x:x+2],vtm,ax=ax)

                listSlopes.append(-_)
                listX.append(np.log10(fNT[x]))
        print((fNT[0]),(fNT[0])/(1.+valImin)**(slope),(1.+valImin)**(-slope))
        return [np.log10(fNT[0]/((1.+valImin)**(slope))),-slope,listSlopes,listX,listStates,a[:,0],a[:,1]]
    else: return [np.NaN, np.NaN,[],[],[],[],[]]


def findXmin(sizeEvent):
    #Look for powerLaw xmin (Add a small amount for data loss in chi test)
    best_chi = -1
    i = 0

    lenS = len(np.unique(sizeEvent))
    a = 1
    for xmin in np.unique(sizeEvent)[:-10]:
        i+=1
        sizeEvent2 = np.sort(sizeEvent[sizeEvent>=xmin])
        [xCCDF,yCCDF,PDF] = get_CCDF(sizeEvent2)
        [chi,alpha,Theoretical_CCDF] = get_alpha(xCCDF,yCCDF,xmin,len(sizeEvent2))
        #pv =  stats.chi2.cdf(chi,len(np.unique(sizeEvent[sizeEvent>=xmin])))
        if chi > best_chi:
            best_chi = chi
            a = xmin

        #print chi*len(sizeEvent)/len(sizeEvent2),xmin


    return a
