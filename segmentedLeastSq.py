# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:30:55 2012

This program uses Segmented Least Squares algorithm in order to 
fit composite series of point into several curves. 


See usage example below. 

Bibliography:
[1] Kleinberg, Jon, and Éva Tardos. Algorithm Design. 1st ed. Addison-Wesley, 2005.
[2] http://stackoverflow.com/a/4084918



Author: Ronen Abravanel, ronen@tx.technion.ac.il

Copyright (C) 2012,     Ronen Abravanel

This work is licensed under a Creative Commons Attribution 3.0 Unported License.
http://creativecommons.org/licenses/by/3.0/

"""

import numpy as np
from numpy import *





def getCoeffLinear(x,y, errof = None, degree=0):
    
    n = len(x)
    if n==1:
        dx = x[0]
        dy = y[0]
        return (dy/dx, 0)
    if n==0:
        return (1,0)
    a = (n*np.sum(x*y) - x.sum() *y.sum())/(n*np.sum(x**2) - x.sum()**2)
    b = (y.sum() - a * x.sum())/n
    return (a, b)
    

from scipy.optimize import leastsq 
def getCoeffLeastSquare(x,y, errof, degree=1):
        p0 = ones(degree+1)
        
        if len(x) < len(p0):
            z = len(p0) - len(x)
            x = concatenate((x,zeros(z+1)))
            y = concatenate((y,zeros(z+1)))
            
        ret =  leastsq(errof,  p0, args=(y,x))
        
        return ret[0]
        

class multiFit():
    
    def __init__(self, getCoeff, split_cost = 1, degree = 1):
        self.split_cost = split_cost
        self._getCoeff = getCoeff
        self.degree = degree
        self.polyval = np.polyval

    
    


    def getCoeff(self, x,y):
        return self._getCoeff(x,y,self.errof, self.degree)
        
    def errof(self, p,y,x):
        return y-polyval(p,x)
    
    def lsqFitCost(self, points):
            trans = transpose(points)
            
            x = trans[0]
            y = trans[1]
            p = self.getCoeff(x,y)
            return  sum((y-polyval(p,x))**2)
    
    def lsqFitCostC(self, points, i, j):    
        return self.lsqFitCost(points[i:j])
            
    
    
    def optimalSolution(self, points):
        split_cost = self.split_cost
        solutions = {0:{'cost':0,'splits':[]}}
        for j in range(1,len(points)):
            best_split = None
            best_cost = self.lsqFitCostC(points,0,j)
            for i in range(0,j):
                cost = solutions[i]['cost'] + split_cost + self.lsqFitCostC(points,i+1,j)
                if cost == NaN: print(points, self.lsqFitCostC(points,i+1,j), split_cost)
                if cost < best_cost:
                   best_cost = cost
                   best_split = i
            if best_split != None:
                solutions[j] = {'cost':best_cost,'splits':solutions[best_split]['splits']+[best_split]}
            else:
                solutions[j] = {'cost':best_cost,'splits':[]}

        return solutions


    def getPointSegments(self, points):
        sol = self.optimalSolution(points)
        GoodSol = sol[max(sol.keys())] 
        split = GoodSol["splits"]
        split.append(len(points))

        pointSplit = []
        j = 0
        for i in split:
            pointSplit.append(points[j:i])
            j = i
        return pointSplit






if __name__ == "__main__":
    from pylab import * 
    def test(fitter, x,y):
        points  = transpose((x,y))
        pointSplit = fitter.getPointSegments(points)

        for ps in pointSplit:
            trans = transpose(ps)
            ox = trans[0]
            oy = trans[1]
            pt = fitter.getCoeff(ox,oy)
            
            plot(ox, polyval(pt,ox),"-", label=(",".join([str(p) for p in pt])))
            
        
        legend(loc=2)
        plot(x,y, ".")

    
    x = linspace(0,5,250)
    
    y1 = concatenate([x[:50] , x[50:100]*2-1 , x[100:150]+1 , 2.5+0.5*x[150:200], 16.5-x[200:]*3])
    y2 = concatenate([(x[:50]-0.5)**2 , (x[50:100]-1.5)**2 , (x[100:150]-2.5)**2 , (x[150:200]-4)**2, (x[200:]-4.5)**2])
    
    
    linfit = multiFit(getCoeffLinear, 0.1)
    squarefit = multiFit(getCoeffLeastSquare, 0.00001, degree=2)
    
    figure(0)
    test(linfit,x,y1)
    figure(1)
    test(squarefit,x,y2)
    show()
