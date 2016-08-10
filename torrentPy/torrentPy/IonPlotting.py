#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
"""
Created on Thu Aug  8 11:01:00 2013

@author: eingerman
"""
import pylab as pl
import statsmodels.api as sm

def plot_regression(x,y,smoothing=.3):
    fit=sm.nonparametric.lowess(y,x,frac=smoothing)
    df=(fit[:,1]-y)**2
    fit_var=sm.nonparametric.lowess(df,x,frac=smoothing)
    isheld = pl.ishold()
    pl.hold(1)
    pl.plot(fit[:,0],fit[:,1])
    pl.fill_between(fit_var[:,0],fit[:,1]-1*pl.sqrt(fit_var[:,1]),fit[:,1]+1*pl.sqrt(fit_var[:,1]),color=((0,0,.99,.2),))
    pl.plot(x,y,'.')
    pl.hold(isheld)
