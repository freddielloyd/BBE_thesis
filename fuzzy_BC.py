#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 16:45:38 2022

@author: freddielloyd
"""

import numpy as np
import skfuzzy as fuzz

class fuzzy_BC:
    
    def __init__(self):
        
        x_lbound = 0
        x_ubound = 1.05
        x_interval = 0.1
        self.x = np.arange(x_lbound, x_ubound, x_interval) 
        
    def fuzzification(self, mfx):
        
        # Triangular membership function output - values are bottom left, top and, bottom right of triangle respectively
        if mfx == 'triangular':
            
            # triangle(x; a,b,c) = 
            # 0,       x<= a
            # x-a/b-a, a<=x<=b
            # c-x/c-b, b<=x<=c
            # 0,       c<=x        
            
            vs_mfx = fuzz.trimf(self.x,[0, 0, 0.1])     # very small
            qs_mfx = fuzz.trimf(self.x,[0, 0.1, 0.2])   # quite small
            s_mfx = fuzz.trimf(self.x,[0.1, 0.2, 0.3]) # small
            l_mfx = fuzz.trimf(self.x,[0.3, 0.4, 0.5]) # large
            ql_mfx = fuzz.trimf(self.x,[0.4, 0.5, 0.6]) # quite large
            vl_mfx = fuzz.trimf(self.x,[0.5, 1, 1])     # very large
            
            tri_mfxs = [vs_mfx, qs_mfx, s_mfx, l_mfx, ql_mfx, vl_mfx]
            
            return tri_mfxs
        
        # Trapezoidal membership function output - values are four corners of trapezium
        
        elif mfx == 'trapezoidal':
            
            # trapezoid(x; a,b,c,d) = 
            # 0,       x<= a
            # x-a/b-a, a<=x<=b
            # 1,       b<=x<=c
            # d-x/d-c, c<=x<=d
            # 0,       d<=x
            
            x = np.arange(0, 5.05, 0.1)
            mfx = fuzz.trapmf(x,[2, 2.5, 3, 4.5])
            
            for i in range(len(x)):
                print(x[i], mfx[i])
                
                
    def defuzz_xvals(self, mfxs, method):
        # method = centroid, bisector, or mean of maximum
    
        defuzz_xvals = []
        
    
        for mfx in mfxs:
  
            if method == 'centroid':
                defuzz_centroid = fuzz.defuzz(self.x, mfx, 'centroid')
                defuzz_xvals.append(defuzz_centroid)
                
            elif method == 'bisector':
                defuzz_bisector = fuzz.defuzz(self.x, mfx, 'bisector')
                defuzz_xvals.append(defuzz_bisector)
                
            elif method == 'mean_of_max':
                defuzz_mom = fuzz.defuzz(self.x, mfx, 'mom')
                defuzz_xvals.append(defuzz_mom)
     
        return defuzz_xvals
                
    
    def defuzz_yvals(self, mfxs, defuzz_xvals):
        
        defuzz_yvals = []
        
        for mfx in mfxs:
            # y max is estimated interpolated y values based on x defuzz values, interpolated from x and mfx values
            ymax = [fuzz.interp_membership(self.x, mfx, i) for i in defuzz_xvals] # i is x to evaluate, x is x coords, x
            
            defuzz_yvals.append(ymax)
            
        return defuzz_yvals
        

m = fuzzy_BC()

mfxs = m.fuzzification('triangular')

defuzz_xvals = m.defuzz_xvals(mfxs, 'centroid')

defuzz_yvals = m.defuzz_yvals(mfxs, defuzz_xvals)




# =============================================================================
# x_lbound = 0
# x_ubound = 1.05
# x_interval = 0.1
# x = np.arange(x_lbound, x_ubound, x_interval) 
# 
# #monotonically increasing with each stronger membership function
# fuzz.defuzz(x, mfxs[0], 'centroid')
# fuzz.defuzz(x, mfxs[1], 'centroid')
# fuzz.defuzz(x, mfxs[2], 'centroid')
# fuzz.defuzz(x, mfxs[3], 'centroid')
# fuzz.defuzz(x, mfxs[4], 'centroid')
# fuzz.defuzz(x, mfxs[5], 'centroid')
# 
# 
# fuzz.defuzz(x, mfx, 'bisector')
# fuzz.defuzz(x, mfx, 'mom') # mean of maximum
# =============================================================================
