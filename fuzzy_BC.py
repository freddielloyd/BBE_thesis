#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 16:45:38 2022

@author: freddielloyd
"""

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

class fuzzy_BC:
    
    def __init__(self):
        
        x_lbound = 0
        x_ubound = 1.01
        x_interval = 0.01
        self.x = np.arange(x_lbound, x_ubound, x_interval) 
        
    def fuzzification(self, mfx, opinion_gap):
        
        # Triangular membership function output - values are bottom left, top and, bottom right of triangle respectively
        if mfx == 'triangular':
            
            # triangle(x; a,b,c) = 
            # 0,       x<= a
            # x-a/b-a, a<=x<=b
            # c-x/c-b, b<=x<=c
            # 0,       c<=x       
            
# =============================================================================
#             vs_mfx = fuzz.trimf(self.x,[0, 0, 0.1])     # very small
#             qs_mfx = fuzz.trimf(self.x,[0, 0.1, 0.2])   # quite small
#             s_mfx = fuzz.trimf(self.x,[0.1, 0.2, 0.3]) # small
#             l_mfx = fuzz.trimf(self.x,[0.3, 0.4, 0.5]) # large
#             ql_mfx = fuzz.trimf(self.x,[0.4, 0.5, 0.6]) # quite large
#             vl_mfx = fuzz.trimf(self.x,[0.5, 1, 1])     # very large
#             
#             tri_mfxs = [vs_mfx, qs_mfx, s_mfx, l_mfx, ql_mfx, vl_mfx]
# =============================================================================
            
            tri_mfxs = []

            if 0 <= opinion_gap <= 0.1:
                vs_mfx = fuzz.trimf(self.x,[0, 0, 0.1])     # very small
                tri_mfxs.append(vs_mfx)
            if 0 <= opinion_gap <= 0.2:
                qs_mfx = fuzz.trimf(self.x,[0, 0.1, 0.2])     # quite small
                tri_mfxs.append(qs_mfx)
            if 0.1 <= opinion_gap <= 0.3:
                s_mfx = fuzz.trimf(self.x,[0.1, 0.2, 0.3])     # small
                tri_mfxs.append(s_mfx)
            if 0.3 <= opinion_gap <= 0.5:
                l_mfx = fuzz.trimf(self.x,[0.3, 0.4, 0.5])     # large
                tri_mfxs.append(l_mfx)
            if 0.4 <= opinion_gap <= 0.6:
                ql_mfx = fuzz.trimf(self.x,[0.4, 0.5, 0.6])     # quite large
                tri_mfxs.append(ql_mfx)
            if 0.5 <= opinion_gap <= 1:
                vl_mfx = fuzz.trimf(self.x,[0.5, 1, 1])     # very large
                tri_mfxs.append(vl_mfx)
                
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
                
                
    def defuzz_xvals(self, mfxs, method): # method = centroid, bisector, or mean of maximum
        
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
        
        
# =============================================================================
#         defuzz_yvals = []
#         
#         for mfx in mfxs:
#             # y max is estimated interpolated y values based on x defuzz values, interpolated from x and mfx values
#             ymax = [fuzz.interp_membership(self.x, mfx, i) for i in defuzz_xvals] # i is x to evaluate, x is x coords, x
#             
#             defuzz_yvals.append(ymax)
#             
#         return defuzz_yvals
#     
# =============================================================================
    
        defuzz_yvals = []
        for i in range(len(defuzz_xvals)):
            
            #print(tri_mfxs[i], defuzz_xvals[i])
            
            ymax = fuzz.interp_membership(self.x, mfxs[i], defuzz_xvals[i])
            
            defuzz_yvals.append(ymax)
            
        return defuzz_yvals
    
    
    def plot_membership_fxs(self, mfxs, defuzz_xvals, defuzz_yvals):
        
        
        labels=['very small','quite small','small','large','quite large','very large']
        colors = ['r','b','g','c','m','y']
        
        for i in range(len(mfxs)):
            plt.figure(figsize=(8,5))
            plt.plot(self.x, mfxs[i], 'k')

            plt.vlines(defuzz_xvals[i], ymin = 0, ymax = defuzz_yvals[i] ,label=labels[i], color=list(colors[i]))
            plt.ylabel('Fuzzy membership')
            plt.xlabel('Agent Total Opinion')
            plt.ylim(-0.1,1.1)
            plt.legend(loc=2)
            
            plt.show()
        

# =============================================================================
# m = fuzzy_BC()
# 
# mfxs = m.fuzzification('triangular')
# 
# defuzz_x = m.defuzz_xvals(mfxs, 'centroid')
# 
# defuzz_y = m.defuzz_yvals(mfxs, defuzz_x)
# 
# m.plot_membership_fxs(mfxs, defuzz_x, defuzz_y)
# =============================================================================







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
