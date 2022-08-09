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
        
    
                
        # Generate universe variables
        self.x_opinion_gap = np.arange(x_lbound, x_ubound, x_interval)
        
        self.x_weight = np.arange(x_lbound, x_ubound, x_interval)
        
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

            
            
            # Generate fuzzy membership functions 
            
            # Opinion Gap Membership
            op_vsm = fuzz.trimf(self.x_opinion_gap, [0, 0, 0.1]) # very small
            op_qsm = fuzz.trimf(self.x_opinion_gap, [0, 0.1, 0.2]) # quite small
            op_sm = fuzz.trimf(self.x_opinion_gap, [0.1, 0.2, 0.3]) # small
            op_l = fuzz.trimf(self.x_opinion_gap, [0.3, 0.4, 0.5]) # large
            op_ql = fuzz.trimf(self.x_opinion_gap, [0.4, 0.5, 0.6]) # quite large
            op_vl = fuzz.trimf(self.x_opinion_gap, [0.5, 1, 1]) # very large
            
            
            # Agent interaction update weight membership
            w_vstr = fuzz.trimf(self.x_weight, [0.8, 1, 1]) # very strong
            w_qstr = fuzz.trimf(self.x_weight, [0.6, 0.8, 1]) # quite strong
            w_str = fuzz.trimf(self.x_weight, [0.4, 0.6, 0.8]) # strong
            w_w = fuzz.trimf(self.x_weight, [0.2, 0.4, 0.6]) # weak
            w_qw = fuzz.trimf(self.x_weight, [0, 0.2, 0.4]) # quite weak
            w_vw = fuzz.trimf(self.x_weight, [0, 0, 0.2]) # very weak




# =============================================================================
#             tri_mfxs = []
# 
#             if 0 <= opinion_gap <= 0.1:
#                 vs_mfx = fuzz.trimf(self.x,[0, 0, 0.1])     # very small
#                 tri_mfxs.append(vs_mfx)
#             if 0 <= opinion_gap <= 0.2:
#                 qs_mfx = fuzz.trimf(self.x,[0, 0.1, 0.2])     # quite small
#                 tri_mfxs.append(qs_mfx)
#             if 0.1 <= opinion_gap <= 0.3:
#                 s_mfx = fuzz.trimf(self.x,[0.1, 0.2, 0.3])     # small
#                 tri_mfxs.append(s_mfx)
#             if 0.3 <= opinion_gap <= 0.5:
#                 l_mfx = fuzz.trimf(self.x,[0.3, 0.4, 0.5])     # large
#                 tri_mfxs.append(l_mfx)
#             if 0.4 <= opinion_gap <= 0.6:
#                 ql_mfx = fuzz.trimf(self.x,[0.4, 0.5, 0.6])     # quite large
#                 tri_mfxs.append(ql_mfx)
#             if 0.5 <= opinion_gap <= 1:
#                 vl_mfx = fuzz.trimf(self.x,[0.5, 1, 1])     # very large
#                 tri_mfxs.append(vl_mfx)
#                 
#             return tri_mfxs
# =============================================================================

            
            
            # We need the activation of our fuzzy membership functions at a value for opinion gap.
            # fuzz.interp_membership retrieves the y value for where the given opinion gap x value meets any of the 
            # triangular membership functions
            
            
            op_level_vsm = fuzz.interp_membership(self.x_opinion_gap, op_vsm, opinion_gap)
            op_level_qsm = fuzz.interp_membership(self.x_opinion_gap, op_qsm, opinion_gap)
            op_level_sm = fuzz.interp_membership(self.x_opinion_gap, op_sm, opinion_gap)
            op_level_l = fuzz.interp_membership(self.x_opinion_gap, op_l, opinion_gap)
            op_level_ql = fuzz.interp_membership(self.x_opinion_gap, op_ql, opinion_gap)
            op_level_vl = fuzz.interp_membership(self.x_opinion_gap, op_vl, opinion_gap)
            
            
            # Now apply the rules - here just 'if then' statements for each rule
            # use np.fmin to take the element wise minimum of the y value obtained above for the opinion gap membership functions
            # and the membership functions for weight - this essentially clips the triangular weight membership functions so
            # their y values do not exceed the obtained values from above which is necessary for correct defuzzification
            w_activation_1 = np.fmin(op_level_vsm, w_vstr)   
            w_activation_2 = np.fmin(op_level_qsm, w_qstr)   
            w_activation_3 = np.fmin(op_level_sm, w_str) 
            w_activation_4 = np.fmin(op_level_l, w_w)   
            w_activation_5 = np.fmin(op_level_ql, w_qw)   
            w_activation_6 = np.fmin(op_level_vl, w_vw)         
            
            
            
            # array of zeros same length as input array
            weight0 = np.zeros_like(self.x_weight)
            
            
                    
            # Aggregate all six output membership functions together by taking element wise maximums
            aggregated = np.fmax(w_activation_1,
                                 np.fmax(w_activation_2,
                                         np.fmax(w_activation_3,
                                                 np.fmax(w_activation_4,
                                                         np.fmax(w_activation_5, w_activation_6)))))
            
            
            
            
            
            #for i in range(len(x_weight)):
            #    print(x_weight[i], aggregated[i])
    
            # Calculate defuzzified result
            crisp_weight = fuzz.defuzz(self.x_weight, aggregated, 'centroid')
            weight_activation = fuzz.interp_membership(self.x_weight, aggregated, crisp_weight)  # for plot
    
            
# =============================================================================
#             # Visualize every centroid defuzzification process - weight on x axis, membership on y axis
#             fig, ax0 = plt.subplots(figsize=(8, 4))
#     
#             ax0.plot(self.x_weight, w_vstr, 'r', linewidth=0.5, linestyle='--', )
#             ax0.plot(self.x_weight, w_qstr, 'g', linewidth=0.5, linestyle='--')
#             ax0.plot(self.x_weight, w_str, 'b', linewidth=0.5, linestyle='--')
#             ax0.plot(self.x_weight, w_w, 'c', linewidth=0.5, linestyle='--', )
#             ax0.plot(self.x_weight, w_qw, 'm', linewidth=0.5, linestyle='--')
#             ax0.plot(self.x_weight, w_vw, 'y', linewidth=0.5, linestyle='--')
#             ax0.fill_between(self.x_weight, weight0, aggregated, facecolor='Red', alpha=0.7)
#             ax0.plot([crisp_weight, crisp_weight], [0, weight_activation], 'k', linewidth=1.5, alpha=0.9)
#             ax0.set_title('Aggregated membership and result (line)')
#     
#             # Turn off top/right axes
#             for ax in (ax0,):
#                 ax.spines['top'].set_visible(False)
#                 ax.spines['right'].set_visible(False)
#                 ax.get_xaxis().tick_bottom()
#                 ax.get_yaxis().tick_left()
#     
#             plt.tight_layout()
# =============================================================================
    
            return crisp_weight


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
