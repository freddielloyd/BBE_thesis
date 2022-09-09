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
        
    def fuzzification(self, mfx, opinion_gap, weight_segmentation):
        
        # Triangular membership function output - values are bottom left, top and, bottom right of triangle respectively
        if mfx == 'triangular':
            
            # triangle(x; a,b,c) = 
            # 0,       x<= a
            # x-a/b-a, a<=x<=b
            # c-x/c-b, b<=x<=c
            # 0,       c<=x       
            
            # Generate triangular membership functions 
            
# =============================================================================
#             # test membership functions
#             # Opinion Gap Membership
#             op_vsm = fuzz.trimf(self.x_opinion_gap, [0, 0, 0.1]) # very small
#             op_qsm = fuzz.trimf(self.x_opinion_gap, [0, 0.1, 0.2]) # quite small
#             op_sm = fuzz.trimf(self.x_opinion_gap, [0.1, 0.2, 0.3]) # small
#             op_l = fuzz.trimf(self.x_opinion_gap, [0.3, 0.4, 0.5]) # large
#             op_ql = fuzz.trimf(self.x_opinion_gap, [0.4, 0.5, 0.6]) # quite large
#             op_vl = fuzz.trimf(self.x_opinion_gap, [0.5, 1, 1]) # very large
#             
#             
#             # Agent interaction update weight membership
#             w_vstr = fuzz.trimf(self.x_weight, [0.8, 1, 1]) # very strong
#             w_qstr = fuzz.trimf(self.x_weight, [0.6, 0.8, 1]) # quite strong
#             w_str = fuzz.trimf(self.x_weight, [0.4, 0.6, 0.8]) # strong
#             w_w = fuzz.trimf(self.x_weight, [0.2, 0.4, 0.6]) # weak
#             w_qw = fuzz.trimf(self.x_weight, [0, 0.2, 0.4]) # quite weak
#             w_vw = fuzz.trimf(self.x_weight, [0, 0, 0.2]) # very weak
# =============================================================================


            # Opinion Gap Membership - NEW: 0.276 is very large and correspond to zero weight
            op_vsm = fuzz.trimf(self.x_opinion_gap, [0, 0, 0.07]) # very small
            op_qsm = fuzz.trimf(self.x_opinion_gap, [0, 0.07, 0.14]) # quite small
            op_sm = fuzz.trimf(self.x_opinion_gap, [0.07, 0.14, 0.21]) # small
            op_l = fuzz.trimf(self.x_opinion_gap, [0.14, 0.21, 0.28]) # large
            op_ql = fuzz.trimf(self.x_opinion_gap, [0.21, 0.28, 0.35]) # quite large
            op_vl = fuzz.trimf(self.x_opinion_gap, [0.28, 1, 1]) # very large
            
            if weight_segmentation =='a':
                w_vstr = fuzz.trimf(self.x_weight, [0.8, 1, 1]) # very strong
                w_qstr = fuzz.trimf(self.x_weight, [0.67, 0.83, 0.83]) # quite strong
                w_str = fuzz.trimf(self.x_weight, [0.55, 0.7, 0.7]) # strong
                w_w = fuzz.trimf(self.x_weight, [0.33, 0.33, 0.58]) # weak
                w_qw = fuzz.trimf(self.x_weight, [0, 0.03, 0.35]) # quite weak
                w_vw = fuzz.trimf(self.x_weight, [0, 0, 0]) # very weak
                        
            elif weight_segmentation == 'b':
                w_vstr = fuzz.trimf(self.x_weight, [0.78, 1, 1]) # very strong
                w_qstr = fuzz.trimf(self.x_weight, [0.58, 0.8, 0.8]) # quite strong
                w_str = fuzz.trimf(self.x_weight, [0.28, 0.6, 0.6]) # strong
                w_w = fuzz.trimf(self.x_weight, [0.18, 0.18, 0.38]) # weak
                w_qw = fuzz.trimf(self.x_weight, [0, 0.03, 0.2]) # quite weak
                w_vw = fuzz.trimf(self.x_weight, [0, 0, 0]) # very weak
            
            elif weight_segmentation == 'c':
                w_vstr = fuzz.trimf(self.x_weight, [0.16, 1, 1]) # very strong
                w_qstr = fuzz.trimf(self.x_weight, [0.12, 0.18, 0.18]) # quite strong
                w_str = fuzz.trimf(self.x_weight, [0.08, 0.14, 0.14]) # strong
                w_w = fuzz.trimf(self.x_weight, [0.04, 0.04, 0.1]) # weak
                w_qw = fuzz.trimf(self.x_weight, [0, 0.02, 0.06]) # quite weak
                w_vw = fuzz.trimf(self.x_weight, [0, 0, 0]) # very weak
                        
            elif weight_segmentation == 'd':
                w_vstr = fuzz.trimf(self.x_weight, [0.04, 1, 1]) # very strong
                w_qstr = fuzz.trimf(self.x_weight, [0.03, 0.04, 0.04]) # quite strong
                w_str = fuzz.trimf(self.x_weight, [0.02, 0.03, 0.03]) # strong
                w_w = fuzz.trimf(self.x_weight, [0, 0.01, 0.02]) # weak
                w_qw = fuzz.trimf(self.x_weight, [0, 0.01, 0.01]) # quite weak
                w_vw = fuzz.trimf(self.x_weight, [0, 0, 0]) # very weak
                
            elif weight_segmentation == 'a1':
                w_vstr = fuzz.trimf(self.x_weight, [0.84, 1, 1]) # very strong
                w_qstr = fuzz.trimf(self.x_weight, [0.7, 0.9, 0.9]) # quite strong
                w_str = fuzz.trimf(self.x_weight, [0.61, 0.77, 0.77]) # strong
                w_w = fuzz.trimf(self.x_weight, [0.39, 0.39, 0.65]) # weak
                w_qw = fuzz.trimf(self.x_weight, [0, 0.09, 0.42]) # quite weak
                w_vw = fuzz.trimf(self.x_weight, [0, 0, 0]) # very weak
                        
            
            
            
            
            
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
            

            # Aggregate all six output membership functions together by taking element wise maximums
            fuzzy_set = np.fmax(w_activation_1,
                                 np.fmax(w_activation_2,
                                         np.fmax(w_activation_3,
                                                 np.fmax(w_activation_4,
                                                         np.fmax(w_activation_5, w_activation_6)))))
            

            return fuzzy_set


        # Trapezoidal membership function output - values are four corners of trapezium
        
        elif mfx == 'trapezoidal':
          
          
            
            # trapezoid(x; a,b,c,d) = 
            # 0,       x<= a
            # x-a/b-a, a<=x<=b
            # 1,       b<=x<=c
            # d-x/d-c, c<=x<=d
            # 0,       d<=x
            

            
            # Generate trapezoidal membership functions 
            
            # Opinion Gap Membership
            op_vsm = fuzz.trapmf(self.x_opinion_gap, [0, 0, 0.1, 0.2]) # very small
            op_qsm = fuzz.trapmf(self.x_opinion_gap, [0, 0.1, 0.2, 0.3]) # quite small
            op_sm = fuzz.trapmf(self.x_opinion_gap, [0.1, 0.2, 0.3, 0.4]) # small
            op_l = fuzz.trapmf(self.x_opinion_gap, [0.3, 0.4, 0.5, 0.6]) # large
            op_ql = fuzz.trapmf(self.x_opinion_gap, [0.4, 0.5, 0.6, 0.7]) # quite large
            op_vl = fuzz.trapmf(self.x_opinion_gap, [0.5, 0.7, 0.8, 1]) # very large
            
            
            # Agent interaction update weight membership
            w_vstr = fuzz.trapmf(self.x_weight, [0.8, 0.9, 1, 1]) # very strong
            w_qstr = fuzz.trapmf(self.x_weight, [0.6, 0.7, 0.8, 0.8]) # quite strong
            w_str = fuzz.trapmf(self.x_weight, [0.4, 0.5, 0.6, 0.7]) # strong
            w_w = fuzz.trapmf(self.x_weight, [0.2, 0.3, 0.4, 0.5]) # weak
            w_qw = fuzz.trapmf(self.x_weight, [0.1, 0.2, 0.3, 0.4]) # quite weak
            w_vw = fuzz.trapmf(self.x_weight, [0, 0, 0.1, 0.2]) # very weak



        elif mfx == 'gaussian':
            

            # Generate gaussian membership functions 
            
            # arguments of gaussmf are mean and standard deviation
            
            # Opinion Gap Membership
            op_vsm = fuzz.gaussmf(self.x_opinion_gap, [0.1, 0.05]) # very small
            op_qsm = fuzz.gaussmf(self.x_opinion_gap, [0.2, 0.05]) # quite small
            op_sm = fuzz.gaussmf(self.x_opinion_gap, [0.3, 0.1]) # small
            op_l = fuzz.gaussmf(self.x_opinion_gap, [0.4, 0.1]) # large
            op_ql = fuzz.gaussmf(self.x_opinion_gap, [0.6, 0.2]) # quite large
            op_vl = fuzz.gaussmf(self.x_opinion_gap, [0.8, 0.2]) # very large
            
            
            # Agent interaction update weight membership
            w_vstr = fuzz.gaussmf(self.x_weight, [0.8, 0.2]) # very strong
            w_qstr = fuzz.trimf(self.x_weight, [0.6, 0.2]) # quite strong
            w_str = fuzz.trimf(self.x_weight, [0.4, 0.1]) # strong
            w_w = fuzz.trimf(self.x_weight, [0.3, 0.1]) # weak
            w_qw = fuzz.trimf(self.x_weight, [0.2, 0.05]) # quite weak
            w_vw = fuzz.trimf(self.x_weight, [0.1, 0.05]) # very weak


            
    def defuzzification(self, fuzzy_set, method):
        
        # Calculate defuzzified result
        if method == 'centroid':
            crisp_weight = fuzz.defuzz(self.x_weight, fuzzy_set, 'centroid')
        elif method == 'bisector':
            crisp_weight = fuzz.defuzz(self.x_weight, fuzzy_set, 'bisector')
        elif method == 'mean_of_max':
            crisp_weight = fuzz.defuzz(self.x_weight, fuzzy_set, 'mom')
            
        return crisp_weight
            
        
        ## Visualize every centroid defuzzification process - weight on x axis, membership on y axis
        #
        #array of zeros same length as input array
        #weight0 = np.zeros_like(self.x_weight)
        #           
        #weight_activation = fuzz.interp_membership(self.x_weight, aggregated, crisp_weight)  # for plot
        #
        #fig, ax0 = plt.subplots(figsize=(8, 4))
        #
        #ax0.plot(self.x_weight, w_vstr, 'r', linewidth=0.5, linestyle='--', )
        #ax0.plot(self.x_weight, w_qstr, 'g', linewidth=0.5, linestyle='--')
        #ax0.plot(self.x_weight, w_str, 'b', linewidth=0.5, linestyle='--')
        #ax0.plot(self.x_weight, w_w, 'c', linewidth=0.5, linestyle='--', )
        #ax0.plot(self.x_weight, w_qw, 'm', linewidth=0.5, linestyle='--')
        #ax0.plot(self.x_weight, w_vw, 'y', linewidth=0.5, linestyle='--')
        #ax0.fill_between(self.x_weight, weight0, aggregated, facecolor='Red', alpha=0.7)
        #ax0.plot([crisp_weight, crisp_weight], [0, weight_activation], 'k', linewidth=1.5, alpha=0.9)
        #ax0.set_title('Aggregated membership and result (line)')
        #
        # Turn off top/right axes
        #for ax in (ax0,):
        #    ax.spines['top'].set_visible(False)
        #    ax.spines['right'].set_visible(False)
        #    ax.get_xaxis().tick_bottom()
        #    ax.get_yaxis().tick_left()
        #
        #plt.tight_layout()


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
        