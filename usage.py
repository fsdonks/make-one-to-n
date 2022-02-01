#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 21:08:33 2021

@author: craig
"""

import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('/home/craig/workspace/taa_processor/')
import taa_post_processing as proc

#Weights used for a weighted score.
phase_weights= {"comp1" : 0.125,
               "comp2" : 0.125,
               "phase1" : .0625,
               "phase2" : .0625,
               "phase3" : .5,
               "phase4" : .125}

results_map = {'D' : proc.resources+"results_no_truncate_and_1_supply.txt", 
               'C' : proc.resources+"results.txt",
              'E' : proc.resources+"results_no_truncation.txt"}

peak_max_workbook=proc.resources+"computed_maxes.xlsx"
baseline_path = proc.resources+'/TAA24-28_SRC_BASELINE_201130_DRAFTv6.xlsx'
out_location="/home/craig/workspace/taa_processor/"
proc.make_one_n(results_map, 
                peak_max_workbook, 
                out_location, 
                phase_weights, 
                'TAA24-28_Modeling_Results.xlsx', 
                baseline_path)
