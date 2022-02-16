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

resources="/home/craig/workspace/taa_processor/resources/"

results_map = {#'D' : resources+"results_no_truncate_and_1_supply.txt", 
               'C' : resources+"results.txt",
              'E' : resources+"results_no_truncation.txt"
              }

peak_max_workbook=resources+"computed_maxes.xlsx"
baseline_path = resources+'/TAA24-28_SRC_BASELINE_201130_DRAFTv6.xlsx'
out_location="/home/craig/workspace/taa_processor/"
proc.make_one_n(results_map, 
                peak_max_workbook, 
                out_location, 
                phase_weights, 
                'TAA24-28_Modeling_Results.xlsx', 
                baseline_path, True)

proc.remove_blank_row(out_location+"out_of_order.xlsx", out_location+"out_of_order.xlsx")