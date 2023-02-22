#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 21:08:33 2021

@author: craig
"""

from pathlib import Path
import sys
if '__file__' not in globals():
    repo_path=Path(__file__).parent
else: 
    repo_path='/home/craig/workspace/taa_processor'
repo_path=repo_path+'/'
sys.path.append(repo_path)

import taa_post_processing as proc

#Weights used for a weighted score.
phase_weights= {"comp1" : 0.125,
               "comp2" : 0.125,
               "phase1" : .0625,
               "phase2" : .0625,
               "phase3" : .5,
               "phase4" : .125}

resources=repo_path+"resources/"

results_map = {#'D' : resources+"results_no_truncate_and_1_supply.txt", 
               'C' : resources+"results.txt",
              'E' : resources+"results_no_truncation.txt"
              }

peak_max_workbook=resources+"computed_maxes.xlsx"
baseline_path = resources+'/SRC_BASELINE.xlsx'
out_location="/home/craig/workspace/taa_processor/"
proc.make_one_n(results_map, 
                peak_max_workbook, 
                out_location, 
                phase_weights, 
                'Modeling_Results.xlsx', 
                baseline_path, True)

proc.remove_blank_row(out_location+"out_of_order.xlsx", out_location+"out_of_order.xlsx")