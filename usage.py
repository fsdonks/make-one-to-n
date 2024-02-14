#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 21:08:33 2021

@author: craig
"""

from pathlib import Path

import sys
if '__file__' in globals():
    repo_path=str(Path(__file__).parent)
else: 
    repo_path='/home/craig/workspace/make-one-to-n'
    
repo_path=repo_path+'/'
sys.path.append(repo_path)

import taa_post_processing as proc #examples
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
out_location=repo_path+"test-output/"
Path(out_location).mkdir(parents=False, exist_ok=True)
proc.make_one_n(results_map, 
                peak_max_workbook, 
                out_location, 
                phase_weights, 
                'Modeling_Results.xlsx', 
                baseline_path, True)

proc.make_one_n(results_map, 
                peak_max_workbook, 
                out_location, 
                phase_weights, 
                'Modeling_Results_no-dropdown.xlsx', 
                baseline_path, True, drop_down=False)

proc.remove_blank_row(out_location+"out_of_order.xlsx", out_location+"out_of_order.xlsx")

import src_replacer as replacer #examples

replacer.do_tests(repo_path)

phase_weights_concated= {"comp1" : 0.125, #won't be used
               "comp2" : 0.125,  #won't be used
               "phase1-C" : .1,
               "phase2-C" : .1,
               "phase3-C" : .1,
               "phase4-C" : .1,
               "comp1-E" :  .1,
               "comp2-E" :  .1,
               "phase1-E" : .1,
               "phase2-E" : .1,
               "phase3-E" : .1,
               "phase4-E" : .1}

#If demands are weighted like C is worth 90% and E is worth 10%,
#can also make the concated phase weights more easily with something like
splits=proc.split_run_weights({"C" : .9, "E" : .1}, phase_weights)

proc.one_n_across_runs(results_map, 
                resources+"computed_maxes_dummy.xlsx", 
                out_location, 
                phase_weights_concated, 
                'Modeling_Results_concatenated.xlsx', 
                baseline_path, True)