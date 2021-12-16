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

results_map = {'D' : proc.resources+"results_no_truncate_and_1_supply.txt", 
               'C' : proc.resources+"results.txt",
              'E' : proc.resources+"results_no_truncation.txt"}

peak_max_workbook=proc.resources+"computed_maxes.xlsx"

proc.make_one_n(results_map, peak_max_workbook)
