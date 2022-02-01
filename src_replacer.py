#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

#takes a list of values in the key column to delete all of the records in the old file with
#the SRC records from the new file
#pass a value_list of False in order to replace all of SRCs in the new records
def replace_srcs(old_file, new_file, value_list, out_path, key='SRC',):
    #replace with new rows
    new_df=pd.read_csv(new_file, sep='\t')
    if not value_list:
        value_list=new_df[key].unique()
    old_df=pd.read_csv(old_file, sep='\t')
    #removes old rows
    old_df=old_df[~old_df[key].isin(value_list)]

    #remove any rows from new_df that don't have a value from value_list in the key column
    new_df=new_df[new_df[key].isin(value_list)]
    out_df = pd.concat([old_df, new_df], ignore_index=True)
    out_df.to_csv(out_path, sep='\t', index=False)
    print("SRCs being replaced: ", value_list)

two_periods = "/home/craig/runs/results_no_post_demand/results.txt"
def test_1():
    three_periods = "/home/craig/runs/test-run/results.txt"
    out = "/home/craig/runs/replaced_1.txt"
    replace_srcs(three_periods, two_periods, ['01205K000'], out)

def test_2():
    big_phases = "/home/craig/runs/big_test/results_no_truncate_and_1_supply.txt"
    out = "/home/craig/runs/replaced_2.txt"
    replace_srcs(big_phases, two_periods, ['01205K000'], out)

def test_3():
    big_phases = "/home/craig/runs/big_test/results_no_truncate_and_1_supply.txt"
    no_truncation = "/home/craig/runs/big_test/results.txt"
    out = "/home/craig/runs/replaced_3.txt"
    replace_srcs(big_phases, no_truncation, ['01205K000', "01225K000"], out)
    #Try passing False as the value_list to replace all SRCs in the new
    replace_srcs(big_phases, "/home/craig/runs/big_test/results_4_SRCs.csv", False, out)
    
def do_tests():  
    test_1()
    test_2()
    test_3()

do_tests()


