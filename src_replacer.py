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

def resource_path(repo_path, filename):
    return repo_path+"resources/"+filename
def output_path(repo_path, filename):
    return repo_path+"test-output/"+filename

def test_1(repo_path):  
    three_periods = resource_path(repo_path, "results_small.txt")
    two_periods = resource_path(repo_path, "results_no_post_demand.txt")
    out = output_path(repo_path, "replaced_1.txt")
    replace_srcs(three_periods, two_periods, ['01205K000'], out)

def test_2(repo_path):
    big_phases = resource_path(repo_path, 
                               "results_no_truncate_and_1_supply.txt")
    two_periods = resource_path(repo_path, "results_no_post_demand.txt")
    out = output_path(repo_path, "replaced_2.txt")
    replace_srcs(big_phases, two_periods, ['01205K000'], out)

def test_3(repo_path):
    big_phases = resource_path(repo_path, "results_no_truncate_and_1_supply.txt")
    no_truncation = resource_path(repo_path, "results.txt")
    out = output_path(repo_path, "replaced_3.txt")
    replace_srcs(big_phases, no_truncation, ['01205K000', "01225K000"], out)
    #Try passing False as the value_list to replace all SRCs in the new
    four_srcs = resource_path(repo_path, "results_4_SRCs.csv")
    out_2 = output_path(repo_path, "replaced_4.txt")
    replace_srcs(big_phases, four_srcs, False, out_2)
    
def do_tests(repo_path):  
    test_1(repo_path)
    test_2(repo_path)
    test_3(repo_path)


