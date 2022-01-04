#!/usr/bin/env python
# coding: utf-8

# # Willow Cash Flow Prediction - Discovery

# In[1]:


# Import necessary libraries
import numpy as np
import pandas as pd
import WillowCashFlowConfig as cfg
from WillowCashFlowUtils import *


# In[2]:


# Read input training and test files. Prepare dataframe for output
data = pd.read_csv(cfg.INPUT_FILE)
test_set=pd.read_csv(cfg.INPUT_TEST_FILE)
targets = [cfg.INVENTORY, cfg.PAYABLES_ACCRUALS, cfg.RECEIVABLES, cfg.RETURNS_REBATES]
reg_df = get_region_futuresteps()
mkt_df = reg_df.copy()


# In[3]:


# For each of the target variables, generate region level and market level predictions
allmkt_df = pd.DataFrame()
for t in targets:
    print('========= Predicting for ', t.upper(), ' =========')
    target_data = data[(data['Account-Mapping'] == t) & (data['Version'] == 'ACT')]
    reg_total, train_df = get_M01_values(target_data)
    reg_out = get_region_predictions(train_df, reg_df)
    mkt_out = get_market_predictions(target_data, reg_total, mkt_df)
    allmkt_df = pd.concat([allmkt_df, mkt_out])


# In[4]:


# Prepare output data and write submission file 
allmkt_df = allmkt_df.merge(reg_out, how='outer', on=['Region-Mapping', 'Year-mapping', 'Month-mapping'])
allmkt_df['Adj_Value_Sum'] = allmkt_df['Ratio'] * allmkt_df['Values']
test_set = test_set.merge(allmkt_df, how='left', on=['Account-Mapping', 'Region-Mapping', 'Year-mapping', 'Month-mapping', 'Market-Mapping'])
req_cols = ['Account-Mapping', 'Acc2-Mapping_x', 'Version_x', 'Year-mapping', 'Month-mapping', 'Region-Mapping', 'Market-Mapping', 'Adj_Value_Sum_y']
test_set = test_set[req_cols]
test_set.fillna(0, inplace=True)
test_set.rename(columns={'Acc2-Mapping_x': 'Acc2-Mapping', 'Version_x': 'Version',                          'Adj_Value_Sum_y':'Adj_Value_Sum'}, inplace=True)
test_set.to_csv(cfg.OUTPUT_FILE, index=False)


# In[ ]:




