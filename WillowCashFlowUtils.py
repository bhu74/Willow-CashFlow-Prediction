#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm
import WillowCashFlowConfig as cfg
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings("ignore")

def get_naive_forecast(series):
    '''
    Naive forecast
    '''
    train = series[:-cfg.PRED_STEPS]
    test = series[-cfg.PRED_STEPS:]
    seasons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    best_seas=1
    best_rmse=99999
    best_mape=99999
    for k in seasons:
        snaive_predict, snaive_future, sn_rmse, sn_mape = naive_seasonal(train, test, k)
        if sn_rmse < best_rmse:
            best_seas = k
            best_rmse = sn_rmse
            best_mape = sn_mape
            best_predict = snaive_predict
            best_future = snaive_future
    return best_seas

def naive_seasonal(train, test, k):
    '''
    Seasonal naive forecast
    '''
    series = pd.concat((train, test))
    forecast_series = series.shift(k)
    forecast_series.dropna(inplace=True)
    snaive_rmse = calculate_rmse(series[-cfg.PRED_STEPS:], forecast_series[-cfg.PRED_STEPS:])
    snaive_mape = calculate_mape(series[-cfg.PRED_STEPS:], forecast_series[-cfg.PRED_STEPS:])
    history = series.values
    y_pred = []
    for i in range(cfg.PRED_STEPS):
        y_pred.append(history[-k])
        history=np.append(history, y_pred)
    return forecast_series, y_pred, snaive_rmse, snaive_mape

def sarimax_configs(seasonal=[1]):
    '''
    Get all configurations for Sarimax model
    '''
    cfg_list = []
    trend = ['n', 'c', 't', 'ct']
    p_params = [0, 1, 2]
    d_params = [0, 1]
    q_params = [0, 1, 2]
    P_params = [0, 1, 2]
    D_params = [0, 1]
    Q_params = [0, 1, 2]
    m_params = seasonal
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for P in P_params:
                    for D in D_params:
                        for Q in Q_params:
                            for m in m_params:
                                for t in trend:
                                    cfg = [(p,d,q), (P,D,Q,m), t]
                                    cfg_list.append(cfg)
    return cfg_list

def find_best_params_sarimax(train, test, cf):
    '''
    Finds the best parameters for a given series for SARIMAX algorithm.
    '''
    bp_rmse = 999999
    bp_mape = 999999
    ord, sord, trnd = cf
    try:
        bp_mod = SARIMAX(train, order=ord, seasonal_order=sord, trend=trnd, enforce_stationarity=False, enforce_invertibility=False)
        bp_res = bp_mod.fit()
        bp_pred = bp_res.predict(start=1, end=(len(train)+len(test)))
        bp_rmse = calculate_rmse(test, bp_pred[-len(test):])
        bp_mape = calculate_mape(test, bp_pred[-len(test):])
        print(' > Model[%s], rmse=%.3f, mape=%.3f' % (cfg, bp_rmse, bp_mape))
    except:
        bp_rmse = 999999
        bp_mape = 999999
    return (cf, bp_rmse, bp_mape)

def get_sarimax_predictions(series, predict_steps, best_param):
    '''
    Make predictions for future timesteps
    '''
    o, so, t = best_param
    history = series
    for h in range(predict_steps):
        f_mod = SARIMAX(history, order=o, seasonal_order=so, trend=t, enforce_stationarity=False, enforce_invertibility=False)
        f_res = f_mod.fit()
        forecast = f_res.get_forecast()
        history = np.append(history, forecast.predicted_mean.tolist()[0])
        series_fitted = f_res.fittedvalues
    return series_fitted, history[-predict_steps:]

def train_and_predict_sarimax(train, test, seasonal=[0]):
    '''
    Train a SARIMAX and predicts the required future timesteps.
    '''
    cfg_list = sarimax_configs(seasonal)

    executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
    tasks = (delayed(find_best_params_sarimax)(train, test, cf) for cf in tqdm(cfg_list))
    scores = executor(tasks)
    scores.sort(key=lambda tup: tup[1])
    best_param, best_rmse, best_mape = scores[0][0], scores[0][1], scores[0][2]
    print(' > Best Model[%s] selected, rmse=%.3f, mape=%.3f' % (best_param, best_rmse, best_mape))
    series = np.append(train, test)
    series_predict, future_values = get_sarimax_predictions(series, cfg.PRED_STEPS, best_param)
    return series_predict, future_values, best_param

def get_region_futuresteps():
    '''
    Prepare the dataframe for predicting future timesteps
    '''
    reg_df = pd.DataFrame()
    yr_ctr = 0
    pred_yrs = []
    pred_mths = []
    mth_ind = int(cfg.PRED_START.split('-')[1][1:])-1
    for i in range(cfg.PRED_STEPS):
        mth_ind += 1
        if mth_ind > 12:
            mth_ind = 1
            yr_ctr += 1
        if mth_ind < 10:
            mth_str = 'M' + '0' + str(mth_ind)
        else:
            mth_str = 'M' + str(mth_ind)
        pred_mths.append(mth_str)
        pred_yrs.append('Y' + str(int(cfg.PRED_START.split('-')[0][1:])+yr_ctr))
    reg_df['Year-mapping'] = pred_yrs
    reg_df['Month-mapping'] = pred_mths
    return reg_df

def get_M01_values(target_data):
    '''
    Compute M01 values at region level, using M02 data
    '''
    train_data = target_data.groupby(['Region-Mapping','Year-mapping', 'Month-mapping']).sum().reset_index()
    train_data['Values'] = train_data['Adj_Value_Sum']
    train_data.loc[train_data['Month-mapping'] == 'M02', ['Values']] = train_data['Adj_Value_Sum'].apply(lambda x: float(x)/2)
    m02_df = train_data[train_data['Month-mapping'] == 'M02'].copy()
    m02_df['Month-mapping'] = 'M01'
    train_df = pd.concat([train_data,m02_df])
    train_df = train_df.sort_values(['Region-Mapping', 'Year-mapping', 'Month-mapping']).reset_index()
    train_df.drop(columns=['index', 'Adj_Value_Sum'], inplace=True)
    return train_data, train_df

def get_region_predictions(train_df, reg_df):
    '''
    Generate region level predictions for future timesteps
    '''
    reg_out = pd.DataFrame(columns=train_df.columns)
    for r in train_df['Region-Mapping'].unique():
        print('-------- Region: ', r, '--------')
        series = train_df[train_df['Region-Mapping'] == r]['Values']
        train = series[:-cfg.PRED_STEPS]
        test = series[-cfg.PRED_STEPS:]
        best_seas = get_naive_forecast(series)
        seasonal=[best_seas]
        sarima_predict, future_values, best_param = train_and_predict_sarimax(train, test, seasonal)
        reg_df['Region-Mapping'] = r
        reg_df['Values'] = future_values
        reg_out=pd.concat([reg_out, reg_df])
    return reg_out

def get_market_predictions(target_df, reg_total, mkt_df):
    '''
    Generate market level predictions for future timesteps
    '''
    # Determine the ratio of the market level value to total region level value in training data
    target_df = target_df.merge(reg_total, how='outer', on=['Region-Mapping', 'Year-mapping', 'Month-mapping'])
    target_df.drop(columns=['Values'], inplace=True)
    target_df['Ratio'] = target_df['Adj_Value_Sum_x'] / target_df['Adj_Value_Sum_y']
    target_df.rename(columns={'Adj_Value_Sum_x':'Value', 'Adj_Value_Sum_y':'Region_Total'}, inplace=True)

    # Prepare data to predict the ratio of the market for future timesteps
    X = target_df.copy()
    X.drop(columns=['Account-Mapping', 'Acc2-Mapping', 'Version', 'Value', 'Region_Total' ], inplace=True)
    X['Year'] = X['Year-mapping'].apply(lambda x: int(x[1:]))
    mth_ohe = pd.get_dummies(X['Month-mapping'])
    reg_ohe = pd.get_dummies(X['Region-Mapping'])
    mkt_ohe = pd.get_dummies(X['Market-Mapping'])
    X = pd.concat([X, mth_ohe, reg_ohe, mkt_ohe], axis=1)
    y = X['Ratio']
    X.drop(columns=['Year-mapping', 'Month-mapping', 'Region-Mapping', 'Market-Mapping', 'Ratio'], inplace=True)

    # Prepare output dataframe to generate market level predictions
    inv_out = pd.DataFrame(columns=mkt_df.columns)
    mkt_df['Account-Mapping'] = target_df['Account-Mapping']
    mkt_df['Acc2-Mapping'] = 'Working Capital'
    mkt_df['Version'] = 'ACT'
    for r in target_df['Region-Mapping'].unique():
        for m in target_df[target_df['Region-Mapping'] == r]['Market-Mapping'].unique():
            mkt_df['Region-Mapping'] = r
            mkt_df['Market-Mapping'] = m
            inv_out = pd.concat([inv_out, mkt_df])

    # Prepare test dataframe to generate market level predictions
    x_pred = inv_out.copy()
    x_pred.drop(columns=['Account-Mapping', 'Acc2-Mapping', 'Version'], inplace=True)
    x_pred['Year'] = x_pred['Year-mapping'].apply(lambda x: int(x[1:]))
    mth_ohe = pd.get_dummies(x_pred['Month-mapping'])
    reg_ohe = pd.get_dummies(x_pred['Region-Mapping'])
    mkt_ohe = pd.get_dummies(x_pred['Market-Mapping'])
    x_pred = pd.concat([x_pred, mth_ohe, reg_ohe, mkt_ohe], axis=1)
    x_pred.drop(columns=['Year-mapping', 'Month-mapping', 'Region-Mapping', 'Market-Mapping'], inplace=True)

    cols = list(set(X.columns) - set(x_pred.columns))
    x_pred = x_pred.assign(**dict.fromkeys(cols, 0))
    x_pred = x_pred[X.columns]

    # Fit regression model to predict market level ratios for future timesteps
    seed = 43
    gb_opt = GradientBoostingRegressor(n_estimators = 3000, learning_rate = 0.05,
                                   max_depth = 4, max_features = 'sqrt',
                                   min_samples_leaf = 15, min_samples_split = 10,
                                   loss = 'huber', random_state = seed)
    gb_opt.fit(X, y)
    y_pred = gb_opt.predict(x_pred)
    inv_out['Ratio'] = y_pred
    return inv_out

def calculate_rmse(y_true, y_pred):
    '''
    Calculate RMSE
    '''
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mape(y_true, y_pred):
    '''
    Calculate MAPE
    '''
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    try:
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    except:
        return 0
