# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 09:47:11 2019

@author: admin
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

info_cols = ['code', 'name', 'ipo_date', 'industry_zx', 
             'industry_sw', 'MKT_CAP_FLOAT', 
             'is_open1', 'PCT_CHG_NM']

def get_factor_data(datdf, names=None):
    """
    根据输入的因子名称将原始因子截面数据分割
    """
    global info_cols
    if names:
        try:
            fac_names = [fac.lower() for fac in datdf.columns]
            idx = [fac_names.index(fac_name.lower()) for fac_name in names]
        except:
            msg = "请重新确认因子名称是否正确"
            raise Exception(msg)
    
    #对截面数据中，基准信息列存在空缺的股票（行）进行删除处理
    cond = ~pd.isnull(datdf['MKT_CAP_FLOAT'])
    for col in info_cols:
        if col == 'PCT_CHG_NM':
            if pd.isnull(datdf['PCT_CHG_NM']).all():
                continue
        if col != 'MKT_CAP_FLOAT':
            cond &= ~pd.isnull(datdf[col])
    datdf = datdf.loc[cond]
    
    #将原截面数据按照预处理与否分别划分，返回需处理和不需处理2个因子截面数据
    if names is None:
        return datdf, pd.DataFrame()
    else:
        dat_to_process = datdf.iloc[:, idx]
        dat_to_process = pd.merge(datdf[info_cols], dat_to_process,
                                  left_index=True, right_index=True)
        unchanged_cols = sorted(set(datdf.columns) - set(dat_to_process.columns))
        dat_unchanged = datdf[unchanged_cols]
    return dat_to_process, dat_unchanged

def fill_na(data, ind='zx'):
    """
    缺失值填充：缺失值少于10%的情况下使用行业中位数代替
    """
    global info_cols
    datdf = data.copy()
    if ind == 'sw':
        datdf = datdf.loc[~pd.isnull(datdf['industry_sw']), :]
    
    facs_to_fill = datdf.columns.difference(set(info_cols))
    facs_to_fill = [fac for fac in facs_to_fill            #筛选缺失值少于10%的因子
                    if pd.isnull(datdf[fac]).sum() / len(datdf) <= 0.1]
    
    for fac in facs_to_fill:
        try:
            fac_median_by_ind = datdf[[f'industry_{ind}', fac]].groupby(f'industry_{ind}').median() 
        except:
            #处理原始数据中的负无穷（-Inf)                
            datdf.loc[:, fac] = datdf[[fac]].applymap(coerce_numeric)
            fac_median_by_ind = datdf[[f'industry_{ind}', fac]].groupby(f'industry_{ind}').median()
        fac_ind_map = fac_median_by_ind.to_dict()[fac]
        fac_to_fill = datdf.loc[pd.isnull(datdf[fac]), [f'industry_{ind}', fac]]
        fac_to_fill.loc[:, fac] = fac_to_fill[f'industry_{ind}'].map(fac_ind_map)
        datdf.loc[fac_to_fill.index, fac] = fac_to_fill[fac].values

    #针对sw行业存在缺失值的情况
    if len(datdf) < len(data):
        idx_to_append = data.index.difference(datdf.index)
        datdf = pd.concat([datdf, data.loc[idx_to_append,:]])
        datdf.sort_index()
    
    return datdf

def coerce_numeric(s):
    try:
        return float(s)
    except:
        return -np.inf

def winsorize(data, n=5):
    """
    去极值：5倍中位数标准差法（5mad）
    """
    global info_cols
    
    datdf = data.copy()
    
    if_contain_na = pd.isnull(datdf).sum().sort_values(ascending=True)
    facs_to_remove = if_contain_na.loc[if_contain_na > 0].index.tolist()
    if 'PCT_CHG_NM' in facs_to_remove:
        facs_to_remove.remove('PCT_CHG_NM')
        
    facs_to_win = datdf.columns.difference(set(info_cols)).difference(set(tuple(facs_to_remove)))
    
    dat_win = datdf[facs_to_win]
    fac_vals = dat_win.values
    dm = np.nanmedian(fac_vals, axis=0)
    dm1 = np.nanmedian(np.abs(fac_vals - dm), axis=0)
    if 0 in (dm + n*dm1): 
        #针对存在去极值后均变为零的特殊情况（2009-05-27-'DP')
        cut_points = [i for i in np.argwhere(dm1 == 0)[0]]
        #提取对应列，对其不进行去极值处理
        facs_unchanged = [facs_to_win[cut_points[i]] for i in range(len(cut_points))] 
        #仅对剩余列进行去极值处理
        facs_to_win_median = facs_to_win.difference(set(tuple(facs_unchanged)))
        
        dat_win_median = datdf[facs_to_win_median]
        fac_median_vals = dat_win_median.values
        dmed = np.nanmedian(fac_median_vals, axis=0)
        dmed1 = np.nanmedian(np.abs(fac_median_vals - dmed), axis=0)
        dmed = np.repeat(dmed.reshape(1,-1), fac_median_vals.shape[0], axis=0)
        dmed1 = np.repeat(dmed1.reshape(1,-1), fac_median_vals.shape[0], axis=0)
        
        fac_median_vals = np.where(fac_median_vals > dmed + n*dmed1, dmed+n*dmed1, 
              np.where(fac_median_vals < dmed - n*dmed1, dmed - n*dmed1, fac_median_vals))
        res1 = pd.DataFrame(fac_median_vals, index=dat_win_median.index, columns=dat_win_median.columns)
        res2 = datdf[facs_unchanged]
        res = pd.concat([res1, res2], axis=1)
    else:
        dm = np.repeat(dm.reshape(1,-1), fac_vals.shape[0], axis=0)
        dm1 = np.repeat(dm1.reshape(1,-1), fac_vals.shape[0], axis=0)
        fac_vals = np.where(fac_vals > dm + n*dm1, dm+n*dm1, 
              np.where(fac_vals < dm - n*dm1, dm - n*dm1, fac_vals))
        res = pd.DataFrame(fac_vals, index=dat_win.index, columns=dat_win.columns)

    datdf[facs_to_win] = res
    return datdf  
    
def neutralize(data, ind='zx'):
    """
    中性化：因子暴露度对行业哑变量（ind_dummy_matrix）和对数流通市值（lncap_barra）
            做线性回归, 取残差作为新的因子暴露度
    """
    global info_cols
    datdf = data.copy()
    if ind == 'sw':
        datdf = datdf.loc[~pd.isnull(datdf['industry_sw']), :]
    
    cols_to_neu = datdf.columns.difference(set(info_cols))
    y = datdf[cols_to_neu]
    y = y.dropna(how='any', axis=1)
    cols_neu = y.columns
        
    lncap = np.log(datdf[['MKT_CAP_FLOAT']])
    ind_dummy_matrix = pd.get_dummies(datdf[f'industry_{ind}'])
    X = pd.concat([lncap, ind_dummy_matrix], axis=1)
    
    model = LinearRegression(fit_intercept=False)
    res = model.fit(X, y)
    coef = res.coef_
    residue = y - np.dot(X, coef.T)
    
    assert len(datdf.index.difference(residue.index)) == 0

    datdf.loc[residue.index, cols_neu] = residue
    return datdf
    
def standardize(data):
    """
    标准化：Z-score标准化方法，减去均值，除以标准差
    """
    global info_cols
    
    datdf = data.copy()
    facs_to_sta = datdf.columns.difference(set(info_cols))
    
    dat_sta = datdf[facs_to_sta].values
    dat_sta = (dat_sta - np.mean(dat_sta, axis=0)) / np.std(dat_sta, axis=0)
    
    datdf.loc[:, facs_to_sta] = dat_sta
    return datdf

def process_input_names(factor_names):
    if factor_names == 'a':
        factor_names = None
    else:
        factor_names = [f.replace("'","").replace('"',"") for f in factor_names.split(',')]
    return factor_names



        
