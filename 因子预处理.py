# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 09:47:54 2019

@author: admin
"""
import os
import pandas as pd
from factor_data_preprocess import *

file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                         '因子预处理模块', '因子')             #原始因子数据所在目录
save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                         '因子预处理模块', '因子（已预处理）')   #预处理后因子数据保存目录
industry_benchmark = 'zx'               #行业基准（用于缺失值填充和中性化）
                                        #zx - 中信； sw - 申万
                                                                                
#基准信息所在列名（分别对应：
#code - 证券wind代码；name - 证券简称；ipo_date - 上市日期；
#industry_zx - 中信一级行业；industry_sw - 申万一级行业；
#MKT_CAP_FLOAT - 流通市值；is_open1 - 当日是否开盘；
#PCT_CHG_NM - 下个月的月收益率
info_cols = ['code', 'name', 'ipo_date', 'industry_zx', 
             'industry_sw', 'MKT_CAP_FLOAT', 
             'is_open1', 'PCT_CHG_NM']

def main(fpath, factor_names=None):
    """
    输入： 需要进行预处理的因子名称（可为1个或多个，默认为对所有因子进行预处理）
    输出： 预处理后的因子截面数据（如2009-01-23.csv文件）
    
    对指定的原始因子数据进行预处理
    顺序：缺失值填充、去极值、中性化、标准化
    （因输入的截面数据中所含财务类因子默认已经过
    财务日期对齐处理，故在此不再进行该步处理）
    """
    global file_path, save_path, industry_benchmark 
    #读取原始因子截面数据
    data = pd.read_csv(os.path.join(file_path, fpath), engine='python',
                       encoding='gbk', index_col=[0])
    #根据输入的因子名称将原始因子截面数据分割
    data_to_process, data_unchanged = get_factor_data(data, factor_names)

    #预处理步骤依次进行
    data_to_process = fill_na(data_to_process, industry_benchmark)      #缺失值填充
    data_to_process = winsorize(data_to_process)                        #去极值
    data_to_process = neutralize(data_to_process, industry_benchmark)   #中性化
    data_to_process = standardize(data_to_process)                         #标准化

    #合并生成经过处理后的总因子文件
    if len(data_unchanged) > 0:
        data_final = pd.concat([data_to_process, data_unchanged.loc[data_to_process.index]], axis=1)
    else:
        data_final = data_to_process
    data_final.index = range(1, len(data_final)+1)
    data_final.index.name = 'No'
    data_final.to_csv(os.path.join(save_path, fpath), encoding='gbk')   

if __name__ == '__main__':
    #创建处理后因子的存放目录
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    #收集需要处理的因子名称
    factor_names = input("请输入需处理的因子名称（请使用英文逗号','分隔多个因子名称，输入'a'代表全部处理）：")
    factor_names = process_input_names(factor_names)
        
    #对所有横截面数据进行遍历（2009-01至2019-01每个月月末（交易日））
    for fpath in os.listdir(file_path)[:]:
        main(fpath, factor_names)
    print('因子截面数据已全部处理！')
