# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:23:06 2019

@author: HP
"""
import os
import warnings
warnings.filterwarnings('ignore')  #将运行中的警告信息设置为“忽略”，从而不在控制台显示
from index_enhance import *

#工作目录，存放代码
work_dir = os.path.dirname(os.path.dirname(__file__))

def main(method):
    if method == 's':
        method_name = 'stratified_sample'
    elif method == 'l':
        method_name = 'linear_programming'
        
    factors_to_concat = {
            'mom': ['M_reverse_180', 'M_reverse_20', 'M_reverse_60'],
            'liq_barra': ['STOA_Barra', 'STOM_Barra', 'STOQ_Barra'],
            'vol': ['std_1m', 'std_3m', 'std_6m', 'std_12m'],
            'growth': ['Profit_G_q', 'Sales_G_q', 'ROE_G_q'],
            'lev': ['BLEV_barra', 'DTOA_barra', 'MLEV_barra'],
            }
    factors_ortho = {
            'vol_con_equal':['mom_con_equal', 'liq_barra_con_equal', 'beta_barra'],
            'ROE_q': ['EP'], 
                     }    
    methods = {
          'linear_programming': 
                {'factors': ['BP', 'ROE_q', 'mom_con_equal_ortho', 
                             'liq_barra_con_equal_ortho', 'growth_con_equal'],
                 'risk_factors': ['beta_barra_ortho', 'LNCAP_barra', 
                                  'nonlinearsize_barra'],
                 'window': 24, 
                 'half_life': None},
           'stratified_sample': 
                {'factors': ['BP', 'growth_con_equal','lev_con_equal',
                             'ROE_q', 'nonlinearsize_barra'],
                 'risk_factors': None,
                 'window': 12, 
                 'half_life': 6},
          }
    
    start_date = '2011-01-20'
    end_date = '2019-02-27'
    benchmark = '000300.SH'
    factors = methods[method_name]['factors']
    risk_factors = methods[method_name]['risk_factors']
    print('开始运行模型...')   
    print('*'*80)     
    pctchgnm = get_factor(['PCT_CHG_NM'])['PCT_CHG_NM']
    index_wt = get_stock_wt_in_index(benchmark)    
    mut_codes = index_wt.index.intersection(pctchgnm.index)
    print('开始进行因子合成与正交处理...')
    factor_process(method, factors_to_concat, factors_ortho, 
                   index_wt, mut_codes, factors, risk_factors)
    print('因子处理完成！')
    print('*'*80)
    print('开始运行指数增强模型...')
    index_enhance_model(method, benchmark, start_date, end_date, methods)
    
if __name__ == '__main__':
    method = input("请选择指数增强模型方法（'l'-线性规划; 's'-分层抽样）: ")
    if method not in ('l', 's'):
        raise TypeError(f"暂不支持的方法：{method}, 请重新运行并输入")
    main(method)
