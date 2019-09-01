# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:00:11 2019

@author: HP
"""
import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from return_model import get_factor
from single_factor_test import *

warnings.filterwarnings('ignore')  #将运行中的警告信息设置为“忽略”，从而不在控制台显示

#工作目录，存放代码和因子基本信息
work_dir = os.path.dirname(os.path.dirname(__file__))
#经过预处理后的因子截面数据存放目录
factor_path = os.path.join(work_dir, '因子预处理模块', '因子（已预处理）')
#测试结果图表存放目录
sf_test_save_path = os.path.join(work_dir, '单因子检验')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
plt.rcParams['figure.figsize'] = (15.0, 6.0)  # 图片尺寸设定（宽 * 高 cm^2)

def single_factor_test(factors):
    global sf_test_save_path
    print("\n开始进行T检验和IC检验...")
    test_yearly(factors)   #T检验&IC检验n
    print(f"检验完毕！结果见目录：{sf_test_save_path}")
    print('*'*80)

def layer_division_bt(factors):  
    global work_dir, sf_test_save_path
    start_date='2009-01-23'
    end_date='2019-2-20'
    if_concise = True          #是否进行月频简化回测
    factor_matrix_path = os.path.join(sf_test_save_path, '因子矩阵')
    
    #创建分层回测结果图的存放目录
    if not os.path.exists(os.path.join(sf_test_save_path, '分层回测')):
        os.mkdir(os.path.join(sf_test_save_path, '分层回测'))
    
    #创建因子矩阵文件，为分层回测做准备
    panel_to_matrix(factors)    
    print('因子数据创建完毕')
    pct_chg_nm = get_factor(['PCT_CHG_NM'])['PCT_CHG_NM']
    #对选中的因子或者全部因子遍历
    print("开始进行因子分层回测...")
    for fname in factors:
        openname = fname.replace('/','_div_')
        facdat = pd.read_csv(os.path.join(factor_matrix_path, openname+'.csv'),
                             encoding='gbk', engine='python', index_col=[0])
        facdat.columns = pd.to_datetime(facdat.columns)
        
        s = SingleFactorLayerDivisionBacktest(factor_name=fname, 
                                              factor_data=facdat, 
                                              num_layers=5,
                                              if_concise=if_concise,
                                              start_date=start_date,
                                              end_date=end_date,
                                              pct_chg_nm=pct_chg_nm)
    
        records = s.run_layer_division_bt(equal_weight=True)
        
        plot_layerdivision(records, fname, if_concise)         #绘制分层图
        bar_plot_yearly(records, fname, if_concise)            #绘制分年分层收益柱形图
        plot_group_diff_plot(records, fname, if_concise)       #绘制组1-组5净值图
    
    print(f"分层回测结束！结果见目录：{sf_test_save_path}")
    print('*'*80)
    
def main():
    factors = input("请输入待进行检验的因子（以,分隔），'a'为全部因子：")
    if factors == 'a':
        factors = get_factor_names()
    else:
        factors = factors.split(',')
        
    single_factor_test(factors)
    layer_division_bt(factors)
    
if __name__ == '__main__':
    main()
