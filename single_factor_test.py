# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 08:25:00 2019

@author: HP
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pandas.tseries.offsets as toffsets
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from copy import deepcopy
warnings.filterwarnings('ignore')  #将运行中的警告信息设置为“忽略”，从而不在控制台显示

__all__ = [test_yearly, get_factor_names, panel_to_matrix, Backtest_stock, 	 
        SingleFactorLayerDivisionBacktest, plot_layerdivision, bar_plot_yearly, 
        plot_group_diff_plot]

#工作目录，存放代码和因子基本信息
work_dir = os.path.dirname(os.path.dirname(__file__))
#经过预处理后的因子截面数据存放目录
factor_path = os.path.join(work_dir, '因子预处理模块', '因子（已预处理）')
#测试结果图表存放目录（如无则自动生成）
sf_test_save_path = os.path.join(work_dir, '单因子检验')

industry_benchmark = 'zx'     #行业基准-中信一级行业

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
plt.rcParams['figure.figsize'] = (15.0, 6.0)  # 图片尺寸设定（宽 * 高 cm^2)
plt.rcParams['font.size'] = 15                # 字体大小
num_layers = 5                                # 设置分层层数
tick_spacing1 = 9                              # 设置画图的横轴密度
tick_spacing2 = 150                            # 设置横坐标密度

def get_factor_names():
    global work_dir
    factor_info = pd.read_excel(os.path.join(work_dir, '新沃因子列表.xlsx'), 
                                encoding='gbk', sheetname=1, index_col=[0])   
    return factor_info['因子名称'].values.tolist()

def regress(y, X, w=1, intercept=False):
    if intercept:
        X = sm.add_constant(X)
    model = sm.WLS(y, X, weights=w)
    result = model.fit()
    
    ts, params = result.tvalues, result.params
    ts.index = X.columns
    params.index = X.columns
    resid = y - np.dot(X, params.T)   
    return ts, params, resid

def get_ind_mktcap_matrix(datdf, ind=True, mktcap=True):
    global industry_benchmark
    if mktcap:
        lncap = np.log(datdf['MKT_CAP_FLOAT'])
        lncap.name = 'ln_mkt_cap'
    else:
        lncap = pd.DataFrame()
    if ind:
        ind_dummy_matrix = pd.get_dummies(datdf[f'industry_{industry_benchmark}'])
    else:
        ind_dummy_matrix = pd.DataFrame()
    
    return pd.concat([lncap, ind_dummy_matrix], axis=1)

def get_ic(datdf, fac_name, neutralize=False):
    pctchgnm = datdf['PCT_CHG_NM']
    facdat = datdf[fac_name]
    if neutralize:
        ind_mktcap_matrix = get_ind_mktcap_matrix(facdat)
        _, _, facdat = regress(facdat, ind_mktcap_matrix)
    dat = pd.concat([facdat, pctchgnm], axis=1)
    ic = dat.corr().iat[0,1]
    return ic

def regression_summary(ts, params, ics):
    res = {}
    res['t值绝对值平均值'] = np.mean(np.abs(ts))           #t值绝对值平均值
    res['t值绝对值>2概率'] = len(ts[np.abs(ts) > 2]) / len(ts)  #t值绝对值>2概率
     
    res['因子收益平均值'] = np.mean(params)          #因子收益平均值
    res['因子收益标准差'] = np.std(params)           #因子收益标准差
    res['因子收益t值'] = stats.ttest_1samp(params, 0).statistic   #因子收益t值
    res['因子收益>0概率'] = len(params[params > 0]) / len(params) #因子收益>0概率
    
    res['IC平均值'] = np.mean(ics)         #IC平均值
    res['IC标准差'] = np.std(ics)          #IC标准差
    res['IRIC'] = res['IC平均值'] / res['IC标准差']          #IRIC
    res['IC>0概率'] = len(ics[ics>0]) / len(ics)   #IC>0概率
    return pd.Series(res)

def t_ic_test(datpanel, factor_name):
    t_series, fret_series, ic_series = pd.Series(), pd.Series(), pd.Series()
    for date, datdf in datpanel.items():
        w = np.sqrt(datdf['MKT_CAP_FLOAT'])
        y = datdf['PCT_CHG_NM']
        
        X = datdf[factor_name]
        ind_mktcap_matrix = get_ind_mktcap_matrix(datdf)
        X = pd.concat([X, ind_mktcap_matrix], axis=1)  

        ts, f_rets, _= regress(y, X, w)
        
        t_series[date] = ts[factor_name]
        fret_series[date] = f_rets[factor_name]

        ic = get_ic(datdf, factor_name)
        ic_series[date] = ic
    
    summary = regression_summary(t_series.values, fret_series.values, 
                                 ic_series.values)
    return summary, t_series, fret_series, ic_series

def get_datdf_in_year(year):
    global factor_path
    dates = []
    for f in os.listdir(factor_path):
        curdate = f.split('.')[0]
        curyear = pd.to_datetime(curdate).year
        if curyear == year:
            dates.append(curdate)
        
    datpanel = {}
    for date in dates:     
        datdf = pd.read_csv(os.path.join(factor_path, date+'.csv'), 
                            engine='python', encoding='gbk', index_col=[0]) 
        date = pd.to_datetime(date)
        datpanel[date] = datdf
    return datpanel

def get_test_result(factors, datpanel):
    res = pd.DataFrame()
    ts_all, frets_all, ics_all = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for factor_name in factors:
        cur_fac_res, ts, frets, ics = t_ic_test(datpanel, factor_name)
        col_name = factor_name.replace('/','_div_') if '/' in factor_name else factor_name
        
        cur_fac_res.name = col_name
        ts.name = col_name
        frets.name = col_name
        ics.name = col_name
        
        res = pd.concat([res, cur_fac_res], axis=1)
        ts_all = pd.concat([ts_all, ts], axis=1)
        frets_all = pd.concat([frets_all, frets], axis=1)
        ics_all = pd.concat([ics_all, ics], axis=1)
        
    ts_all = ts_all.sort_index()
    frets_all = frets_all.sort_index()
    ics_all = ics_all.sort_index()
    return res, ts_all, frets_all, ics_all
        
def test_yearly(factors=None, start_year=2009, end_year=2018):
    global factor_path, sf_test_save_path, info_cols
    years = range(start_year, end_year+1)
    test_result = {}
    ts_all, frets_all, ics_all = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for year in years:
        datpanel = get_datdf_in_year(year)
        if factors is None:
            factors = get_factor_names()
        cur_test_res, ts, frets, ics = get_test_result(factors, datpanel)
        test_result[year] = cur_test_res
        
        ts_all = pd.concat([ts_all, ts])
        frets_all = pd.concat([frets_all, frets])
        ics_all = pd.concat([ics_all, ics])
    
    #存储所有t值、因子收益率、ic值时间序列数据
    for save_name, df in zip(['t_value', 'factor_return', 'ic'], 
                             [ts_all, frets_all, ics_all]):
        df.to_csv(os.path.join(sf_test_save_path, save_name+'.csv'),
                  encoding='gbk')
        
    #存储检验结果表格
    test_result = pd.Panel(test_result)
    test_result = test_result.swapaxes(2, 0)
    test_result = test_result.swapaxes(1, 2)
    test_result.to_excel(os.path.join(sf_test_save_path, 'T检验&IC检验结果.xlsx'),
                         encoding='gbk')
    
    #绘制单因子检验图，并进行存储
    plot_test_figure(ts_all, frets_all, ics_all, save=True)

def plot_test_figure(ts, frets, ics, save=True):
    global sf_test_save_path
    ts = np.abs(ts)
    factors = ts.columns
    fig_save_path = os.path.join(sf_test_save_path, 'T检验与IC检验结果图')
    if not os.path.exists(fig_save_path):
        os.mkdir(fig_save_path)
    for fac in factors:
        t, fret, ic = ts[fac], frets[fac], ics[fac]
        sharedx = [str(d)[:10] for d in t.index]
        
        fig, axes = plt.subplots(3,1, sharex=True)
        fig.suptitle(fac)
        bar_plot(axes[0], sharedx, t.values, 't value绝对值')
        bar_plot(axes[1], sharedx, fret.values, '因子收益率')
        bar_plot(axes[2], sharedx, ic.values, 'IC')
        fig.savefig(os.path.join(fig_save_path, fac+'.png'))
        plt.close()

def bar_plot(ax, x, y, title):
    global tick_spacing1
    ax.bar(x, y)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing1))
    ax.set_title(title)

###净值回测
class Backtest_stock:
    def __init__(self, *, market_data, start_date, end_date, benchmarkdata=None,
                 stock_weights=None, initial_capital=100000000, tradedays=None, 
                 refreshdays=None, trade_signal=None, stk_basic_data=None, 
                 rf_rate=0.04, use_pctchg=False, **kwargs):
        if stock_weights is None:
            if trade_signal is None:
                raise AttributeError("PARAM::stock_weights must be passed in.")
                #证券权重数据和交易信号均为空时，报错（两者至少传入其一）
        self.use_pctchg = use_pctchg            #是否采用pctchg进行回测净值计算     
        if stk_basic_data is None:
            self.stock_pool = stock_weights.index    #股票池
        else:
            self.stock_pool = stk_basic_data.index
            
        self.stock_weights = stock_weights           #各组的证券权重
        self.market_data = market_data               
        #行情数据（全A股复权收盘价 或 A股日涨跌幅）
        self.benchmark_data = benchmarkdata         #基准（000300或000905日涨跌幅）
        
        self.start_date = start_date                 #回测开始日期
        self.end_date = end_date                     #回测结束日期
        self.capital = initial_capital               #可用资金
        self.net_value = initial_capital             #账户市值
        
        self.curdate = None                #当前调仓交易日对应日期
        self.lstdate = None                #上一个调仓交易日对应日期
        
        if tradedays:                           #回测期内所有交易日list
            tradedays = pd.to_datetime(tradedays)
        else:
            tradedays = pd.to_datetime(self.market_data.columns)
        self.tradedays = sorted(tradedays)
            
        if refreshdays:         #回测期内所有调仓交易日list（默认为每个月首个交易日）
            self.refreshdays = refreshdays
        else:
            self.refreshdays = list(self.get_refresh_days())
        
        self.position_record = {}  #每个交易日持仓记录
        self.portfolio_record = {} #组合净值每日记录
        self.rf_rate = rf_rate
        
    def get_refresh_days(self):
        """
        获取调仓日期（回测期内的每个月首个交易日）
        """
        tdays = self.tradedays
        sindex = self._get_date_idx(self.start_date)
        eindex = self._get_date_idx(self.end_date)
        tdays = tdays[sindex:eindex+1]
        return (nd for td, nd in zip(tdays[:-1], tdays[1:]) 
                if td.month != nd.month)
    
    def _get_date_idx(self, date):
        """
        返回传入的交易日对应在全部交易日列表中的下标索引
        """
        datelist = list(self.tradedays)
        date = pd.to_datetime(date)
        try:
            idx = datelist.index(date)
        except ValueError:
            datelist.append(date)
            datelist.sort()
            idx = datelist.index(date)
            if idx == 0:
                return idx + 1
            else:
                return idx - 1
        return idx
    
    def _get_stocks_weights(self, date):
        """
        根据传入的交易日日期（当月第一个交易日）获取对应
        前一截面（上个月最后一个交易日）的该层对应的各股票权重
        """
        idx = self._get_date_idx(date)
        date = self.tradedays[idx-1]
        cur_stk_weights = self.stock_weights.loc[:, date]
        return cur_stk_weights.dropna()
    
    def run_backtest(self):
        """
        回测主函数
        """
        start_idx = self._get_date_idx(self.start_date)
        end_idx = self._get_date_idx(self.end_date)
        
        hold = False
        for date in self.tradedays[start_idx:end_idx+1]:
            #对回测期内全部交易日遍历，每日更新净值
            if date in self.refreshdays:
                #如果当日为调仓交易日，则进行调仓
                hold = True
                idx = self.refreshdays.index(date)
                if idx == 0:
                    #首个调仓交易日
                    self.curdate = date
                self.lstdate, self.curdate = self.curdate, date
                
                if not self.use_pctchg:
                    stocks_to_buy = self._get_stocks_weights(date)
                    if len(stocks_to_buy) > 0:
                        #采用复权价格回测的情况下, 如果待买入股票列表非空，则进行调仓交易
                        self.rebalance(stocks_to_buy)

            if hold:
                #在有持仓的情况下，对净值每日更新计算
                self.update_port_netvalue(date)
        
        #回测后进行的处理
        self.after_backtest()
    
    def after_backtest(self):
        #主要针对净值记录格式进行调整，将pctchg转换为净值数值；
        #同时将持仓记录转化为矩
        self.portfolio_record = pd.DataFrame(self.portfolio_record, index=[0]).T
        if self.use_pctchg:
            self.portfolio_record.columns = ['netval_pctchg']
            self.portfolio_record['net_value'] = self.capital * \
                    (1 + self.portfolio_record['netval_pctchg']).cumprod()
            self.position_record = self.stock_weights
        else:
            self.portfolio_record.columns = ['net_value']
            self.position_record = pd.DataFrame.from_dict(self.position_record)
    
    def _get_latest_mktval(self, date):
        """
        获取传入交易日对应持仓市值
        """
        holdings = self.position_record[self.lstdate].items()
        holding_codes = [code for code, num in holdings]
        holding_nums = np.asarray([num for code, num in holdings])
        latest_price = self.market_data.loc[holding_codes, date].values
        holding_mktval = np.sum(holding_nums * latest_price)
        return holding_mktval
    
    def cal_weighted_pctchg(self, date):
        weights = self._get_stocks_weights(self.curdate)
        weights /= np.sum(weights)
        codes = weights.index
        pct_chg = self.market_data.loc[codes, date].values
        return codes, np.nansum(pct_chg * weights.values)
    
    def update_port_netvalue(self, date):
        """
        更新每日净值
        """
        if self.use_pctchg:
            stk_codes, cur_wt_pctchg = self.cal_weighted_pctchg(date)
            self.portfolio_record[date] = cur_wt_pctchg
        else:
            holding_mktval = self._get_latest_mktval(date)
            total_val = self.capital + holding_mktval
            self.portfolio_record[date] = total_val
    
    def rebalance(self, stocks_data):
        """
        调仓，实际将上一交易日对应持仓市值加入到可用资金中
        """
        if self.position_record:
            self.capital += self._get_latest_mktval(self.curdate)
        self._buy(stocks_data)

    def _buy(self, new_stocks_to_buy):
        """
        根据最新股票列表买入，更新可用资金以及当日持仓
        """
        codes = new_stocks_to_buy.index
        
        trade_price = self.market_data.loc[codes, self.curdate]
        stks_avail = trade_price.dropna().index
        
        weights = new_stocks_to_buy.loc[stks_avail]
        amount = weights / np.sum(weights) * self.capital
        nums = amount / trade_price.loc[stks_avail]
        
        self.capital -= np.sum(amount)
        self.position_record[self.curdate] = \
                    {code:num for code, num in zip(stks_avail, nums)}
    
    def summary(self, start_date=None, end_date=None):
        if start_date is None and end_date is None:
            start_date, end_date = self.portfolio_record.index[0], self.portfolio_record.index[-1]

        ann_ret = self._annual_return(None, start_date, end_date)   #年化收益
        ann_vol = self._annual_vol(None, start_date, end_date)      #年化波动
        max_wd = self._max_drawdown(None, start_date, end_date)     #最大回撤
        sharpe = self._sharpe_ratio(start_date, end_date)     #夏普比率
        ann_excess_ret = self._ann_excess_ret(start_date, end_date)  #年化超额收益
        ic_rate = self._ic_rate(start_date, end_date)
        win_rate = self._winning_rate(start_date, end_date)          #相对基准日胜率
        te = self._te(start_date, end_date)                   #跟踪误差
        turnover_rate = self._turnover_rate(start_date, end_date)  #换手率
        summary = {
                '年度收益': ann_ret, 
                '年度波动': ann_vol, 
                '最大回撤': max_wd, 
                '夏普比率': sharpe, 
                '年度超额收益': ann_excess_ret, 
                '日胜率': win_rate,
                '跟踪误差': te,
                '换手率': turnover_rate,
                '信息比率': ic_rate
                  }
        return pd.Series(summary)
    
    def summary_yearly(self):
        if len(self.portfolio_record) == 0:
            raise RuntimeError("请运行回测函数后再查看回测统计.")
        if 'benchmark' not in self.portfolio_record.columns:
            self.portfolio_record['benchmark'] = self._get_benchmark()
        
        all_dates = self.portfolio_record.index
        start_dates = all_dates[:1].tolist() + list(before_date for before_date, after_date in zip(all_dates[1:], all_dates[:-1])
                                          if before_date.year != after_date.year)
        end_dates = list(before_date for before_date, after_date in zip(all_dates[:-1], all_dates[1:])
                         if before_date.year != after_date.year) + all_dates[-1:].tolist()
        res = pd.DataFrame()

        for sdate, edate in zip(start_dates, end_dates):
            summary_year = self.summary(sdate, edate)
            summary_year.name = str(sdate.year)
            res = pd.concat([res, summary_year], axis=1)
        summary_all = self.summary()
        summary_all.name = '总计'
        res = pd.concat([res, summary_all], axis=1)
        res = res.T[['年度超额收益', '年度收益', '年度波动', '夏普比率', '信息比率', 
                     '最大回撤', '日胜率', '跟踪误差','换手率' ]]
        return res
    
    def _get_benchmark(self):
        start_date, end_date = self.portfolio_record.index[0], \
                                self.portfolio_record.index[-1]
        return self.benchmark_data.loc[start_date:end_date]
    
    def _get_date_gap(self, start_date=None, end_date=None, freq='d'):
        if start_date is None and end_date is None:
            start_date = self.portfolio_record.index[0]
            end_date = self.portfolio_record.index[-1]
        days = (end_date - start_date) / toffsets.timedelta(1)
        if freq == 'y':
            return days / 365
        elif freq == 'q':
            return days / 365 * 4
        elif freq == 'M':
            return days / 365 * 12
        elif freq == 'd':
            return days 
    
    def _te(self, start_date=None, end_date=None):
        if start_date and end_date:
            pr = self.portfolio_record.loc[start_date:end_date]
        else:
            pr = self.portfolio_record
        td = (pr['netval_pctchg'] - pr['benchmark'])
        te = np.sqrt(min(len(pr), 252)) * np.sqrt(1 / (len(td) - 1) * np.sum((td - np.mean(td))**2))
        return te
    
    def _ic_rate(self, start_date=None, end_date=None):
        excess_acc_ret = self._get_excess_acc_ret(start_date, end_date)
        ann_excess_ret = self._ann_excess_ret(start_date, end_date)
        ann_excess_ret_vol = self._annual_vol(excess_acc_ret, 
                                              start_date, end_date)
        return (ann_excess_ret - self.rf_rate) / ann_excess_ret_vol
    
    def _turnover_rate(self, start_date=None, end_date=None):            
        positions = self.position_record.fillna(0).T
        if start_date and end_date:
            positions = positions.loc[start_date:end_date]
        turnover_rate = np.sum(np.abs(positions - positions.shift(1)) / 2, axis=1)
        turnover_rate = np.mean(turnover_rate) * 12
        return turnover_rate
        
    def _winning_rate(self, start_date=None, end_date=None):
        if self.use_pctchg:
            nv_pctchg = self.portfolio_record['netval_pctchg']
            bm_pctchg = self.portfolio_record['benchmark']
        else:
            nv_pctchg = self.portfolio_record['net_value'] / \
                            self.portfolio_record['net_value'].shift(1) - 1
            bm_pctchg = self.portfolio_record['benchmark'] / \
                            self.portfolio_record['benchmark'].shift(1) - 1
            nv_pctchg, bm_pctchg = nv_pctchg.iloc[1:], bm_pctchg.iloc[1:]
        if start_date and end_date:
            nv_pctchg, bm_pctchg = nv_pctchg.loc[start_date:end_date], bm_pctchg.loc[start_date:end_date]
        win_daily = (nv_pctchg > bm_pctchg)
        win_rate = np.sum(win_daily) / len(win_daily)
        return win_rate
    
    def _annual_return(self, net_vals=None, start_date=None, end_date=None):
        if net_vals is None:
            net_vals = self.portfolio_record['net_value']
        if start_date and end_date:
            net_vals = net_vals.loc[start_date:end_date]
        total_ret = net_vals.values[-1] / net_vals.values[0] - 1
        date_gap = self._get_date_gap(start_date, end_date, freq='d')
        exp = 365 / date_gap
        ann_ret = (1 + total_ret) ** exp - 1
        if date_gap <= 365:
            return total_ret
        else:
            return ann_ret
    
    def _annual_vol(self, net_vals=None, start_date=None, end_date=None):
        if net_vals is not None or not self.use_pctchg:
            if net_vals is None:
                net_vals = self.portfolio_record['net_value']
            ret_per_period = net_vals / net_vals.shift(1) - 1
            ret_per_period = ret_per_period.iloc[1:]
        elif self.use_pctchg:
            ret_per_period = self.portfolio_record['netval_pctchg']
        if start_date and end_date:
            ret_per_period = ret_per_period.loc[start_date:end_date]
        ann_vol = ret_per_period.std() * np.sqrt(min(len(ret_per_period),252))
        return ann_vol

    def _max_drawdown(self, acc_rets=None, start_date=None, end_date=None):
        if acc_rets is None:
            acc_rets = self.portfolio_record['net_value'] / self.portfolio_record['net_value'].values[0] - 1
        if start_date and end_date:
            acc_rets = acc_rets.loc[start_date:end_date]
        max_drawdown = (1 - (1 + acc_rets) / (1 + acc_rets.expanding().max())).max()
        return max_drawdown

    def _sharpe_ratio(self, start_date=None, end_date=None, ann_ret=None, ann_vol=None):
        if ann_ret is None:
            ann_ret = self._annual_return(None, start_date, end_date)
        if ann_vol is None:
            ann_vol = self._annual_vol(None, start_date, end_date)
        return (ann_ret - self.rf_rate) / ann_vol
    
    def _get_excess_acc_ret(self, start_date=None, end_date=None):
        if self.use_pctchg:
            bm_ret = self.portfolio_record['benchmark']
            nv_ret = self.portfolio_record['netval_pctchg']
        else:
            bm_ret = self.portfolio_record['benchmark'] / self.portfolio_record['benchmark'].shift(1) - 1
            nv_ret = self.portfolio_record['net_value'] / self.portfolio_record['net_value'].shift(1) - 1
        if start_date and end_date:
            bm_ret = bm_ret.loc[start_date:end_date] 
            nv_ret = nv_ret.loc[start_date:end_date]
        excess_ret = nv_ret.values.flatten() - bm_ret.values.flatten()
        excess_acc_ret = pd.Series(np.cumprod(1+excess_ret), index=nv_ret.index)
        return excess_acc_ret
        
    def _ann_excess_ret(self, start_date=None, end_date=None):
        excess_acc_ret = self._get_excess_acc_ret(start_date, end_date)
        ann_excess_ret = self._annual_return(net_vals=excess_acc_ret,
                                             start_date=start_date,
                                             end_date=end_date)
        return ann_excess_ret     

#因子分层回测
class SingleFactorLayerDivisionBacktest:
    def __init__(self, *, factor_name, factor_data, num_layers=5, 
                 if_concise=True, pct_chg_nm, **kwargs):
        self.num_layers = num_layers           #分层回测层数
        self.factor_name = factor_name         #因子名称
        self.factor_data = factor_data         #月频因子矩阵数据（行为证券代码，列为日期）
        self.stock_pool = self.factor_data.index  #股票池
        self.if_concise = if_concise             #是否使用简便回测方式，如是，则使用月涨跌幅进行回测，
                                                #否则采用日度复权价格进行回测
        self.pctchg_nm = pct_chg_nm
        self.kwargs = kwargs
            
    def run_layer_division_bt(self, equal_weight=True):
        #运行分层回测
        if self.if_concise:
            result = self._run_rapid_layer_divbt()
        else:
            stock_weights = self.get_stock_weight(equal_weight)            #获取各层权重  
            result = pd.DataFrame()
            for i in range(self.num_layers):
                kwargs = deepcopy(self.kwargs)
                kwargs['stock_weights'] = stock_weights[i]
                bt = Backtest_stock(**kwargs)
                bt.run_backtest()
                bt.portfolio_record.index = [f'第{i+1}组']
                result = pd.concat([result, bt.portfolio_record.T], axis=1)
        print(f"{self.factor_name}分层回测结束！")
            
        result.index.name = self.factor_name
        return result
    
    def _run_rapid_layer_divbt(self):
        global factor_path
        result = pd.DataFrame()
        for date in self.pctchg_nm.columns:
            cur_weights = self.get_stock_weight_by_group(self.factor_data[date], True)
            cur_pctchg_nm = self.pctchg_nm[date] / 100
            group_monthly_ret = pd.Series()
            for group in cur_weights.columns:
                group_weights = cur_weights[group].dropna()
                cur_layer_stocks = group_weights.index
                group_monthly_ret.loc[group] = np.nanmean(cur_pctchg_nm.loc[cur_layer_stocks])
            group_monthly_ret.name = date
            result = pd.concat([result, group_monthly_ret], axis=1)
        #对齐实际日期与对应月收益
        months = result.columns[1:].tolist()
        del result[months[-1]]
        result.columns = months
        return result.T
    
    def get_stock_weight(self, equal_weight=True):
        #对权重的格式进行转换，以便后续回测
        dates = self.factor_data.columns
        stk_weights = [self.get_stock_weight_by_group(self.factor_data[date], 
                       equal_weight) for date in dates]
        result = {date: stk_weight for date, stk_weight in zip(dates, stk_weights)}
        result = pd.Panel.from_dict(result)
        result = [result.minor_xs(group) for group in result.minor_axis]
        return result
        
    def get_stock_weight_by_group(self, factor, equal_weight=False):
        #根据因子的大小降序排列
        factor = factor.sort_values(ascending=False).dropna()
        #计算获得各层权重
        weights = self.cal_weight(factor.index)
        result = pd.DataFrame(index=factor.index)
        result.index.name = 'code'
        for i in range(len(weights)):
            labels = [factor.index[num] for num, weight in weights[i]]
            values = [weight for num, weight in weights[i]]
            result.loc[labels, f'第{i+1}组'] = values
        if equal_weight:
            #设置为等权
            result = result.where(pd.isnull(result), 1)
        return result
            
    def cal_weight(self, stock_pool):
        #权重计算方法参考华泰证券多因子系列研报
        total_num = len(stock_pool)
        weights = []
        
        total_weights = 0; j = 0
        for i in range(total_num):
            total_weights += 1 / total_num
            if i == 0:
                weights.append([])
            if total_weights > len(weights) * 1 / self.num_layers:
                before = i, len(weights) * 1 / self.num_layers - \
                            sum(n for k in range(j+1) for m, n in weights[k]) 
                after = i, 1 / total_num - before[1]
                
                weights[j].append(before) 
                weights.append([])
                weights[j+1].append(after)
                j += 1
            else:
                cur = i, 1 / total_num
                weights[j].append(cur)
        
        #调整尾差
        if len(weights[-1]) == 1:
            weights.remove(weights[-1])
        
        return weights

def panel_to_matrix(factors, factor_path=factor_path, save_path=sf_test_save_path):
    """
    将经过预处理的因子截面数据转换为因子矩阵数据
    """
    global industry_benchmark
    factors_to_be_saved = [f.replace('/','_div_') for f in factors]
    factor_matrix_path = os.path.join(save_path, '因子矩阵') if not save_path.endswith('因子矩阵') else save_path 
    if not os.path.exists(factor_matrix_path):
        os.mkdir(factor_matrix_path)
    else:
        factors = set(tuple(factors_to_be_saved)) - \
            set(f.split('.')[0] for f in os.listdir(factor_matrix_path))
        if len(factors) == 0:
            return None
        
    factors = sorted(f.replace('_div_', '/') for f in factors) 
    if '预处理' in factor_path:
        factors.extend(['PCT_CHG_NM', f'industry_{industry_benchmark}', 
                        'MKT_CAP_FLOAT'])
    datpanel = {}
    for f in os.listdir(factor_path):
        open_name = f.replace('_div_','/')
        datdf = pd.read_csv(os.path.join(factor_path, open_name), encoding='gbk', 
                            index_col=['code'], engine='python')
        date = pd.to_datetime(f.split('.')[0])
        datpanel[date] = datdf[factors]
        
    datpanel = pd.Panel(datpanel)
    datpanel = datpanel.swapaxes(0, 2)
    for factor in datpanel.items:
        dat = datpanel.loc[factor]
        save_name = factor.replace('/','_div_') if '/' in factor else factor
        dat.to_csv(os.path.join(factor_matrix_path, save_name+'.csv'),
                   encoding='gbk')
    
def plot_layerdivision(records, fname, concise):
    global sf_test_save_path, num_layers
    layerdiv_figpath = os.path.join(sf_test_save_path, '分层回测', '分层图')
    if not os.path.exists(layerdiv_figpath):
        os.mkdir(layerdiv_figpath)
    
    if concise:
        records = np.cumprod(1+records)
        records /= records.iloc[0]
    records = records.T / records.apply(np.mean, axis=1)
    records = records.T
    plt.plot(records)
    plt.title(fname)
    plt.legend(records.columns, loc=0)
    
    save_name = fname.replace('/','_div_') if '/' in fname else fname
    plt.savefig(os.path.join(layerdiv_figpath, save_name+f'_{num_layers}.jpg'))
    plt.close()
    
def bar_plot_yearly(records, fname, concise):
    global sf_test_save_path, num_layers
    barwidth = 1 / num_layers - 0.03
    layerdiv_barpath = os.path.join(sf_test_save_path, '分层回测', '分年收益图')
    if not os.path.exists(layerdiv_barpath):
        os.mkdir(layerdiv_barpath)
    
    if concise:
        records_gp = records.groupby(pd.Grouper(freq='y'))
        records = pd.DataFrame()
        for year, month_ret in records_gp:
            month_netvalue = np.cumprod(1+month_ret)
            year_return = month_netvalue.iloc[-1] / month_netvalue.iloc[0] - 1
            year_return.name = year
            if year == 2009:
                year_return = (1 + year_return) ** (12/11) - 1
            records = pd.concat([records, year_return], axis=1)
        records = records.T
    else:
        records = records.groupby(pd.Grouper(freq='y')).\
                apply(lambda df: df.iloc[-1] / df.iloc[0] - 1)
        records = records.T - records.mean(axis=1)
        records = records.T
    #减去5组间均值
    records = records.T - records.mean(axis=1)
    records = records.T
    time = np.array([d.year for d in records.index])

    plt.bar(time, records['第1组'], barwidth, color='blue', label='第1组')
    plt.bar(time+barwidth, records['第2组'], barwidth, color='green', label='第2组')
    plt.bar(time+2*barwidth, records['第3组'], barwidth, color='red', label='第3组')
    plt.bar(time+3*barwidth, records['第4组'], barwidth, color='#E066FF', label='第4组')
    plt.bar(time+4*barwidth, records['第5组'], barwidth, color='#EEB422', label='第5组')
    plt.xticks(time+2.5*barwidth, time)
    plt.legend(records.columns, loc=0)
    
    save_name = fname.replace('/','_div_') if '/' in fname else fname
    plt.savefig(os.path.join(layerdiv_barpath, save_name+f'_{num_layers}.jpg'))
    plt.close()

def plot_group_diff_plot(records, fname, concise):
    global sf_test_save_path, num_layers, tick_spacing1
    layerdiv_diffpath = os.path.join(sf_test_save_path, '分层回测', '组1-组5')
    if not os.path.exists(layerdiv_diffpath):
        os.mkdir(layerdiv_diffpath)
    
    if concise:
        records = np.cumprod(1+records)
        records /= records.iloc[0]

    records = (records['第1组'] - records['第5组']) / records['第1组']
    
    time = [str(d)[:10] for d in records.index]
    
    fig, ax = plt.subplots(1,1)
    ax.plot(time, records.values)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing1))
    ax.set_title(fname)
    
    save_name = fname.replace('/','_div_') if '/' in fname else fname
    fig.savefig(os.path.join(layerdiv_diffpath, save_name+f'_{num_layers}.jpg'))
    plt.close()
    

