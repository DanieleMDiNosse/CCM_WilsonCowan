import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from sklearn.feature_selection import mutual_info_classif
from scipy.integrate import RK45
from scipy.integrate import solve_ivp
import math


def crosscorr(x, y, max_lag):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    cross_corr = []
    for d in range(max_lag):
        cc = 0
        for i in range(len(x)-d):
            cc += (x[i] - x_mean) * (y[i+d] - y_mean)
        cc = cc / np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
        cross_corr.append(cc)

    plt.figure()
    plt.plot(cross_corr)
    plt.title('Cross-correlation function')
    plt.xlabel('Lags')
    plt.grid()
    return cross_corr


def autocorr(x, max_lag):
    x_mean = np.mean(x)
    auto_corr = []
    for d in range(max_lag):
        ac = 0
        for i in range(len(x)-d):
            ac += (x[i] - x_mean) * (x[i+d] - x_mean)
        ac = ac / np.sqrt(np.sum((x - x_mean)**2) * np.sum((x - x_mean)**2))
        auto_corr.append(ac)

    plt.figure()
    plt.plot(auto_corr)
    plt.title('Auto-correlation function')
    plt.xlabel('Lags')
    plt.grid()
    return auto_corr

def figure():
    plt.figure()
    plt.plot(ALL,'k', label='ALL')
    plt.plot(XRX,'r', label='XRX', alpha=0.8)
    plt.grid()
    plt.legend()
    plt.show()


def augmented_dickey_fuller_statistics(time_series):
    '''
    The Augmented Dickey Fuller test is a test to check the presence of a unit root in the characteristic equation of a stochastic
    process. Such presence will implies the process to be non-stationary
    '''
    result = adfuller(time_series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

def granger_causality_test(x,y, maxlag):
    '''
    Test to check if y Granger causes x
    '''
    if type(x) != 'numpy.ndarry': x = np.array(x)
    if type(y) != 'numpy.ndarry': y = np.array(y)

    gca_list = np.array([x,y])
    gca_matrix = np.transpose(gca_list)
    gca = grangercausalitytests(gca_matrix, maxlag=maxlag, verbose=True)

    return gca

if __name__ == "__main__":
    
    df_returns = pd.read_csv(
        "/home/danielem/Documents/PortfolioML/portfolioML/data/ReturnsData.csv")
    df_price = pd.read_csv(
        "/home/danielem/Documents/PortfolioML/portfolioML/data/PriceData.csv")
    date = df_price[df_price.columns[0]][:-1]

    ALL = df_returns['ALL']
    XRX = df_returns['XRX']

    gca = granger_causality_test(ALL,XRX, maxlag=5)
