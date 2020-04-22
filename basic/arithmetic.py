import numpy as np

def mean(df, weight=None):
    df = np.array(df)
    if weight == None:
        mean = np.mean(df)
    else:
        mean = np.dot(df,weight)
    return mean

def std(df, weight=None):
    df = np.array(df)
    std = (mean(df**2, weight=weight) - (mean(df, weight=weight))**2)**(0.5)
    return std

def cov(df, df2, weight=None):
    num1 = np.array(df)
    num2 = np.array(df2)
    cov = mean((num1 - mean(df, weight=weight))*(num2 - mean(df2, weight=weight)), weight=weight)
    return cov

def corr(df, df2, weight=None):
    corr = cov(df, df2, weight=weight) / (std(df, weight=weight) * std(df2, weight=weight))
    return corr
        
