from arithmetic import *
import numpy as np

class Mvg_Avg(object):
    def __init__(self, df, lam):
        self._df = df
        self._len = len(df)
        self._lam = lam
        print(self.__class__.__name__, "constructed")
        print(self.__repr__())
    def __del__(self):
        print(self.__class__.__name__, "destoryed")
        print(self.__repr__())
    def __repr__(self):
        return '<%s.%s object at %s>' % (
        self.__class__.__module__,
        self.__class__.__name__,
        hex(id(self))
    )   
    def mean(self, win):
        if self._lam == None:
            WEIGHT = None
        else:
            LAM = np.array([self._lam ** i for i in range(win)])
            WEIGHT = LAM/sum(LAM) 
        MEAN = []
        for i in range(self._len - win):
            df_scan = self._df[i:i + win]
            MEAN.append(mean(df_scan, weight=WEIGHT))
        return MEAN
    def std(self, win):
        if self._lam == None:
            WEIGHT = None
        else:
            LAM = np.array([self._lam ** i for i in range(win)])
            WEIGHT = LAM/sum(LAM)         
        STD = []
        for i in range(self._len - win):
            df_scan = self._df[i:i + win]
            STD.append(std(df_scan, weight=WEIGHT))
        return STD
    def cov(self, df2, win):
        if self._lam == None:
            WEIGHT = None
        else:
            LAM = np.array([self._lam ** i for i in range(win)])
            WEIGHT = LAM/sum(LAM)         
        COV = []
        for i in range(self._len - win):
            df_scan = self._df[i:i + win]
            df_scan2 = df2[i:i + win]
            COV.append(cov(df_scan, df_scan2, weight=WEIGHT))
        return COV
    def corr(self, df2, win):
        if self._lam == None:
            WEIGHT = None
        else:
            LAM = np.array([self._lam ** i for i in range(win)])
            WEIGHT = LAM/sum(LAM)        
        CORR = []
        for i in range(self._len - win):
            df_scan = self._df[i:i + win]
            df_scan2 = df2[i:i + win]
            CORR.append(corr(df_scan, df_scan2, weight=WEIGHT))
        return CORR      