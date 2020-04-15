from basic.Return import Log_Return
from Mkrt_Risk.Performance import Dft_Vol_Corr
from scipy.stats import norm
import numpy as np

class Value_at_Risk(object):
    def __init__(self, df):
        self._df = df
        self._dft_vol = Dft_Vol_Corr(df, 1, None)
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
    def hist(self, win, dt=5, p=99, position=1):
        iloc = max(int((1-p/100)*win) - 1, 0)
        lgdf = Log_Return(self._df, dt)
        _len = len(lgdf)
        HIST = [] 
        for i in range(_len - win):
            df_scan = max(0, -position)*2 + position*np.exp(lgdf[i:i + win])
            df_scan.sort()
            HIST.append(df_scan[iloc])
        HIST = 1 - np.array(HIST)  
        return HIST
    def GBM(self, win, dt=5, p=99, position=1):
        percent = 1 - p/100 
        if position == -1:
            percent = p/100
        miu, sigma = self._dft_vol.Drift_Volatility(win)
        VAR = 1 - (max(0, -position)*2 + position*np.exp((miu - 0.5*sigma**2) * (dt/252) + norm.ppf(percent)*sigma*(dt/252)**0.5)) 
        return VAR
    def MonteCarlo(self, win, dt=5, p=99, position=1, times=10000):
        iloc = max(int((1-p/100)*times) - 1, 0)   
        miu, sigma = self._dft_vol.Drift_Volatility(win)        
        _len = len(list(zip(miu, sigma)))
        VAR = []
        for i in range(_len):
            sims = np.random.normal(size=times)
            SIMS = max(0, -position)*2 + position*np.exp((miu[i] - 0.5*sigma[i]**2) * (dt/252) + sims*sigma[i]*(dt/252)**0.5)
            SIMS.sort()
            VAR.append(SIMS[iloc])
        VAR = 1 - np.array(VAR)
        return VAR
    
class Expected_Shortfall(Value_at_Risk):
    def __init__(self, df):
        super().__init__(df)
    def __del__(self):
        super().__del__()
    def __repr__(self):
        return '<%s.%s object at %s>' % (
        self.__class__.__module__,
        self.__class__.__name__,
        hex(id(self))
    )
    def hist(self, win, dt=5, p=97.5, position=1):
        var = super().hist(win, dt=dt, p=p, position=position)
        lgdf = Log_Return(self._df, dt)
        _len = len(lgdf)
        HIST = []
        for i in range(_len - win):
            df_scan = 1 - (max(0, -position)*2 + position*np.exp(lgdf[i:i + win]))
            df_scan = df_scan[df_scan > var[i]]
            df_scan = np.mean(df_scan)
            HIST.append(df_scan) 
        HIST = np.array(HIST)
        return HIST
    def GBM(self, win, dt=5, p=97.5, position=1):
        var = super().GBM(win, dt=dt, p=p, position=position)
        percent = 1 - p/100
        miu, sigma = self._dft_vol.Drift_Volatility(win)
        K = position*(-max(0, -position)*2 + 1 - var)
        d1 = (np.log(1/K) + (miu + 0.5*sigma**2)*(dt/252)) / (sigma*(dt/252)**0.5)
        ES = 1 - (max(0, -position)*2 + position*np.exp(miu*(dt/252))*norm.cdf(-position*d1) / percent)
        return ES
    def MonteCarlo(self, win, dt=5, p=97.5, position=1, times=10000):
        var = super().MonteCarlo(win, dt=dt, p=p, position=position, times=times)   
        miu, sigma = self._dft_vol.Drift_Volatility(win)        
        _len = len(list(zip(miu, sigma)))
        ES = []
        for i in range(_len):
            sims = np.random.normal(size=times)
            SIMS = 1 - (max(0, -position)*2 + position*np.exp((miu[i] - 0.5*sigma[i]**2) * (dt/252) + sims*sigma[i]*(dt/252)**0.5))
            SIMS = SIMS[SIMS > var[i]]
            SIMS = np.mean(SIMS)
            ES.append(SIMS)
        ES = np.array(ES)
        return ES        