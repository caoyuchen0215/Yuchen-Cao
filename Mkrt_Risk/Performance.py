from basic.Return import Log_Return
from basic.Rolling import Mvg_Avg
import numpy as np

class Dft_Vol_Corr(Mvg_Avg):
    def __init__(self, df, dt, lam):
        super().__init__(Log_Return(df, dt), lam)
        self._dt = dt            
    def __del__(self):
        print(self.__class__.__name__, "destoryed")
        print(self.__repr__())
    def __repr__(self):
        return '<%s.%s object at %s>' % (
        self.__class__.__module__,
        self.__class__.__name__,
        hex(id(self))
    )      
    def Drift_Volatility(self, win):
        Mean = np.array(self.mean(win))
        Std = np.array(self.std(win))
        Drift = (Mean + 0.5*Std**2) * (252/self._dt)
        Volatility = Std * (252/self._dt)**0.5
        return Drift, Volatility
    def Correlation(self, df2, win):
        Correlation = np.array(self.corr(df2, win))
        return Correlation