from basic.Return import Simple_Return
from Mkrt_Risk.Risk import Value_at_Risk
import numpy as np

class Backtesting(Value_at_Risk):
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
    def real_loss(self, dt=5, position=1):
        p_l = -position * np.array(Simple_Return(self._df, dt))
        return p_l
    def count(self, VaR, year_win, dt=5, p=99, position=1):
        p_l = self.real_loss(dt=dt, position=position)
        if VaR == 'GBM':
            var = self.GBM(year_win*252, dt=dt, p=p, position=position)
        if VaR == 'MC':
            var = self.MonteCarlo(year_win*252, dt=dt, p=p, position=position)
        if VaR == 'hist':
            var = self.hist(year_win*252, dt=dt, p=p, position=position)       
        _len = len(var)
        exception = []
        for i in range(_len-dt):
            exception.append(bool(p_l[i]>var[i+dt]))
        exception = np.array(exception)
        
        _len = len(exception)
        count = []
        for i in range(_len - 252):
            count.append(sum(exception[i:i+252]))
        count = np.array(count)
        return count
        
        
    
