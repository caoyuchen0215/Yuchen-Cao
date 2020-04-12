from Performance import Dft_Vol_Corr
from Risk import Value_at_Risk, Expected_Shortfall

class Stock(object):
    def __init__(self, df, lam):
        self._VaR = Value_at_Risk(df)
        self._ES = Expected_Shortfall(df)
        self._Dft_Vol = Dft_Vol_Corr(df, 1, lam)
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
    def Dft_Vol(self, year_win):
        Drift, Volatility = self._Dft_Vol.Drift_Volatility(year_win*252)
        return Drift, Volatility
    def HistVaR(self, year_win):
        VaR = self._VaR.hist(year_win*252)
        return VaR
    def HistES(self, year_win):
        ES = self._ES.hist(year_win*252)
        return ES
    def GBMVaR(self, year_win):
        VaR = self._VaR.GBM(year_win*252)
        return VaR
    def GBMES(self, year_win):
        ES = self._ES.GBM(year_win*252)
        return ES
    