from Mkrt_Risk.Performance import Dft_Vol_Corr
from Mkrt_Risk.Risk import Value_at_Risk, Expected_Shortfall

class Stock(object):
    def __init__(self, df):
        self._VaR = Value_at_Risk(df)
        self._ES = Expected_Shortfall(df)
        self._Dft_Vol = Dft_Vol_Corr(df, 1, None)
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
        dft, vol = self._Dft_Vol.Drift_Volatility(year_win*252)
        return dft, vol
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
    def MCVaR(self, year_win):
        VaR = self._VaR.MonteCarlo(year_win*252)
        return VaR
    def MCES(self, year_win):
        ES = self._ES.MonteCarlo(year_win*252)
        return ES