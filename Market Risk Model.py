import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

today = pd.datetime.today().strftime('%m/%d/%Y')    

def initialization(Equity_excel_path,Option_excel_path):
    return Equity_excel_path,Option_excel_path
def numinput(w1,w2,w3,portfolio_position):
    if 0 <= w1 or w2 or w3 <= 1:
        pass
    else:
        try:
            raise TypeError('Not A Valid Number for the weight')
        except TypeError:
            print("please choose the right weight between 0 and 1.\n")
            raise
    if portfolio_position == -1 or 1:
        pass
    else:
        try:
            raise TypeError('Not A Valid Number for the position')
        except TypeError:
            print("please choose the right position. 1 for long and -1 for short.\n")
            raise
    return w1,w2,w3,portfolio_position

Equity_excel_path,Option_excel_path = initialization('C:/Users/caoyu/OneDrive/FRM Practices/Math/Equity Data.xlsx','C:/Users/caoyu/OneDrive/FRM Practices/Math/Option Data.xlsx')
w1,w2,w3,portfolio_position = numinput(0.45,0.5,0.05,-1)

class portfolio(object):
    Equity = pd.read_excel(Equity_excel_path,sheet_name = 'Data Analysis',delimiter=',')
    Option = pd.read_excel(Option_excel_path,sheet_name = 'Data Analysis',delimiter=',')
    SPXOption = Option[['SPX']]
    SPX_imp = Option[['SPX Imp Vol']]/100
    
    def __init__(self,obs,w1,w2,w3,portfolio_position):
        self.obs = obs
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.portfolio_position = portfolio_position
        
        self.S = 10000
        self.size = self.obs*252
        self.time = self.Equity[['Pricing Date']][0:self.obs*252]
        
        self.lookup = self.vlookup()
        self.F0 = self.lookup.at['09/05/1997','F']
        self.XRX0 = self.lookup.at['09/05/1997','XRX']      
        self.SPX0 = self.lookup.at['09/05/1997','SPX option']
        self.Fshares = self.w1*self.S/self.F0
        self.XRXshares = self.w2*self.S/self.XRX0
        self.Optionshares = self.w3*self.S/self.SPX0
        self.lookup['portfolio'] = self.Fshares*self.lookup['F']+self.XRXshares*self.lookup['XRX']+self.Optionshares*self.lookup['SPX option']
        self.lookup['P&L'] = self.S - self.lookup['portfolio']
        self.Return = pd.DataFrame(np.c_[self.struct(1,label='F'),self.struct(1,label='XRX'),self.struct(1,label='SPX'),self.struct(1,label='SPX option')],columns = ['F','XRX','SPX','SPX option'])
    def option(self,S,r=0.01,T=1.00,Type='put'):
        optiontime = self.Option[['Pricing Date']]
        K = np.array(self.SPXOption)
        vol = np.array(self.SPX_imp)
        d1 = (np.log(S)-np.log(K)+(r+0.5*vol**2)*T)/(vol*np.sqrt(T))
        d2 = (np.log(S)-np.log(K)+(r-0.5*vol**2)*T)/(vol*np.sqrt(T))
        put = K*np.exp(-r*T)*stats.norm.cdf(-d2) - S*stats.norm.cdf(-d1)
        call = -K*np.exp(-r*T)*stats.norm.cdf(d2) + S*stats.norm.cdf(d1)
        if Type=='call':
            df = pd.DataFrame(np.array(call),index=optiontime,columns=['SPX option'])
        else:
            df = pd.DataFrame(np.array(put),index=optiontime,columns=['SPX option'])
        df.index = df.index[:,0].strftime('%m/%d/%Y')
        return df
    def vlookup(self):
        lookup = self.Equity.set_index(['Pricing Date'])
        lookup.index=lookup.index.strftime('%m/%d/%Y')
        lookup = pd.concat([lookup,self.option(self.SPXOption)],axis=1,sort=False)
        lookup[['SPX option']]=lookup[['SPX option']].fillna(0)
        return lookup
    def struct(self,days,label=None):
        if label==None:
            inter = np.log(self.lookup[['portfolio']].shift(days)/self.lookup[['portfolio']])
        else:
            if label == 'SPX option':
                inter = np.log(self.lookup[[label]].shift(days)/self.lookup[[label]])
            else:
                inter = np.log(self.Equity[[label]].shift(days)/self.Equity[[label]])
        return inter[days:].reset_index(drop=True)
    def rolling(self,years,label=None):
        data1 = self.struct(1)
        start = years*252-1
        if label == None:
            miu = data1.rolling(years*252).mean()[start:start+self.obs*252]
            sigma = data1.rolling(years*252).std(ddof=0)[start:start+self.obs*252]
        else:
            miu = self.Return[label].rolling(years*252).mean()[start:start+self.obs*252]
            sigma = self.Return[label].rolling(years*252).std(ddof=0)[start:start+self.obs*252]
        return pd.DataFrame(np.c_[miu, sigma],index=self.time,columns=['miu','sigma'])
    def rollingreal(self,years,label=None):
        old_miu = self.rolling(years,label)['miu']
        old_sigma = self.rolling(years,label)['sigma']
        sigma = old_sigma*np.sqrt(252)
        miu = old_miu*252 + 0.5*sigma**2
        return pd.DataFrame(np.c_[miu, sigma],index=self.time,columns=['miu','sigma'])
    def correlation(self,years,label1,label2):
        start = years*252-1
        rho = np.transpose(np.array([self.Return[label1].rolling(years*252).corr(self.Return[label2])[start:start+self.obs*252]]))
        return pd.DataFrame(np.c_[rho],index=self.time,columns=['rho'])

class Historical(portfolio):
    def __init__(self,obs,w1,w2,w3,portfolio_position,years):
        portfolio.__init__(self,obs,w1,w2,w3,portfolio_position)
        self.years = years
        self.data = self.struct(5)
    def VaR(self,percent=0.99):
        S = self.S
        temp1 = max(0,-self.portfolio_position)*2*S
        temp2 = self.portfolio_position*S 
        p = 100 - percent*100
        VaR = np.zeros(self.obs*252)
        inter=self.data 
        for i in range(0,self.obs*252): 
            array = np.sort(temp1+temp2*np.exp(inter[i:i+self.years*252]),axis=None)
            if 0 <= p <= 100:
                VaR[i] = S - np.percentile(array,p,interpolation='lower')
            else:
                try:
                    raise TypeError('Percentile Not In the Range')
                except TypeError:
                    VaR[i] = 0
                    print("please choose the right percentile between 0 and 100. \n")
                    raise
                break
        return pd.DataFrame(VaR,index=self.time,columns=['VaR'])
    def ES(self,percent=0.975):
        S = self.S
        p = 100 - percent*100
        temp1 = max(0,-self.portfolio_position)*2*S
        temp2 = self.portfolio_position*S
        ES = np.zeros(self.obs*252)        
        inter=self.data
        reference = np.array(self.VaR(percent=percent))
        for i in range(0,self.obs*252):
            array = S - np.sort(temp1+temp2*np.exp(inter[i:i+self.years*252]),axis=None)
            if 0 <= p <= 100:
                new_array = array[array > reference[i]]
                ES[i] = np.mean(new_array)
            else:
                try:
                    raise TypeError('Percentile Not In the Range')
                except TypeError:
                    ES[i] = 0
                    print("please choose the right percentile between 0 and 100. \n")
                    raise
                break
        return pd.DataFrame(ES,index=self.time,columns=['ES'])
    
class GBM(portfolio):
    def __init__(self,obs,w1,w2,w3,portfolio_position,years):
        portfolio.__init__(self,obs,w1,w2,w3,portfolio_position)
        self.years = years
    def VaR(self,p=0.99,t=5/252):
        S = self.S
        temp1 = - min(self.portfolio_position,0)*2*S
        temp2 = self.portfolio_position*S
        df = self.rollingreal(self.years,label=None)
        miu = np.array(df[['miu']])
        sigma = np.array(df[['sigma']])
        if self.portfolio_position < 0 :
            percent = p
        else:
            percent = 1-p
        VaR = S - (temp1 + temp2*np.exp((miu-0.5*sigma**2)*t+sigma*np.sqrt(t)*stats.norm.ppf(percent)))
        return pd.DataFrame(VaR,index=self.time,columns=['VaR'])
    def ES(self,p=0.975,t=5/252):
        S = self.S
        temp1 = - min(self.portfolio_position,0)*2*S
        temp2 = self.portfolio_position*S 
        df = self.rollingreal(self.years)
        miu = np.array(df[['miu']])
        sigma = np.array(df[['sigma']])
        if self.portfolio_position < 0 :
            K = temp1 - S + self.VaR(p=p)
        else:
            K = -temp1 + S - self.VaR(p=p)
        d1 = (np.log(abs(temp2)/K) + (miu + 0.5*sigma**2)*t)/(sigma*np.sqrt(t))
        ES = S - (temp1 + temp2*np.exp(miu*t)*stats.norm.cdf(-self.portfolio_position*d1)/(1-p))            
        return pd.DataFrame(ES,index=self.time,columns=['ES'])

class MonteCarlo(portfolio):
    def __init__(self,obs,w1,w2,w3,portfolio_position,years):
        portfolio.__init__(self,obs,w1,w2,w3,portfolio_position)
        self.years = years
        pmc = self.rollingreal(self.years)
        self.pmiu = np.array(pmc[['miu']])
        self.psigma = np.array(pmc[['sigma']])
    def VaRsimulation(self,times=10000,p=0.99,t=5/252):
        S = self.S
        VaR = np.zeros(self.obs*252)
        for i in range(0,self.obs*252):
            inter = np.random.normal(size=times)
            miu_vector = np.ones(times)*self.pmiu[i]
            sigma_vector = np.ones(times)*self.psigma[i]
            result_vector = S*np.exp((miu_vector - 0.5*sigma_vector**2)*t+sigma_vector*np.sqrt(t)*inter)
            result_vector = np.sort(result_vector,axis=None)
            if self.portfolio_position == -1:
                VaR[i] = np.percentile(result_vector,p*100,interpolation='lower') - S
            else:
                VaR[i] = S - np.percentile(result_vector,(1-p)*100,interpolation='lower')                
        return pd.DataFrame(VaR,index=self.time,columns=['VaR'])
    def ESsimulation(self,times=10000,p=0.975,t=5/252):
        S = self.S
        ES = np.zeros(self.obs*252)
        reference = np.array(self.VaRsimulation(p=0.975))
        for i in range(0,self.obs*252):
            inter = np.random.normal(size=times)
            miu_vector = np.ones(times)*self.pmiu[i]
            sigma_vector = np.ones(times)*self.psigma[i]
            result_vector = S*np.exp((miu_vector - 0.5*sigma_vector**2)*t+sigma_vector*np.sqrt(t)*inter)
            if self.portfolio_position == -1:
                result_vector = result_vector - S
            else:
                result_vector = S - result_vector
            result_vector = result_vector[result_vector>=reference[i]]
            ES[i] = np.mean(result_vector)
        return pd.DataFrame(ES,index=self.time,columns=['ES'])   

class backtesting(portfolio):
    def __init__(self,obs,w1,w2,w3,portfolio_position,years,typeVaR):
        portfolio.__init__(self,obs,w1,w2,w3,portfolio_position)
        self.typeVaR = typeVaR
        self.years = years
    def VaR(self):
        if self.typeVaR == 'Hist':
            VaR = Historical(self.obs,self.w1,self.w2,self.w3,self.portfolio_position,self.years).VaR()
        if self.typeVaR == 'GBM':
            VaR = GBM(self.obs,self.w1,self.w2,self.w3,self.portfolio_position,self.years).VaR()
        if self.typeVaR == 'MC':
            VaR = MonteCarlo(self.obs,self.w1,self.w2,self.w3,self.portfolio_position,self.years).VaRsimulation()
        return VaR
    def loss(self,days=5):
        pshares = np.array(self.S/self.lookup[['portfolio']])[days:]
        pdiff = np.array(self.lookup[['portfolio']].shift(days) - self.lookup[['portfolio']])[days:]
        inter = (pshares*pdiff)[days:]
        inter = (-self.portfolio_position)*inter[0:self.obs*252]
        return pd.DataFrame(inter,index=self.time,columns=['P&L'])
    def count(self,days=5):
        reference = np.array(self.VaR())
        exception = np.zeros(self.obs*252)
        realloss = np.array(self.loss(days=days))
        total = np.zeros(self.obs*252)
        for i in range(0,self.obs*252-days):
            exception[i] = (realloss[i]>reference[i+days])
        for j in range(0,self.obs*252):
            total[j] = sum(exception[j:j+252])
        total = pd.DataFrame(total,index=self.time,columns=['exception'])
        return total

hist = Historical(20,w1,w2,w3,portfolio_position,5)
histVaR = hist.VaR()
histES = hist.ES()

gbm = GBM(20,w1,w2,w3,portfolio_position,5)
GBMVaR = gbm.VaR()
GBMES = gbm.ES()

MC = MonteCarlo(20,w1,w2,w3,portfolio_position,5)
MCVaR = MC.VaRsimulation()
MCES = MC.ESsimulation()

Histtest = backtesting(20,w1,w2,w3,portfolio_position,5,'Hist')
Histcount = Histtest.count()
GBMtest = backtesting(20,w1,w2,w3,portfolio_position,5,'GBM')
GBMcount = GBMtest.count()
MCtest = backtesting(20,w1,w2,w3,portfolio_position,5,'MC')
MCcount = MCtest.count()

plt.figure(figsize=(20,10))
plt.title("VaR and ES",fontsize=30)
plt.plot(histVaR,label='Hist VaR')
plt.plot(histES,label='Hist ES')
plt.plot(GBMVaR,label='Para VaR')
plt.plot(GBMES,label='Para ES')
plt.plot(MCVaR,label='Monte Carlo VaR')
plt.plot(MCES,label='Monte Carlo ES')
plt.xlabel('years',fontsize=20)
plt.ylabel('VaR/ES',fontsize=20)
plt.legend(loc='upper right',fontsize=20)
plt.show()

plt.figure(figsize=(20,10))
plt.title("Hist Exception",fontsize=30)
plt.plot(Histcount,label='Exception')
plt.xlabel('years',fontsize=20)
plt.ylabel('Exception',fontsize=20)
plt.axhline(y=5,color='green',label='green zone')
plt.legend(loc='upper right',fontsize=20)
plt.show()

plt.figure(figsize=(20,10))
plt.title("Para Exception",fontsize=30)
plt.plot(GBMcount,label='Exception')
plt.xlabel('years',fontsize=20)
plt.ylabel('Exception',fontsize=20)
plt.axhline(y=5,color='green',label='green zone')
plt.legend(loc='upper right',fontsize=20)
plt.show()

plt.figure(figsize=(20,10))
plt.title("Monte Carlo Exception",fontsize=30)
plt.plot(MCcount,label='Exception')
plt.xlabel('years',fontsize=20)
plt.ylabel('Exception',fontsize=20)
plt.axhline(y=5,color='green',label='green zone')
plt.legend(loc='upper right',fontsize=20)
plt.show()

plt.figure(figsize=(20,10))
plt.title("Loss vs Hist VaR",fontsize=30)
plt.plot(Histtest.VaR(),label='Hist VaR')
plt.plot(Histtest.loss(5),label='P&L')
plt.xlabel('years',fontsize=20)
plt.ylabel('P&L',fontsize=20)
plt.legend(loc='upper right',fontsize=20)
plt.show()

plt.figure(figsize=(20,10))
plt.title("Loss vs Para VaR",fontsize=30)
plt.plot(GBMtest.VaR(),label='Para VaR')
plt.plot(GBMtest.loss(5),label='P&L')
plt.xlabel('years',fontsize=20)
plt.ylabel('P&L',fontsize=20)
plt.legend(loc='upper right',fontsize=20)
plt.show()

plt.figure(figsize=(20,10))
plt.title("Loss vs Monte Carlo VaR",fontsize=30)
plt.plot(MCtest.VaR(),label='Monte Carlo VaR')
plt.plot(MCtest.loss(5),label='P&L')
plt.xlabel('years',fontsize=20)
plt.ylabel('P&L',fontsize=20)
plt.legend(loc='upper right',fontsize=20)
plt.show()
