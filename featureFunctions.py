import pandas as pd
import math
import numpy as np
from scipy import stats
import scipy.optimize
from scipy.optimize import OptimizeWarning
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.dates import date2num
from datetime import datetime
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import matplotlib.pyplot as plt




class holder:
    1


# heiken Ashi candles

def heikenashi_helper(args):
    return heikenashi(*args)


def parallel_heikenashi(prices, periods):

    pool = ThreadPool()
    job_args = [(prices, periods)]
    pool.map(heikenashi_helper, job_args)

def heikenashi(HKA, periods):

    results = holder()

    dict = {}

    HAclose = HKA[['open','high','low','close']].sum(axis=1) / 4
    HAopen = HAclose.copy()
    HAopen.iloc[0] = HAclose.iloc[0]
    HAhigh = HAclose.copy()
    HAlow = HAclose.copy()


    for i in range(0, len(HKA)):
        HAopen.iloc[i] = (HAopen.iloc[i - 1] + HAclose.iloc[i - 1]) / 2

        HAhigh.iloc[i] = np.array([HKA.high.iloc[i]]).max()

        HAlow.iloc[i] = np.array([HKA.low.iloc[i]]).min()

    df = pd.concat((HAopen, HAhigh, HAlow, HAclose), axis=1)
    df.columns = [['open', 'high', 'close', 'low']]

    #df.index = df.index.droplevel(0)

    dict[periods[0]] = df

    results.candles = dict

    return results


def detrend(prices, method='difference'):
    # peram prices: dataframe OHLC currency data
    # peram method: method by which to detrend 'liner' or 'difference'
    # return: the detrended price series

    if method == 'difference':

        detrended = prices.close[1:] - prices.close[:-1].values

    elif method == 'linear':

        x = np.arange(0, len(prices))
        y = prices.close.values

        model = LinearRegression()

        model.fit(x.reshape(-1, 1), y.reshape(-1, 1))

        trend = model.predict(x.reshape(-1, 1))

        trend = trend.reshape((len(prices),))

        detrended = prices.close - trend
    else:
        print('you did not input a valid option')

    return detrended


# fourier series expression fitting function
def fseries(x, a0, a1, b1, w):
    f = a0 + a1 * np.cos(w * x) + b1 * np.sin(w * x)

    return f


# Sine series expression fitting function
def sseries(x, a0, b1, w):
    f = a0 + b1 * np.sin(w * x)

    return f


# Fourier series coefficient calculation function
def fourier_helper(args):
    return fourier(*args)


def parallel_fourier(prices, periods):

    pool = ThreadPool()
    job_args = [(prices, periods)]
    pool.map(fourier_helper, job_args)

def fourier(prices, periods, method='difference'):
    ""
    # param price: from OHLC Data
    # param periods: list of periods to determine coefficients i.e.,[1,3,5,7,]
    # param method: method to detrend data 'linear' or 'difference'
    # return: dictionary of dataframe containing coefficients for the periods
    ""
    results = holder()

    dict = {}
    # compute computation of the series
    plot = False

    detrended = detrend(prices, method)

    for i in range(0,len(periods)):  # loop the periods


        coeffs = []  # creat list for oefficients
        test = []


        for j in range(periods[i],len(prices)):  # shifting through the periods

            x = np.arange(0, periods[i],dtype=float)
            y = detrended.iloc[j - periods[i]:j].values.flatten()

            with warnings.catch_warnings():
                warnings.simplefilter('error',OptimizeWarning)


                try:

                    res = scipy.optimize.curve_fit(fseries, x, y)

                except (RuntimeError, OptimizeWarning):

                    res = np.empty((4, 4))
                    res[0, :] = np.NAN

                #except (KeyError):
                    #pass

            if plot == True:
                xt = np.linspace(0 , periods[i], 100)
                yt = fseries(xt, res[0][0], res[0][1], res[0][2], res[0][3])

                plt.plot(x, y)
                plt.plot(xt, yt, 'r')



                plt.show()

            coeffs = np.append(coeffs, res[0], axis=0)

        coeffs = np.array(coeffs, ).reshape(((len(coeffs) //  4, 4)))

        columns = ['a0', 'a1', 'b1', 'w']
        df = pd.DataFrame(columns=columns)




        #df.fillna(method='bfill')
        #print(df)

        coeffs = coeffs.flatten()
        a = coeffs[0::4]
        df['a0'] = a
        b = coeffs[1::4]
        df['a1'] = b
        c = coeffs[2::4]
        df['b1'] = c
        d = coeffs[3::4]
        df['w'] = d

        #print(df)
        #print(coeffs)


        dict[periods[i]] = df

    results.coeffs = dict

    return results


# sine series coefficient calculation function DATA but the array is dirfferent sizes
def sine_helper(args):
    return sine(*args)


def parallel_sine(prices, periods):

    pool = ThreadPool()
    job_args = [(prices, periods)]
    pool.map(sine_helper, job_args)

def sine(prices, periods, method='difference'):
    ""
    # param price: from OHLC Data
    # param periods: list of periods to determine coefficients i.e.,[1,3,5,7,]
    # param method: method to detrend data 'linear' or 'difference'
    # return: dictionary of dataframe containing coefficients for the periods
    ""
    results = holder()

    dict = {}
    # compute computation of the series
    plot = False

    detrended = detrend(prices, method)

    for i in range(0, len(periods)):  # loop the periods


        coeffs = []  # creat list for oefficients
        test = []

        for j in range(periods[i], len(prices)):  # shifting through the periods

            x = np.arange(0, periods[i],dtype=float)
            y = detrended.iloc[j - periods[i]:j].values.flatten()

            with warnings.catch_warnings():
                warnings.simplefilter('error')


                try:

                    res = scipy.optimize.curve_fit(sseries, x, y)

                except (RuntimeError):

                    res = np.empty((1, 3))
                    res[0, :] = np.NAN



            if plot == True:
                xt = np.linspace(0, periods[i], 100)
                yt = sseries(xt, res[0][0], res[0][1], res[0][2])

                plt.plot(x, y)
                plt.plot(xt, yt, 'r')

                plt.show()

            coeffs = np.append(coeffs, res[0], axis=0)

        coeffs = np.array(coeffs).reshape(((len(coeffs) // 3, 3 )))

        columns = ['a0','b1','w']
        df = pd.DataFrame(columns=columns)

        # df.fillna(method='bfill')


        coeffs = coeffs.flatten()
        a = coeffs[0::3]
        df['a0'] = a
        c = coeffs[1::3]
        df['b1'] = c
        d = coeffs[2::3]
        df['w'] = d
        dict[periods[i]] = df

    results.coeffs = dict



    return results

#stochastic Oscillator Function #

def stochastic_helper(args):
    return stochastic(*args)


def parallel_stochastic(prices, periods):

    pool = ThreadPool()
    job_args = [(prices, periods)]
    pool.map(stochastic_helper, job_args)

def stochastic(prices,periods):
    #param prices: price data
    #param periods: periods list to calculate function value
    #return: oscillator function values
    results = holder()
    close = {}

    for i in range(0, len(periods)):

        Ks = []

        for j in range(periods[i], len(prices)-periods[i]):

            C = prices.close.iloc[j]
            H = prices.high.iloc[j-periods[i]:j].all().max()
            L = prices.low.iloc[j - periods[i]:j].all().min()

            if H == L:
                K = 0
            else:
                K = 100*(C-L)/(H-L)
            Ks = np.append(Ks,K)
        Ks = Ks[0::1]

        columns = ['K','D']
        df = pd.DataFrame(columns=columns, index=prices.iloc[periods[i]+1:-periods[i]+1].index)
        df['K'] = list(Ks)
        a = list(df['K'].rolling(3).mean())
        df['D'] = a
        df = df.dropna()
        #print(df)

        close[periods[i]] = df

    results.close = close

    return results

#Williams Oscillator Function
def williams_helper(args):
    return williams(*args)


def parallel_williams(prices, periods):

    pool = ThreadPool()
    job_args = [(prices, periods)]
    pool.map(williams_helper, job_args)

def williams(prices,periods):
    #param prices: price data
    #param periods: list of periods
    #return; values of williams OSC funtions

    results = holder()
    close = {}

    for i in range(0,len(periods)):
        Rs = []
        for j in range(periods[i], len(prices)):

            C = prices.close.iloc[j]
            H = prices.high.iloc[j - periods[i]:j].max().all()
            L = prices.low.iloc[j - periods[i]:j].min().all()

            if H == L:
                R = 0
            else:
                R = -100 * (H - C) / (H - L)

            Rs = np.append(Rs,R)

        a = Rs[0::1]
        columns = ['R']
        df = pd.DataFrame(a, columns=columns)
        #df.columns = ['R']
        #df['R'] = list(a)
        close[periods[i]] = df

    results.close = close


    return results

# Williams Accumulation Distribution Line Function
def wadl_helper(args):
    return wadl(*args)


def parallel_wadl(prices, periods):

    pool = ThreadPool()
    job_args = [(prices, periods)]
    pool.map(wadl_helper, job_args)


def wadl(prices, periods):
    # param prices: dataframe of OHLC
    # param periods list periods to calculate function
    # return: williams accumulation ditribution lines

    results = holder()
    dict = {}


    for i in range(0, len(periods)):

        wad = []

        for j in range(periods[i], len(prices)):

            trh = np.array([prices.high.iloc[j], prices.close.iloc[j - 1]]).max()

            trl = np.array([prices.low.iloc[j], prices.close.iloc[j - 1]]).min()

            if prices.close.iloc[j].all() > prices.close.iloc[j - 1].all():

                pm = prices.close.iloc[j] - trl

            elif prices.close.iloc[j].all() < prices.close.iloc[j - 1].all():

                pm = prices.close.iloc[j] - trh

            elif prices.close.iloc[j].all() == prices.close.iloc[j - 1].all():

                pm = 0
            else:

                print('these things happen, keep up the good work')

            ad = pm * prices.volume.iloc[j]

            wad = np.append(wad, ad)

        w = wad[0::1]
        columns = ['close']
        wad = pd.DataFrame(columns=columns)#index=prices.iloc[periods[i]:-periods[i]].index)
        wad['close'] = w

        #wad.columns = [['close']]

        dict[periods[i]] = wad

    results.wadl = dict



    return results


# Data Resampling Function - changes candles
def ohlcresample(DataFrame, TimeFrame, column='ask'):
    # Param DataFrame: DataFrame containing data that we want to resample
    # param TmeFrame timeframe that we want for resampling
    # Param column which column we are resampling (buy or sell) default='sell'
    # return resampled OHCL data for the given timeframe

    resampled = holder()

    grouped = DataFrame

    if np.any(DataFrame.ask == True):

        if column == 'ask':
            sell = grouped['ask'].resample(TimeFrame).ohlc()
            volume = grouped['volume'].resample(TimeFrame).count()
            resampled = pd.DataFrame(sell)
            resampled['volume'] = volume

        elif column == 'bid':
            buy = grouped['bid'].resample(TimeFrame).ohlc()
            volume = grouped['volume'].resample(TimeFrame).count()
            resampled = pd.DataFrame(buy)
            resampled['volume'] = volume

        else:

            raise ValueError('bid or ask please')

    elif np.any(DataFrame.columns == 'close'):
        open = grouped['open'].resample(TimeFrame).ohlc()
        high = grouped['high'].resample(TimeFrame).ohlc()
        low = grouped['low'].resample(TimeFrame).ohlc()
        close = grouped['close'].resample(TimeFrame).count()
        volume = grouped['volume'].resample(TimeFrame).count()


        resampled = pd.DataFrame(open)
        resampled['high'] = high
        resampled['low'] = low
        resampled['close'] = close
        resampled['volume'] = volume

        #resampled = resampled.dropna()

    return resampled

# Momentum Function

def momentum_helper(args):
    return momentum(*args)


def parallel_momentum(prices, periods):

    pool = ThreadPool()
    job_args = [(prices, periods)]
    pool.map(momentum_helper, job_args)

def momentum(prices,periods):
    # param prices dataframe or OHLC
    # param periods list of periods to calculate funtion value
    # return momentum indicator

    results = holder()
    open = {}
    close = {}

    for i in range(0,len(periods)):

        open[periods[i]] = pd.DataFrame(prices.open.iloc[periods[i]:]-prices.open.iloc[:-periods[i]].values,
                                        index=prices.iloc[periods[i]:].index)

        close[periods[i]] = pd.DataFrame(prices.close.iloc[periods[i]:] - prices.close.iloc[:-periods[i]].values,
                                        index=prices.iloc[periods[i]:].index)

        open[periods[i]].columns = [['open']]
        close[periods[i]].columns = [['close']]



    results.open = open
    results.close = close


    return results

#Proc Function (Price Rate of Change)

def proc_helper(args):
    return proc(*args)


def parallel_proc(prices, periods):

    pool = ThreadPool()
    job_args = [(prices, periods)]
    pool.map(proc_helper, job_args)

def proc(prices,periods):

    #param prices: dataframe containing prices
    #param periods: periods to calculate Rrice rate of change
    #return: Price rate of change for indicated periods

    results = holder()
    proc = {}


    for i in range(0,len(periods)):

        proc[periods[i]] = pd.DataFrame((prices.close.iloc[periods[i]:]-prices.close.iloc[:-periods[i]].values)\
                                        /prices.close.iloc[:-periods[i]].values)

        proc[periods[i]].columns = ['close']

    results.proc = proc


    return results

#Accumulation Distribution Oscillator

def ado_helper(args):
    return ado(*args)


def parallel_ado(prices, periods):

    pool = ThreadPool()
    job_args = [(prices, periods)]
    pool.map(ado_helper, job_args)

def ado(prices,periods):

    #param prices: OHLC DATAFRAME
    #param periods: periods to compute indicator
    #return indicator values for given periods

    results = holder()
    accdist = {}

    for i in range(0,len(periods)):

        AD = []

        for j in range(periods[i],len(prices)-periods[i]):

            C = prices.close.iloc[j+1].all()
            H = prices.high.iloc[j-periods[i]:j].max().all()
            L = prices.low.iloc[j-periods[i]:j].min().all()
            V = prices.volume.iloc[j+1].all()

            if H==L:
                CLV = 0
            else:
                CLV= ((C-L)-(H-C))/(H-L)
            AD = np.append(AD,CLV*V)
        A = AD[0::1]
        columns = ['AD']

        AD = pd.DataFrame(columns=columns)
        AD['AD'] = A
        #AD.drop_duplicates(inplace=True)
        #AD.columns = [['AD']]

        accdist[periods[i]] = AD

    results.AD = accdist

    return results

#MACD - Moving Average Convergence Divergence

def macd_helper(args):
    return macd(*args)


def parallel_macd(prices, periods):

    pool = ThreadPool()
    job_args = [(prices, periods)]
    pool.map(macd_helper, job_args)

def macd(prices,periods):

    #param prices OHLC dataframe prices
    #param periods 1X2 array of EMA values
    results = holder()


    EMA1 = prices.close.ewm(span=periods[0]).mean()
    EMA2 = prices.close.ewm(span=periods[1]).mean()

    MACD = pd.DataFrame(EMA1 - EMA2)
    MACD.columns = [["L"]]

    SigMACD = MACD.rolling(3).mean()
    SigMACD.columns = [['SL']]


    results.line = MACD
    results.signal = SigMACD


    return results

#CCI (Commodity Channel Index)

def cci_helper(args):
    return cci(*args)


def parallel_cci(prices, periods):

    pool = ThreadPool()
    job_args = [(prices, periods)]
    pool.map(cci_helper, job_args)

def cci(prices,periods):

    #param prices: OHLC dataframe of price data
    #param periods: periods to compute indicator
    #retunr: CCI CCI for given periods

    results = holder()
    CCI = {}


    for i in range(0,len(periods)):

        MA = prices.close.rolling(periods[i]).mean()
        std = prices.close.rolling(periods[i]).std()

        D = (prices.close-MA)/std

        CCI[periods[i]] = pd.DataFrame((prices.close-MA)/(0.015*D))
        CCI[periods[i]].columns = [['close']]


    results.cci = CCI




    return results

# Bollinger Bands


def bollinger_helper(args):
    return bollinger(*args)


def parallel_bollinger(prices, periods, deviations):

    pool = ThreadPool()
    job_args = [(prices, periods, deviations)]
    pool.map(bollinger_helper, job_args)

def bollinger(prices,periods,deviations):
    #param prices: OHLC data
    #param periods: periods to computute bollinger abnds
    #param deviations: deviations to use when calculating bands (upper & lower)
    #return: bollinger bands

    results = holder()
    boll = {}


    for i in range(0, len(periods)):

        mid = prices.close.rolling(periods[i]).mean()
        std = prices.close.rolling(periods[i]).std()

        upper = mid+deviations*std
        lower = mid-deviations*std

        df = pd.concat((upper,mid,lower),axis=1)
        df.columns = [['upper','mid','lower']]

        boll[periods[i]] = df

    results.bands = boll



    return results

# Price Averages -DATA
def paverage_helper(args):
    return paverage(*args)


def parallel_paverage(prices, periods):

    pool = ThreadPool()
    job_args = [(prices, periods)]
    pool.map(paverage_helper, job_args)

def paverage(prices,periods):

    #param prices: price data
    #param periods list of epriods to calculate indicator values
    #return; average over the given periods

    results = holder()

    avs = {}


    for i in range(0,len(periods)):

        avs[periods[i]] = pd.DataFrame(prices[['open','high','low','close']].rolling(periods[i]).mean())

    results.avs = avs

    return results

# Slope functions
def slopes_helper(args):
    return slopes(*args)


def parallel_slopes(prices, periods):

    pool = ThreadPool()
    job_args = [(prices, periods)]
    pool.map(slopes_helper, job_args)

def slopes(prices,periods):

    #param: price data
    #param periods to get the indicaot values
    #return: slopes over given periods

    results = holder()
    slope = {}

    for i in range(0,len(periods)):

        ms = []

        for j in range(periods[i],len(prices)-periods[i]):

            y = prices.high.iloc[j-periods[i]:j].values.flatten()
            x = np.arange(0,len(y))




            res = stats.linregress(x,y=y)
            m = res.slope

            ms = np.append(ms,m)


        ms = pd.DataFrame(ms,index = prices.iloc[periods[i]:-periods[i]].index)

        slope[periods[i]] = ms

    results.slope = slope

    return results