import datetime as dt
import pymongo as pymongo
import pandas as pd
import pytz as pytz
import numpy as np
import cbpro
public_client = cbpro.PublicClient()
from time import*
key = "b6c03d53fce7ba4e7bb9ce1124da710d"
secret = "bhjxgfcy4LVa2CxcX3RIPJzJaZROl0qt3gHdXBul2brr8xQRfO6EgqXxRqYkxschbgovOId9eSAeNbUslHAoHA=="
paraphrase = "76a12op115"
auth_client = cbpro.AuthenticatedClient(key, secret, paraphrase)
auth_client.get_account("b6c03d53fce7ba4e7bb9ce1124da710d")

from featurefunctions import*

class holder:
    1

def queryTimeManager():
    gmt = pytz.timezone('Etc/Greenwich')
    min = dt.timedelta(minutes=2880)  # 48hours- 2880 24-1440 12-720
    now = dt.datetime.now(gmt)
    start = now - min
    start = start.isoformat()
    now = now.isoformat()
    return {"start":start , "now" : now}

def readableTimeManager():
    return 

def getCBProLiveData():
    product = "{}-{}".format("BTC", "USD")
    unit = 3600 # 6 hour time frame
    historicalRates = public_client.get_product_historic_rates(product, start=queryTimeManager()["start"], end=queryTimeManager()["now"],granularity=unit)
    LHOCVRatesDataFrame = pd.DataFrame(historicalRates)
    LHOCVRatesDataFrame.columns = ['date', 'low', 'high', 'open', 'close', 'volume']   
    rearrangeColumns = LHOCVRatesDataFrame.columns.tolist()
    rearrangeColumns = rearrangeColumns[:-5] + rearrangeColumns[3:-2] + rearrangeColumns[2:-3] + rearrangeColumns[1:-4] + rearrangeColumns[4:-1] + rearrangeColumns[5:]
    OHLCVRatesDataFrame = LHOCVRatesDataFrame[rearrangeColumns]
    OHLCVRatesDataFrame.index = pd.to_datetime(OHLCVRatesDataFrame.loc[:, 'date'], unit='s')
    return OHLCVRatesDataFrame

def cbpro_predictions(prices):
    momentumKey = [10, 3]
    stochasticKey = [10, 4]
    williamsKey = [6, 7, 8, 9, 10]
    procKey = [10, 13, 14, 15]
    wadlKey = [15]
    adoKey = [2]
    macdKey = [15, 30]
    cciKey = [10, 10]
    bollingerKey = [15, 15, 2]
    heikenashiKey = [15]
    paverageKey = [2]
    slopeKey = [3, 4, 5, 10, 20]
    fourierKey = [10, 20, 30]
    sineKey = [5, 6]

    keylist = [momentumKey, stochasticKey, williamsKey, procKey, wadlKey, adoKey, macdKey, cciKey, bollingerKey,
               heikenashiKey, paverageKey, slopeKey, fourierKey, sineKey]

    momentumDict = momentum(prices, momentumKey)
    print('1')
    stochasticDict = stochastic(prices, stochasticKey)
    print('2')
    williamsDict = williams(prices, williamsKey)
    print('3')
    procDict = proc(prices, procKey)
    print('4')
    wadlDict = wadl(prices, wadlKey)
    print('5')
    adoDict = ado(prices, adoKey)
    print('6')
    macdDict = macd(prices, macdKey)
    print('7')
    ccidDict = cci(prices, cciKey)
    print('8')
    bollingerDict = bollinger(prices, bollingerKey, 2)
    print('9')
    heikenDict = heikenashi(prices, heikenashiKey)
    print('10')
    paverageDict = paverage(prices, paverageKey)
    print('11')
    slopeDict = slopes(prices, slopeKey)
    print('12')
    fourierDict = fourier(prices, fourierKey)
    print('13')
    sineDict = sine(prices, sineKey)
    print('14')
    
    dictlist = [momentumDict.close, stochasticDict.close, williamsDict.close, procDict.proc, wadlDict.wadl,
                adoDict.AD, macdDict.line, ccidDict.cci, bollingerDict.bands, heikenDict.candles, paverageDict.avs,
                slopeDict.slope, fourierDict.coeffs, sineDict.coeffs]
    print(dictlist[5][2])
    colfeat = ['momentum', 'stoch', 'will', 'proc', 'wadl', 'adosc', 'macd', 'cci', 'bollinger', 'heiken', 'paverage',
               'slope', 'fourier', 'sine']
    # populate masterframe
    time = prices.index  ## uncomment for live
    # time = pd.to_datetime(df['date']) ##,unit='s')              ### uncommnet for live

    masterFrame = pd.DataFrame(time, index=prices.index)  ##add time for live

    for i in range(0, len(dictlist)):

        if colfeat[i] == 'macd':
            colID = colfeat[i] + str(keylist[6][0]) + str(keylist[6][0])
            masterFrame[colID] = dictlist[i]

        else:
            for j in keylist[i]:

                for k in list(dictlist[i][j]):
                    colID = colfeat[i] + str(j) + str(k)
                    masterFrame[colID] = dictlist[i][j][k]

    threshold = round(0.7 * len(masterFrame))

    masterFrame[['open', 'high', 'low', 'close', 'volume']] = prices[['open', 'high', 'low', 'close', 'volume']]
    drop_cols = masterFrame.columns[(masterFrame == 0).sum() > 0.25 * masterFrame.shape[1]]
    masterFrame.drop(drop_cols, axis=1, inplace=True)
    masterFrame.to_csv('masterbt1Frame20.csv')

    return masterFrame.to_json(orient='split',default_handler=str)



x = 360
cbpro_predictions(getCBProLiveData())
getCBProLiveData()
queryTimeManager()["start"]