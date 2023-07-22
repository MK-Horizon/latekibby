import cbpro
import pandas as pd
public_client = cbpro.PublicClient()
import pytz as pytz
import datetime as dt
from time import*
key = "b6c03d53fce7ba4e7bb9ce1124da710d"
secret = "bhjxgfcy4LVa2CxcX3RIPJzJaZROl0qt3gHdXBul2brr8xQRfO6EgqXxRqYkxschbgovOId9eSAeNbUslHAoHA=="
paraphrase = "76a12op115"
auth_client = cbpro.AuthenticatedClient(key, secret, paraphrase)
auth_client.get_account("b6c03d53fce7ba4e7bb9ce1124da710d")




def queryTimeManager():
    gmt = pytz.timezone('Etc/Greenwich')
    min = dt.timedelta(minutes=2880)  # 48hours- 2880 24-1440 12-720
    now = dt.datetime.now(gmt)
    start = now - min
    start = start.isoformat()
    now = now.isoformat()
    return {"start":start , "now" : now}


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

    return historicalRates

print({"bitcoin" : getCBProLiveData()})