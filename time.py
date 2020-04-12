import datetime as dt
import numpy as np

def countdays(string1, string2,*,Today=False):
    m1,d1,y1 = [int(x) for x in string1.split('/')]
    m2,d2,y2 = [int(x) for x in string2.split('/')]
    today = dt.date.today()
    date1 = dt.date(y1, m1, d1)
    date2 = dt.date(y2, m2, d2)
    if Today == True:
        date1 = today
    else:
        pass
    dcount = date2 - date1
    day = dcount.days
    return day

def make_year(time1,time2,*,day=360):
    days = countdays(time1,time2)
    year = np.float16(days/day)
    return year 