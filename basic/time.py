import datetime as dt

def countdays(string1, string2=None):
    date1 = dt.date.fromisoformat(string1)
    if string2 == None:
        date2 = dt.date.today()
    else:
        date2 = dt.date.fromisoformat(string2)
    dcount = date2 - date1
    day = dcount.days
    return day

def make_year(time1,time2=None,day=360):
    days = countdays(time1,time2)
    year = float(days/day)
    return year 