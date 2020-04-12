def Log_Return(df, dt):
    from numpy import log
    Return = [log(df[i-dt]) - log(df[i]) for i in range(dt, len(df))]
    return Return

def Simple_Return(df, dt):
    Return = [df[i-dt]/df[i] - 1 for i in range(dt, len(df))]
    return Return
    
    