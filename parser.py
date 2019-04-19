import pandas as pd
import numpy as np
from datetime import datetime

def read_csv(filename):
    """
    Opens and parses/processes csv files to return data and labels

    Args:
        filename - String containing a csv file's directory

    Returns:
        X - Data to be used as input features for the neural net
        y - The labels for each data point
    """
    data = pd.read_csv(filename, header=0).values

    weeklyData = []
    y = [ None ]

    lastWeekDate = datetime.strptime(data[0, 0], "%Y-%m-%d")
    # row: date,open,high,low,close,volume,unadjustedVolume,change,changePercent,vwap,label,changeOverTime
    for row in data[1:]:
        currentDate = datetime.strptime(row[0], "%Y-%m-%d")
        if (currentDate - lastWeekDate).days >= 7:
            weeklyData.append(row[4])
            y.append(row[4])
            lastWeekDate = currentDate


    # To be concise
    X = np.array(weeklyData[1:])
    y = np.array(y[1:-1])

    return X, y

print(read_csv('./data/AAPL.csv'))
