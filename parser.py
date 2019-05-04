#!/usr/bin/env python

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

def daily_parse(filename):
    """
    Opens and parses/processes csv files to return data and labels

    Args:
        filename - String containing a csv file's directory

    Returns:
        X - Data to be used as input features for the neural net
        y - The labels for each data point
    """
    data = pd.read_csv(f"./data/{filename}.csv", header=0).values
    pd.DataFrame(data[:, :5]).to_csv(f"./parsed/{filename}.csv", index=False, header=False, mode='w')

def weekly_parse(filename):

    data = pd.read_csv(f"./data/{filename}.csv", header=0).values
    weeklyData = []

    lastWeekDate = datetime.strptime(data[0, 0], "%Y-%m-%d")
    # row: date,open,high,low,close,volume,unadjustedVolume,change,changePercent,vwap,label,changeOverTime
    for row in data[:, :5]:
        currentDate = datetime.strptime(row[0], "%Y-%m-%d")
        if (currentDate - lastWeekDate).days >= 7:
            weeklyData.append(row)
            lastWeekDate = currentDate

    pd.DataFrame(weeklyData).to_csv(f"./parsed/{filename}.csv", index=False, header=False, mode='w')
    