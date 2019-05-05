#!/usr/bin/env python

import pandas as pd
import numpy as np
from datetime import datetime

def daily_parse(filename):
    """
    Opens and parses the given filename inside ./data/ directory

    Args:
        filename: String containing a csv filename

    Returns:
        Nothing; writes parsed csv to ./parsed/ directory with same filename
    """
    data = pd.read_csv(f"./data/{filename}.csv", header=0).values
    pd.DataFrame(data[:, :5]).to_csv(f"./parsed/{filename}.csv", index=False, header=False, mode='w')

def weekly_parse(filename):
    """
    Opens and parses the given filename inside ./data/ directory

    Args:
        filename: String containing a csv filename

    Returns:
        Nothing; writes parsed csv to ./parsed/ directory with same filename
    """
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
    