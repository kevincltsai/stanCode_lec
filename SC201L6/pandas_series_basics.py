"""
File: pandas_series_basics.py
Name:
-----------------------------------
This file shows the basic pandas syntax, especially
on Series. Series is a single column
of data, similar to what a 1D array looks like.
We will be practicing creating a pandas Series and
call its attributes and methods
"""

import pandas as pd


def main():
    s = pd.Series([20,20,10])
    new_s = s.append(pd.Series([30,20,20]),ignore_index = True)
    print(new_s)

    new_s_2 = s.append(pd.Series([float('nan'),40,50]), ignore_index = True)

    print(new_s_2)

    print(new_s_2.argmax())
    print(new_s.max())

    print(new_s_2.value_counts())
if __name__ == '__main__':
    main()
