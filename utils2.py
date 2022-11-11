import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt


def compute_rv(annee: str, naive: bool, timesteps: list[int]) -> None:
    df = pd.read_pickle(f'{annee}.pkl')
    df['datetime'] = pd.to_datetime(df.Date)
    df = df.loc[df['ContractName'] != "''", :]
    df = df.fillna(method='ffill')

    df['Date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.time
    df['RV'] = np.nan
    df['BV'] = np.nan
    dates = pd.unique(df.Date)

    for timestep in timesteps:
        dateVec = []
        RV = []
        BV = []
        for dateIdx, date in enumerate(dates):
            dateVec += [dates[dateIdx]]
            current_day = df.loc[df['Date'] == date, :]
            current_day.set_index('time', inplace=True)
            currentLogClose = np.log(current_day['Close']).shift(timestep)
            nextLogClose = np.log(current_day['Close'])
            currentLogReturn = nextLogClose - currentLogClose
            nextLogReturn = currentLogReturn.shift(-timestep)
            if naive:
                idxToKeep = list(map(lambda x: x % timestep == 0,
                                     list(range(currentLogReturn.shape[0]))))
                currentLogReturn = currentLogReturn.loc[idxToKeep]
                nextLogReturn = nextLogReturn.loc[idxToKeep]
                regulate = 1
            else:
                regulate = timestep

            RV += [1 / regulate * np.nansum(currentLogReturn ** 2)]
            BV += [1 / regulate * np.pi / 2 * np.nansum(
                np.abs(nextLogReturn * currentLogReturn))]

        if naive:
            plt.title(f'{annee[5:9]} | Naive Estimation of QV', fontsize=20)
            method = 'naive'
        else:
            plt.title(f'{annee[5:9]} | Subsampling Estimation of QV', fontsize=20)
            method = 'subsampling'

        plt.plot(RV, label=f'{method} QV | k = {timestep} min')
        plt.xlabel('Days', fontsize=20)
        plt.ylabel('QV', fontsize=20)
        plt.legend()
        plt.savefig(f"images2/{annee[5:9]}_{method}_QV.png")


    return None


def compute_bv(annee: str, naive: bool, timesteps: list[int]) -> None:
    df = pd.read_pickle(f'{annee}.pkl')
    df['datetime'] = pd.to_datetime(df.Date)
    df = df.loc[df['ContractName'] != "''", :]
    df = df.fillna(method='ffill')

    df['Date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.time
    df['RV'] = np.nan
    df['BV'] = np.nan
    dates = pd.unique(df.Date)

    for timestep in timesteps:
        dateVec = []
        RV = []
        BV = []
        for dateIdx, date in enumerate(dates):
            dateVec += [dates[dateIdx]]
            current_day = df.loc[df['Date'] == date, :]
            current_day.set_index('time', inplace=True)
            currentLogClose = np.log(current_day['Close']).shift(timestep)
            nextLogClose = np.log(current_day['Close'])
            currentLogReturn = nextLogClose - currentLogClose
            nextLogReturn = currentLogReturn.shift(-timestep)
            if naive:
                idxToKeep = list(map(lambda x: x % timestep == 0,
                                     list(range(currentLogReturn.shape[0]))))
                currentLogReturn = currentLogReturn.loc[idxToKeep]
                nextLogReturn = nextLogReturn.loc[idxToKeep]
                regulate = 1
            else:
                regulate = timestep

            RV += [1 / regulate * np.nansum(currentLogReturn ** 2)]
            BV += [1 / regulate * np.pi / 2 * np.nansum(
                np.abs(nextLogReturn * currentLogReturn))]

        if naive:
            plt.title(f'{annee[5:9]} | Naive Estimation of IV', fontsize=20)
            method = 'naive'
        else:
            plt.title(f'{annee[5:9]} | Subsampling Estimation of IV', fontsize=20)
            method = 'subsampling'

        plt.plot(BV, label=f'{method} IV | k = {timestep} min')
        plt.xlabel('Days', fontsize=20)
        plt.ylabel('BV', fontsize=20)
        plt.legend()
        plt.savefig(f"images2/{annee[5:9]}_{method}_IV.png")


    return None

def compute_jumps(annee: str, naive: bool, timesteps: list[int]) -> None:

    df = pd.read_pickle(f'{annee}.pkl')
    df['datetime'] = pd.to_datetime(df.Date)
    df = df.loc[df['ContractName']!="''", :]
    df = df.fillna(method='ffill')
    plt.rcParams['figure.figsize'] = (18, 6)

    df['Date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.time
    df['RV'] = np.nan
    df['BV'] = np.nan
    dates = pd.unique(df.Date)

    for timestep in timesteps:
        dateVec = []
        RV = []
        BV = []
        for dateIdx, date in enumerate(dates):
            dateVec += [dates[dateIdx]]
            current_day = df.loc[df['Date'] == date, :]
            current_day.set_index('time', inplace=True)
            currentLogClose = np.log(current_day['Close']).shift(timestep)
            nextLogClose = np.log(current_day['Close'])
            currentLogReturn = nextLogClose - currentLogClose
            nextLogReturn = currentLogReturn.shift(-timestep)
            if naive:
                idxToKeep = list(map(lambda x: x % timestep == 0,
                                     list(range(currentLogReturn.shape[0]))))
                currentLogReturn = currentLogReturn.loc[idxToKeep]
                nextLogReturn = nextLogReturn.loc[idxToKeep]
                regulate = 1
            else:
                regulate = timestep

            RV += [1/regulate * np.nansum(currentLogReturn**2)]
            BV += [1/regulate * np.pi/2 * np.nansum(np.abs(nextLogReturn*currentLogReturn))]

        JUMP = np.maximum(np.array(RV) - np.array(BV), np.zeros(len(RV)))

        if naive:
            plt.title(f'{annee[5:9]} | Naive Estimation of Jumps', fontsize=20)
            method = 'naive'
        else:
            plt.title(f'{annee[5:9]} |Subsampling Estimation of Jumps', fontsize=20)
            method = 'subsampling'

        plt.plot(JUMP, label=f'{method} Jumps | k = {timestep} min')
        plt.xlabel('Days', fontsize=20)
        plt.ylabel('Sum Square of Jumps', fontsize=20)
        plt.legend()
        plt.savefig(f"images2/{annee[5:9]}_{method}_Jumps.png")

    return None

def plot_historical_events(df: pd.DataFrame) -> None:
    """
    Plot the historical events for a given year
    :param df: dataframe
    """

    year = df["Date"].iloc[0].year
    n = 390

    if year == 1990:
        n_days = len(df) // n
        n_days_array = np.arange(n_days)
        y = np.full(n_days, np.nan)
        plt.plot(n_days_array, y)
        plt.axvline(x=20, color="orange", label="Black January",
                    linestyle="--")
        plt.axvline(x=149, color="blue", label="Golf War", linestyle="--")
        plt.axvline(x=162, color="aqua",
                    label="German Reunification Announced", linestyle="--")
        plt.axvline(x=188, color="red", label="Reunification", linestyle="--")

    elif year == 2001:
        n_days = len(df) // n
        n_days_array = np.arange(n_days)
        y = np.full(n_days, np.nan)
        plt.plot(n_days_array, y)
        plt.axvline(x=63, color="orange", label="Hainan Island Incident",
                    linestyle="--")
        plt.axvline(x=174, color="aqua", label="September 11 Attacks",
                    linestyle="--")
        plt.axvline(x=193, color="red", label="War on Terror", linestyle="--")
        plt.axvline(x=214, color="blue", label="Sabena Bankruptcy",
                    linestyle="--")

    elif year == 2007:
        n_days = len(df) // n
        n_days_array = np.arange(n_days)
        y = np.full(n_days, np.nan)
        plt.plot(n_days_array, y)
        plt.axvline(
            x=38,
            color="purple",
            label="US and China Stock Market Crash",
            linestyle="--",
        )
        plt.axvline(
            x=125, color="orange", label="Bank of England Emergency",
            linestyle="--"
        )
        plt.axvline(x=156, color="orchid", label="Subprimes Panic",
                    linestyle="--")
        plt.axvline(
            x=242, color="aqua", label="Delta Financial Bankruptcy",
            linestyle="--"
        )

    elif year == 2018:
        n_days = len(df) // n
        n_days_array = np.arange(n_days)
        y = np.full(n_days, np.nan)
        plt.plot(n_days_array, y)
        plt.axvline(
            x=24, color="purple", label="Stock Market Correction",
            linestyle="--"
        )
        plt.axvline(x=40, color="Gold", label="Trade War", linestyle="--")
        plt.axvline(x=87, color="red", label="Sanctions on Iran",
                    linestyle="--")
        plt.axvline(x=215, color="blue", label="BREXIT", linestyle="--")

    else:
        print("No historical events for this year")

    return None




