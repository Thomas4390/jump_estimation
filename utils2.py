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
            plt.title(f'Naive Estimation of QV')
            method = 'naive'
        else:
            plt.title(f'Subsampling Estimation of QV')
            method = 'subsampling'

        plt.plot(RV, label=f'{method} QV | k = {timestep} min')
        plt.xlabel('Days')
        plt.ylabel('QV')
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
            plt.title(f'Naive Estimation of IV')
            method = 'naive'
        else:
            plt.title(f'Subsampling Estimation of IV')
            method = 'subsampling'

        plt.plot(BV, label=f'{method} IV | k = {timestep} min')
        plt.xlabel('Days')
        plt.ylabel('BV')
        plt.legend()
        plt.savefig(f"images2/{annee[5:9]}_{method}_IV.png")


    return None


