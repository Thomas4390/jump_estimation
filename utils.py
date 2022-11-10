import datetime as dt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from numpy import ndarray


def read_range_days(
        df: pd.DataFrame, date: str, days: int = 1, hours: int = 0,
        minutes: int = 0
) -> pd.DataFrame:
    """
    Read a specific day in the data
    :param df: dataframe
    :param date: date to read in the following format: 'YYYY-MM-DD'
    :param days: number of days to read
    :param hours: number of hours to read
    :param minutes: number of minutes to read
    :return: dataframe with the data of the day
    """
    try:
        year, month, day = map(int, date.split("-"))
        # Create a datetime object
        start_date = dt.datetime(year, month, day)
        end_date = dt.datetime(year, month, day) + dt.timedelta(
            days=days, hours=hours, minutes=minutes
        )
        df_range = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
    except Exception as e:
        print(
            "Error in the date format or selected period. \
              Please use the following format: YYYY-MM-DD"
        )
        print(e)
        df_range = pd.DataFrame()

    return df_range


def compute_total_return(df: pd.DataFrame) -> float:
    """
    Compute the total returns of the day
    :param df: dataframe
    :return: total returns
    """

    total_return = np.log(df["Close"].iloc[-1] / df["Close"].iloc[0])

    return total_return


def plot_price_data(df: pd.DataFrame) -> None:
    """
    Plot the figure
    :param df: dataframe
    """

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
            )
        ]
    )

    fig.show()

    return None


def plot_number_of_trades_per_day(df: pd.DataFrame) -> None:
    """
    Plot the number of trades
    :param df: dataframe
    """
    # Plot the first 30 data points in blue

    trace1 = go.Scatter(x=df["Date"].iloc[0:30], y=df["NbTrade"].iloc[0:30])
    trace2 = go.Scatter(x=df["Date"].iloc[30:], y=df["NbTrade"].iloc[30:])

    fig = go.Figure(data=[trace1, trace2])
    fig.show()

    return None


def compute_naive_qv(df: pd.DataFrame, n: int = 390, k: int = 1) -> ndarray:
    """
    Compute the naive Quadratic Variation
    :param df: dataframe
    :param n: number of minutes in one day
    :param k: time step
    :return: Quadratic Variation
    """
    n_days = len(df) // n
    n_k = np.floor(n / k)

    naives_qv = np.zeros(n_days)

    for day in range(n_days):
        for i in range(1, int(n_k)):
            naives_qv[day] += \
                (np.log(df["Close"].iloc[day * n + i * k])
                 - (np.log(df["Close"].iloc[day * n + (i - 1) * k]))) ** 2

    return naives_qv


def plot_naive_qv(df: pd.DataFrame, k_list: list[int]) -> None:
    """
    Plot the naive Quadratic Variation
    :param df: dataframe
    :param k_list: list of minutes returns
    """
    for k in k_list:
        rvs = compute_naive_qv(df, k=k)

        plt.plot(
            rvs,
            label=f"{df['Date'].iloc[0].year} Naive QV \
         k = {k} minute(s)",
        )

    plt.xlabel("Days")
    plt.ylabel("Daily estimated Quadratic Variation")
    plt.legend()
    plt.title(
        f"{df['Date'].iloc[0].year} \
    Naive Quadratic Variation for various time step"
    )
    plt.savefig(f"images/{df['Date'].iloc[0].year}_Naive_QV.png")
    plt.show()
    plt.close()

    return None


def compute_subsampling_qv(df: pd.DataFrame, n: int = 390,
                           k: int = 1) -> ndarray:
    """
    Compute the Subsampling Quadratic Variance
    :param df: dataframe
    :param n: number of minutes in one day
    :param k: time step
    :return: subsampling Quadratic Variation
    """
    n_days = len(df) // n
    n_k = np.floor(n / k)
    rv_subsampling = np.zeros((n_days, k))

    for subgrid in range(k):
        for day in range(n_days):
            for i in range(1, int(n_k)):
                rv_subsampling[day, subgrid] += \
                    (np.log(df["Close"].iloc[day * n + i * k + subgrid])
                     - (np.log(df["Close"].iloc[
                                   day * n + (i - 1) * k + subgrid]))) ** 2

    mean_rv_subsampling = np.mean(rv_subsampling, axis=1)

    return mean_rv_subsampling





def plot_subsampling_qv(df: pd.DataFrame, k_list: list[int]) -> None:
    """
    Plot the realized variance
    :param df: dataframe
    :param k_list: list of time steps
    """
    for k in k_list:
        srv = compute_subsampling_qv(df, k=k)
        plt.plot(
            srv,
            label=f"{df['Date'].iloc[0].year} Subsampling QV \
         k = {k} minute(s)",
        )

    plt.xlabel("Days")
    plt.ylabel("Daily estimated Quadratic Variation")
    plt.legend()
    plt.title(
        f"{df['Date'].iloc[0].year} \
    Subsampling Quadratic Variation for various time steps"
    )
    plt.savefig(f"images/{df['Date'].iloc[0].year}_Subsampling_QV.png")
    plt.show()
    plt.close()

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


def compute_naive_bv(df: pd.DataFrame, n: int = 390, k: int = 1) -> ndarray:
    """
    Compute the naive bipower variance
    :param df: dataframe
    :param n: number of minutes in one day
    :param k: time step
    :return: naive bipower variance
    """
    n_days = len(df) // n
    n_k = np.floor(n / k)

    naives_bv = np.zeros(n_days)

    for day in range(n_days):
        for i in range(2, int(n_k)):
            naives_bv[day] += np.abs(
                ((np.log(df["Close"].iloc[day * n + i * k])
                 - (np.log(df["Close"].iloc[day * n + (i - 1) * k])))
                    * (np.log(df["Close"].iloc[day * n + (i - 1) * k])
                          - (np.log(df["Close"].iloc[day * n + (i - 2) * k]))))
            )
    naives_bv = (np.pi / 2) * naives_bv
    return naives_bv


def plot_naive_bv(df: pd.DataFrame, k_list: list[int]) -> None:
    """
    Plot the naive bipower variation
    :param df: dataframe
    :param k_list: list of minutes returns
    """
    for k in k_list:
        nbv = compute_naive_bv(df, k=k)

        plt.plot(
            nbv,
            label=f"{df['Date'].iloc[0].year} Naive BV \
         k = {k} minute(s)",
        )

    plt.xlabel("Days")
    plt.ylabel("Daily estimated Bipower Variation")
    plt.legend()
    plt.title(
        f"{df['Date'].iloc[0].year} \
    Naive Bipower Variation for various time step"
    )
    plt.savefig(f"images/{df['Date'].iloc[0].year}_Naive_BV.png")
    plt.show()
    plt.close()

    return None


def compute_subsampling_bv(df: pd.DataFrame, n: int = 390,
                           k: int = 1) -> ndarray:
    """
    Compute the subsampling bipower variance
    :param df: dataframe
    :param n: number of minutes in one day
    :param k: time step
    :return: subsampling bipower variance
    """
    n_days = len(df) // n
    n_k = np.floor(n / k)
    bv_subsampling = np.zeros((n_days, k))

    for subgrid in range(k):
        for day in range(n_days):
            for i in range(1, int(n_k)):
                bv_subsampling[day, subgrid] += np.abs(
                    ((np.log(df["Close"].iloc[day * n + i * k + subgrid])
                      - (np.log(df["Close"].iloc[day * n + (i - 1) * k + subgrid])))
                     * (np.log(df["Close"].iloc[day * n + (i - 1) * k + subgrid])
                        - (np.log(df["Close"].iloc[day * n + (i - 2) * k + subgrid]))))
                )

    bv_subsampling = (np.pi / 2) * bv_subsampling
    mean_bv_subsampling = np.mean(bv_subsampling, axis=1)

    return mean_bv_subsampling


def plot_subsampling_bv(df: pd.DataFrame, k_list: list[int]) -> None:
    """
    Plot the subsampling Bipower variation
    :param df: dataframe
    :param k_list: list of time steps
    """
    for k in k_list:
        sbv = compute_subsampling_bv(df, k=k)
        plt.plot(
            sbv,
            label=f"{df['Date'].iloc[0].year} Subsampling BV \
         k = {k} minute(s)",
        )

    plt.xlabel("Days")
    plt.ylabel("Daily estimated Bipower Variation")
    plt.legend()
    plt.title(
        f"{df['Date'].iloc[0].year} \
    Subsampling Bipower Variation for various time steps"
    )
    plt.savefig(f"images/{df['Date'].iloc[0].year}_Subsampling_BV.png")
    plt.show()
    plt.close()

    return None


def compute_sum_squared_jumps(df: pd.DataFrame, n: int = 390,
                              k: int = 1) -> ndarray:
    """
    Compute the estimated Sum of squared Jumps
    :param df: dataframe
    :param k: time step
    :return: Sum of squared Jumps"""

    n_days = len(df) // n
    sum_squared_jumps = np.zeros(n_days)

    naive_qv = compute_subsampling_qv(df, k=k)
    naive_bv = compute_subsampling_bv(df, k=k)

    diff = naive_qv - naive_bv

    for i in range(n_days):
        # Je passe par cette méthode car le maximum classique a du mal à fonctionner
        if diff[i] > 0:
            sum_squared_jumps[i] = diff[i]
        else:
            sum_squared_jumps[i] = 0

    return sum_squared_jumps


def plot_ssj(df: pd.DataFrame, k_list: list[int]) -> None:
    """
    Plot the sum of squared jumps
    :param df: dataframe
    :param k_list: list of minutes returns
    """
    for k in k_list:
        ssj = compute_sum_squared_jumps(df, k=k)

        plt.plot(
            ssj,
            label=f"{df['Date'].iloc[0].year} Sum Squared Jumps \
         k = {k} minute(s)",
        )

    plt.xlabel("Days")
    plt.ylabel("Daily estimated Sum Squared Jumps")
    plt.legend()
    plt.title(
        f"{df['Date'].iloc[0].year} \
    Sum Squared Jumps Variation for various time step"
    )
    plt.savefig(f"images/{df['Date'].iloc[0].year}_SSJ.png")
    plt.show()
    plt.close()

    return None
