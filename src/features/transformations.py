from typing import Any
import pandas as pd
from haversine import haversine

from src.utils.time import (
    robust_hour_of_iso_date,
    robust_day_of_iso_date,
    robust_weekday_of_iso_date,
)


def driver_distance_to_pickup(df: pd.DataFrame) -> pd.DataFrame:
    df["driver_distance"] = df.apply(
        lambda r: haversine(
            (r["driver_latitude"], r["driver_longitude"]),
            (r["pickup_latitude"], r["pickup_longitude"]),
        ),
        axis=1,
    )
    return df


def hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    df["event_hour"] = df["event_timestamp"].apply(robust_hour_of_iso_date)
    return df


def day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    df["event_weekday"] = df["event_timestamp"].apply(robust_weekday_of_iso_date)
    return df


class DriverHistoricalCompletedBookings:
    def __init__(self) -> None:
        self.save_path = "data/processed/driver_history_data.csv"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        driver_history_df = pd.read_csv(self.save_path)

        df["event_day"] = df["event_timestamp"].apply(robust_day_of_iso_date)

        df = df.merge(driver_history_df, on=["driver_id", "event_day"], how="left")
        df = df.drop(columns=["event_day"])
        df = df.fillna(0.0)
        return df

    def build(self, df: pd.DataFrame):
        df["event_day"] = df["event_timestamp"].apply(robust_day_of_iso_date)
        driver_day_events = (
            df[df["participant_status"].isin(["ACCEPTED", "IGNORED", "REJECTED"])]
            .groupby(["driver_id", "event_day", "participant_status"], as_index=False)
            .agg(num_orders=pd.NamedAgg("order_id", lambda x: len(set(x))))
        )
        driver_day_events = driver_day_events.pivot(
            index=["driver_id", "event_day"],
            columns=["participant_status"],
            values=["num_orders"],
        )
        driver_day_events.columns = [
            "_".join(a) for a in driver_day_events.columns.to_flat_index()
        ]

        driver_day_events = driver_day_events.reset_index(drop=False)
        driver_day_events.fillna(value=0.0, inplace=True)

        driver_day_total = df.groupby(["driver_id", "event_day"], as_index=False).agg(
            num_orders_TOTAL=pd.NamedAgg("order_id", lambda x: len(set(x)))
        )

        driver_day_events = driver_day_events.merge(
            driver_day_total, on=["driver_id", "event_day"], how="inner"
        )
        driver_day_events["acceptance_rate"] = (
            driver_day_events["num_orders_ACCEPTED"]
            / driver_day_events["num_orders_TOTAL"]
        )
        driver_day_events["rejection_rate"] = (
            driver_day_events["num_orders_REJECTED"]
            / driver_day_events["num_orders_TOTAL"]
        )
        driver_day_events["ignorance_rate"] = (
            driver_day_events["num_orders_IGNORED"]
            / driver_day_events["num_orders_TOTAL"]
        )

        driver_day_events = driver_day_events.merge(
            driver_day_events, on=["driver_id"], how="left"
        )
        driver_day_events = driver_day_events.loc[
            driver_day_events["event_day_x"] > driver_day_events["event_day_y"]
        ]
        driver_day_events = driver_day_events.groupby(
            by=["driver_id", "event_day_x"], as_index=False
        ).agg(
            acceptance_rate=pd.NamedAgg("acceptance_rate_y", "mean"),
            rejection_rate=pd.NamedAgg("rejection_rate_y", "mean"),
            ignorance_rate=pd.NamedAgg("ignorance_rate_y", "mean"),
            num_orders=pd.NamedAgg("num_orders_TOTAL_y", "sum"),
        )
        driver_day_events = driver_day_events.rename(
            columns={"event_day_x": "event_day"}
        )
        driver_day_events.to_csv(self.save_path, header=True, index=None)
