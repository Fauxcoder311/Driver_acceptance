import pandas as pd

from src.utils.config import load_config
from src.utils.store import AssignmentStore
from src.utils.time import robust_iso_date


def main():
    store = AssignmentStore()
    config = load_config()

    booking_df = store.get_raw("booking_log.csv")
    booking_df = clean_booking_df(booking_df)

    participant_df = store.get_raw("participant_log.csv")
    participant_df = clean_participant_df(participant_df)

    dataset = merge_dataset(booking_df, participant_df)
    dataset = create_target(dataset, config["target"])

    store.put_processed("dataset.csv", dataset)


def clean_booking_df(df: pd.DataFrame) -> pd.DataFrame:
    unique_columns = [
        "order_id",
        "trip_distance",
        "pickup_latitude",
        "pickup_longitude",
    ]
    df = df.drop_duplicates(subset=unique_columns)
    return df[unique_columns]


def clean_participant_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    return df


def merge_dataset(bookings: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(participants, bookings, on="order_id", how="left")
    df["event_ts"] = df["event_timestamp"].apply(robust_iso_date)
    df = df.sort_values(by="event_ts", ascending=True)
    df = df.drop_duplicates(subset=["order_id", "driver_id"], keep="last")
    df = df.drop(columns=["event_ts"])

    return df


def create_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df[target_col] = df["participant_status"].apply(lambda x: int(x == "ACCEPTED"))
    return df


if __name__ == "__main__":
    main()
