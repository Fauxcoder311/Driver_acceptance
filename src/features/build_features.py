import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.transformations import (
    driver_distance_to_pickup,
    DriverHistoricalCompletedBookings,
    hour_of_day,
    day_of_week,
)
from src.utils.store import AssignmentStore

driver_historical = DriverHistoricalCompletedBookings()


def main():
    store = AssignmentStore()

    dataset = store.get_processed("dataset.csv")
    driver_historical.build(dataset)
    dataset = apply_feature_engineering(dataset)

    store.put_processed("transformed_dataset.csv", dataset)


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pipe(driver_distance_to_pickup)
        .pipe(hour_of_day)
        .pipe(day_of_week)
        .pipe(driver_historical)
    )


if __name__ == "__main__":
    main()
