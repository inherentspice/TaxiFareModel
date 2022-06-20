import pandas as pd

# PATH_1k = "/home/inherentspice/code/inherentspice/TaxiFareModel/raw_data/train_1k.csv"
# PATH_10K = "/home/inherentspice/code/inherentspice/TaxiFareModel/raw_data/train_10k.csv"
PATH_10K_PLUS = "/home/inherentspice/code/inherentspice/TaxiFareModel/raw_data/train.csv"


def get_data(nrows=10_000):
    '''returns a DataFrame with nrows from s3 bucket'''
    # if nrows > 10_000:
    df = pd.read_csv(PATH_10K_PLUS, nrows=nrows)
    # if nrows <=10_000 and nrows > 1_000:
    #     df = pd.read_csv(PATH_10K, nrows=nrows)
    # else:
    #     df = pd.read_csv(PATH_1k, nrows=nrows)

    return df


def clean_data(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df


if __name__ == '__main__':
    df = get_data()
