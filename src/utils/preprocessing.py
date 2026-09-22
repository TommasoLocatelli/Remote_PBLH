def describe_data(data):
    print("Shape:", data.shape)

    print("\nColumns:", data.columns.tolist())

    print("\nColumn types:")
    print(data.dtypes)

    print("\nMissing values:")
    print(data.isna().sum())

    print("\nStats:")
    print(data.describe(include='all'))

def filter_data(data, 
                start_hour=12,
                end_hour=13,
                upper_bound=4000,
                lower_bound=0,
                cols=["time", "height", "altitude", "v", "beta", "beta_raw"]):

    # Time window (with +30 min offset)
    start = data["time"].dt.normalize().iloc[0] + pd.Timedelta(hours=start_hour, minutes=30)
    end   = data["time"].dt.normalize().iloc[0] + pd.Timedelta(hours=end_hour, minutes=30)

    # Apply time filter
    data = data[(data["time"] >= start) & (data["time"] <= end)]

    # Apply height filter
    data = data[data["height"] < upper_bound]
    data = data[data["height"] > lower_bound]

    # Select columns
    data = data[cols]

    return data
