import os
import pandas as pd

dtype_dict = {
    "tag_id": str,
    "sensor": str,
    "datetime": str,
    "value": float,
    "label": str,
}


def load_and_merge_data(data_dir):
    """
    Loads and merges all measurements_filtered.csv files in the data folder, adds conceptdoi (folder name),
    and joins scientific_name from tags.csv if available. Returns the merged DataFrame.
    """
    dfs = []
    for conceptdoi in os.listdir(data_dir):
        concept_path = os.path.join(data_dir, conceptdoi)
        if not os.path.isdir(concept_path):
            continue
        print(f"Processing record: {conceptdoi}")
        # Find version folder (should be only one)
        for version in os.listdir(concept_path):
            version_path = os.path.join(concept_path, version)
            if not os.path.isdir(version_path):
                continue
            print(f"  Version: {version}")
            # Find measurements_filtered.csv
            for f in os.listdir(version_path):
                if f.endswith("_filtered.csv"):
                    m_path = os.path.join(version_path, f)
                    print(f"    Loading: {f}")
                    df = pd.read_csv(
                        m_path,
                        dtype=dtype_dict,
                        keep_default_na=False,  # Prevent empty strings from being read as NaN
                        na_values={"label": []},  # Do not treat empty label as NaN
                    )
                    df["conceptdoi"] = conceptdoi
                    # Try to join scientific_name from tags.csv or tags.csv.gz
                    tags_path = None
                    for tname in ["tags.csv", "tags.csv.gz"]:
                        tpath = os.path.join(version_path, tname)
                        if os.path.exists(tpath):
                            tags_path = tpath
                            break
                    if tags_path:
                        print(f"    Joining tags from: {os.path.basename(tags_path)}")
                        if tags_path.endswith(".gz"):
                            tags_df = pd.read_csv(tags_path, compression="gzip")
                        else:
                            tags_df = pd.read_csv(tags_path)
                        # Only keep tag_id and scientific_name
                        tags_df = tags_df[["tag_id", "scientific_name"]]
                        df = df.merge(tags_df, on="tag_id", how="left")
                    else:
                        print(f"    No tags file found.")
                        df["scientific_name"] = None
                    dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def round_time(df):
    """
    Round datetime to the nearest 5 minutes and check that the temporal resolution is consistent for each tag_id and sensor.
    """
    dt_rounded = df["datetime"].dt.round("5min")

    # Find tag_id for which rounding changes the time by more than 2.5 minutes
    rounded_err = (dt_rounded - df["datetime"]).abs()
    rounded_err_id = rounded_err > pd.Timedelta("2min")

    # Count number of affected points per tag_id
    tag_counts = df.loc[rounded_err_id, "tag_id"].value_counts().reset_index()
    tag_counts.columns = ["tag_id", "n_changed"]
    tag_counts = tag_counts.sort_values("n_changed", ascending=False).reset_index(
        drop=True
    )

    # Display nicely
    print(tag_counts)

    df["datetime"] = dt_rounded

    return df


def get_tag_sensor_table(df):
    """
    Round datetime to the nearest 5 minutes and check that the temporal resolution is consistent for each tag_id and sensor.
    """
    # For each tag_id and sensor, get the most frequent diff (in minutes) and its count
    most_freq_diff = []
    for (tag_id, sensor), group in df.groupby(["tag_id", "sensor"]):
        group = group.sort_values("datetime")
        diffs = group["datetime"].diff().dropna().dt.total_seconds() / 60  # in minutes
        if not diffs.empty:
            vc = diffs.value_counts()
            most_common = vc.idxmax()
            count = vc.max()
            most_freq_diff.append(
                {
                    "tag_id": tag_id,
                    "sensor": sensor,
                    "resolution": most_common,
                    "count": count,
                    "count_non_resolution": len(diffs) - count,
                }
            )
    most_freq_diff_df = pd.DataFrame(most_freq_diff)

    return most_freq_diff_df


# Only run this if the script is executed directly
if __name__ == "__main__":
    data_dir = "data"

    df = load_and_merge_data(data_dir)

    df = round_time(df)

    df.to_csv(os.path.join(data_dir, "labels.csv"), index=False)

    tag_sensor = get_tag_sensor_table(df)
    tag_sensor.to_csv(os.path.join(data_dir, "tag_sensor.csv"), index=False)

    # Show those with non-most-frequent counts > 0
    problematic = tag_sensor[tag_sensor["count_non_resolution"] > 0]
    print(problematic)
