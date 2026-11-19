"""CSE 163 EDA: sleep, physical activity, BMI, and age."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


SLEEP_GROUPS = ["Under 6", "6 to under 9", "9 or more"]
AGE_GROUPS = ["18-34", "35-54", "55+"]


def load_data():
    """Read NHANES data and match participants with SEQN."""
    sleep = pd.read_sas("data/SLQ_J.xpt")
    body = pd.read_sas("data/BMX_J.xpt")
    activity = pd.read_sas("data/PAQ_J.xpt")
    demo = pd.read_sas("data/DEMO_J.xpt")

    print("Original file sizes:")
    print("Sleep:", sleep.shape)
    print("Body:", body.shape)
    print("Activity:", activity.shape)
    print("Demographics:", demo.shape)

    df = demo[["SEQN", "RIDAGEYR", "WTMEC2YR",
               "SDMVSTRA", "SDMVPSU"]].merge(
        sleep[["SEQN", "SLD012", "SLD013"]], on="SEQN"
    )
    df = df.merge(body[["SEQN", "BMXBMI"]], on="SEQN")
    df = df.merge(activity[["SEQN", "PAQ650", "PAQ665"]],
                  on="SEQN")
    return df


def sleep_group(hours):
    """Put valid weekday sleep hours into one of three groups."""
    if pd.isna(hours):
        return "Unknown"
    if hours < 6:
        return "Under 6"
    if hours < 9:
        return "6 to under 9"
    return "9 or more"


def age_group(age):
    """Group adult ages."""
    if age < 35:
        return "18-34"
    if age < 55:
        return "35-54"
    return "55+"


def activity_group(row):
    """Use yes/no recreational activity answers."""
    vigorous = row["PAQ650"]
    moderate = row["PAQ665"]
    if vigorous == 1 or moderate == 1:
        return "Active"
    if vigorous == 2 and moderate == 2:
        return "Inactive"
    return "Unknown"


def clean_data(df):
    """Keep adults and mark invalid responses as missing."""
    df = df[df["RIDAGEYR"] >= 18].copy()
    for column in ["SLD012", "SLD013"]:
        df.loc[(df[column] < 3) | (df[column] > 14), column] = np.nan
    df.loc[(df["BMXBMI"] <= 0), "BMXBMI"] = np.nan
    df.loc[(df["WTMEC2YR"] <= 0), "WTMEC2YR"] = np.nan
    for column in ["PAQ650", "PAQ665"]:
        df.loc[~df[column].isin([1, 2]), column] = np.nan

    df["sleep_group"] = df["SLD012"].apply(sleep_group)
    df["age_group"] = df["RIDAGEYR"].apply(age_group)
    df["activity_group"] = df.apply(activity_group, axis=1)
    return df


def weighted_mean(df, column):
    """Calculate an average, giving each participant their survey weight."""
    valid = df[[column, "WTMEC2YR"]].dropna()
    if len(valid) == 0 or valid["WTMEC2YR"].sum() == 0:
        return np.nan
    return np.average(valid[column], weights=valid["WTMEC2YR"])


def summarize(df):
    """Print descriptive statistics for the report."""
    print("\nMerged adult data:", df.shape)
    print("\nMissing data:")
    print(df.isna().sum())
    print("\nSeven-number summaries (and counts):")
    columns = ["SLD012", "SLD013", "BMXBMI", "RIDAGEYR"]
    print(df[columns].describe())

    for column in ["sleep_group", "age_group", "activity_group",
                   "PAQ650", "PAQ665"]:
        print("\nCounts for", column)
        print(df[column].value_counts(dropna=False))

    print("\nWeighted BMI by sleep group:")
    for name in SLEEP_GROUPS:
        group = df[df["sleep_group"] == name]
        print(name, round(weighted_mean(group, "BMXBMI"), 2),
              "participants:", group["BMXBMI"].count())

    print("\nWeighted sleep hours by activity group:")
    for name in ["Active", "Inactive"]:
        group = df[df["activity_group"] == name]
        print(name, round(weighted_mean(group, "SLD012"), 2),
              "participants:", group["SLD012"].count())


def plot_data(df):
    """Save four charts of the merged NHANES dataset."""
    plt.figure()
    df["SLD012"].dropna().hist(bins=12)
    plt.title("Weekday sleep duration")
    plt.xlabel("Hours of sleep")
    plt.ylabel("Number of participants")
    plt.tight_layout()
    plt.savefig("results/sleep_histogram.png")
    plt.close()

    bmi_means = []
    for name in SLEEP_GROUPS:
        group = df[df["sleep_group"] == name]
        bmi_means.append(weighted_mean(group, "BMXBMI"))
    plt.figure()
    plt.bar(SLEEP_GROUPS, bmi_means)
    plt.title("BMI by weekday sleep duration")
    plt.xlabel("Sleep group")
    plt.ylabel("Weighted average BMI")
    plt.tight_layout()
    plt.savefig("results/sleep_bmi.png")
    plt.close()

    activity_means = []
    for name in ["Active", "Inactive"]:
        group = df[df["activity_group"] == name]
        activity_means.append(weighted_mean(group, "SLD012"))
    plt.figure()
    plt.bar(["Active", "Inactive"], activity_means)
    plt.title("Weekday sleep by recreational activity")
    plt.xlabel("Recreational activity")
    plt.ylabel("Weighted average sleep hours")
    plt.tight_layout()
    plt.savefig("results/activity_sleep.png")
    plt.close()

    age_means = []
    for age in AGE_GROUPS:
        row = []
        for sleep in SLEEP_GROUPS:
            group = df[(df["age_group"] == age) &
                       (df["sleep_group"] == sleep)]
            row.append(weighted_mean(group, "BMXBMI"))
        age_means.append(row)

    positions = [0, 1, 2]
    plt.figure()
    for i in range(3):
        x = [pos + i * 0.25 for pos in positions]
        heights = [row[i] for row in age_means]
        plt.bar(x, heights, width=0.25, label=SLEEP_GROUPS[i])
    plt.xticks([0.25, 1.25, 2.25], AGE_GROUPS)
    plt.title("BMI by age group and weekday sleep")
    plt.xlabel("Age group")
    plt.ylabel("Weighted average BMI")
    plt.legend(title="Sleep group")
    plt.tight_layout()
    plt.savefig("results/age_sleep_bmi.png")
    plt.close()


def main():
    df = clean_data(load_data())
    summarize(df)
    plot_data(df)
    from result_validity import run_tests
    run_tests(df)
    print("\nFour graphs were saved to the results folder.")


if __name__ == "__main__":
    main()
