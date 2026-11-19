"""Design-aware comparisons for the NHANES CSE 163 final project.

Uses a Taylor-linearized standard error calculated across survey PSUs.
This is an educational implementation, not a replacement for dedicated
survey-analysis software. The full DEMO file supplies all sampled PSUs.
"""

import pandas as pd
import numpy as np
from scipy.stats import t


# Each comparison describes a weighted difference of group means.
# A positive result means the first group has a higher mean.
TESTS = [
    ("Q1: BMI, under 6 vs 6 to under 9 hours", "BMXBMI",
     [(1, "sleep_group", "Under 6"),
      (-1, "sleep_group", "6 to under 9")]),
    ("Q2: sleep, active vs inactive", "SLD012",
     [(1, "activity_group", "Active"),
      (-1, "activity_group", "Inactive")]),
    ("Q3: short-sleep BMI difference, 18-34 vs 55+", "BMXBMI",
     [(1, "age_sleep", "18-34|Under 6"),
      (-1, "age_sleep", "18-34|6 to under 9"),
      (-1, "age_sleep", "55+|Under 6"),
      (1, "age_sleep", "55+|6 to under 9")])
]


def survey_psus():
    """Get every sampled stratum/PSU pair, including outside our domain."""
    demo = pd.read_sas("data/DEMO_J.xpt")
    return demo[["SDMVSTRA", "SDMVPSU"]].dropna().drop_duplicates()


def make_groups(df):
    """Add a label for each combined age/sleep group."""
    df = df.copy()
    df["age_sleep"] = df["age_group"] + "|" + df["sleep_group"]
    return df


def group_influence(df, outcome, group_column, group_name):
    """Compute a group's weighted mean and PSU linearization values."""
    valid = df[(df[group_column] == group_name) &
               df[outcome].notna() &
               df["WTMEC2YR"].notna() &
               (df["WTMEC2YR"] > 0) &
               df["SDMVSTRA"].notna() &
               df["SDMVPSU"].notna()].copy()
    if len(valid) < 2:
        raise ValueError("Too few usable rows in " + group_name)

    weights = valid["WTMEC2YR"]
    mean = (valid[outcome] * weights).sum() / weights.sum()
    valid["influence"] = weights * (valid[outcome] - mean) / weights.sum()
    psus = valid.groupby(["SDMVSTRA", "SDMVPSU"],
                         as_index=False)["influence"].sum()
    return mean, len(valid), psus


def contrast(df, psu_frame, outcome, terms):
    """Estimate a contrast and its Taylor-linearized sampling error."""
    psus = psu_frame.copy()
    psus["influence"] = 0.0
    difference = 0.0
    counts = []

    for sign, column, value in terms:
        mean, count, psu_values = group_influence(
            df, outcome, column, value)
        difference += sign * mean
        counts.append(count)
        psus = psus.merge(psu_values, on=["SDMVSTRA", "SDMVPSU"],
                          how="left", suffixes=("", "_add"))
        psus["influence_add"] = psus["influence_add"].fillna(0)
        psus["influence"] += sign * psus["influence_add"]
        psus = psus.drop(columns="influence_add")

    variance = 0.0
    strata_count = 0
    psu_count = 0
    for _, group in psus.groupby("SDMVSTRA"):
        number = len(group)
        if number < 2:
            raise ValueError("A stratum has fewer than two PSUs")
        average = group["influence"].mean()
        variance += number / (number - 1) * (
            (group["influence"] - average) ** 2).sum()
        strata_count += 1
        psu_count += number

    degrees_freedom = psu_count - strata_count
    se = np.sqrt(variance)
    if se == 0:
        raise ValueError("Estimated standard error is zero")
    statistic = difference / se
    p_value = 2 * t.sf(abs(statistic), degrees_freedom)
    critical = t.ppf(0.975, degrees_freedom)
    return {
        "difference": difference,
        "standard_error": se,
        "ci_low": difference - critical * se,
        "ci_high": difference + critical * se,
        "t": statistic,
        "p": p_value,
        "df": degrees_freedom,
        "group_counts": counts,
        "strata": strata_count,
        "psus": psu_count
    }


def check_assumptions(df, psus):
    """Display observable checks; some design assumptions cannot be tested."""
    print("\nSurvey design / assumptions check")
    print("Strata:", psus["SDMVSTRA"].nunique())
    print("PSUs:", len(psus))
    print("PSUs per stratum:")
    print(psus.groupby("SDMVSTRA").size().value_counts())
    print("Adult rows:", len(df))
    print("Positive examination weights:", (df["WTMEC2YR"] > 0).sum())
    print("Missing weekday sleep:", df["SLD012"].isna().sum())
    print("Missing BMI:", df["BMXBMI"].isna().sum())
    print("Unknown recreational activity:",
          (df["activity_group"] == "Unknown").sum())
    print("\nAssumptions not directly testable from these files:")
    print("  Sample selection follows the NHANES survey design.")
    print("  PSU contributions are independent between design strata.")
    print("  Missing data do not cause major uncorrected bias.")
    print("  Linearization gives a useful large-sample approximation.")


def run_tests(df):
    """Run the three pre-specified hypothesis tests and save results."""
    df = make_groups(df)
    psus = survey_psus()
    check_assumptions(df, psus)
    rows = []
    for name, outcome, terms in TESTS:
        result = contrast(df, psus, outcome, terms)
        rows.append({"comparison": name, **result})
        print("\n" + name)
        print("Difference:", round(result["difference"], 3))
        print("95% CI:", round(result["ci_low"], 3), "to",
              round(result["ci_high"], 3))
        print("p-value:", round(result["p"], 4))
        print("Group sample sizes:", result["group_counts"])
    pd.DataFrame(rows).to_csv("results/result_validity.csv", index=False)
    print("\nSaved results/result_validity.csv")


if __name__ == "__main__":
    from eda_analysis import clean_data, load_data
    run_tests(clean_data(load_data()))
