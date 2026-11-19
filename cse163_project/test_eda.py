"""Tests for the CSE 163 EDA functions."""

import pandas as pd
import numpy as np

from eda_analysis import (clean_data, weighted_mean, sleep_group,
                          age_group)


def test_groups():
    assert sleep_group(5) == "Under 6"
    assert sleep_group(7) == "6 to under 9"
    assert sleep_group(9) == "9 or more"
    assert sleep_group(np.nan) == "Unknown"
    assert age_group(20) == "18-34"
    assert age_group(40) == "35-54"
    assert age_group(60) == "55+"


def test_clean_data():
    sample = pd.DataFrame({
        "RIDAGEYR": [20, 40, 15],
        "SLD012": [5, 99, 8],
        "SLD013": [6, 8, 8],
        "BMXBMI": [22, 31, 25],
        "PAQ650": [1, 2, 2],
        "PAQ665": [2, 2, 2],
        "WTMEC2YR": [1, 2, 1]
    })
    result = clean_data(sample)
    assert len(result) == 2
    assert result.iloc[0]["sleep_group"] == "Under 6"
    assert pd.isna(result.iloc[1]["SLD012"])
    assert result.iloc[1]["sleep_group"] == "Unknown"
    assert result.iloc[0]["activity_group"] == "Active"
    assert result.iloc[1]["activity_group"] == "Inactive"


def test_weighted_mean():
    sample = pd.DataFrame({
        "BMXBMI": [20, 30, np.nan],
        "WTMEC2YR": [1, 3, 8]
    })
    assert np.isclose(weighted_mean(sample, "BMXBMI"), 27.5)


def test_survey_contrast():
    """Check a known weighted contrast with two strata and two PSUs each."""
    from result_validity import contrast

    sample = pd.DataFrame({
        "SDMVSTRA": [1, 1, 1, 1, 2, 2, 2, 2],
        "SDMVPSU": [1, 1, 2, 2, 1, 1, 2, 2],
        "WTMEC2YR": [1] * 8,
        "group": ["A", "B"] * 4,
        "value": [10, 5, 12, 6, 15, 8, 14, 7]
    })
    psus = sample[["SDMVSTRA", "SDMVPSU"]].drop_duplicates()
    result = contrast(sample, psus, "value",
                      [(1, "group", "A"), (-1, "group", "B")])
    assert result["difference"] == 6.25
    assert result["standard_error"] > 0
    assert result["df"] == 2
    assert result["ci_low"] < result["difference"]
    assert result["ci_high"] > result["difference"]


def main():
    test_groups()
    test_clean_data()
    test_weighted_mean()
    test_survey_contrast()
    print("All tests passed!")


if __name__ == "__main__":
    main()
