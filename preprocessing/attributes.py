"""
attributes.py
-------------
Defines sensitive attributes (individual and intersectional) for the
CD-AFMS causal diagnosis pipeline, starting from the raw ML-100K user
attributes DataFrame produced by load_data.load_user_attributes().

Individual attributes supported:
    - gender        : binary (M/F) -> encoded 0/1
    - age_group     : 4 fixed bins -> <25, 25-34, 35-49, 50+

Intersectional attribute:
    - gender x age_group -> up to 8 groups

Design choice (default, can be revisited): fixed age bins rather than
quartiles, so bin edges are interpretable in the paper (e.g. "<25")
rather than data-dependent cutoffs like "27.5". If a data-driven split
is preferred later, swap AGE_BINS/AGE_LABELS for a quantile-based cut.
"""

import pandas as pd

AGE_BINS = [0, 25, 35, 50, 200]
AGE_LABELS = ["<25", "25-34", "35-49", "50+"]

# Minimum group size below which a group is flagged as "small" -- i.e. the
# causal diagnosis test (Section: independence_test.py, not yet built) is
# unlikely to have enough power and Bonferroni/FDR correction will need to
# treat it with extra caution. This threshold is a placeholder; we will
# revisit it once we run the power analysis in causal_diagnosis/simulation.py.
MIN_GROUP_SIZE = 20


def add_gender_code(user_attrs: pd.DataFrame) -> pd.DataFrame:
    """Add a 'gender_code' column: M -> 0, F -> 1."""
    df = user_attrs.copy()
    mapping = {"M": 0, "F": 1}
    if not set(df["gender"].unique()).issubset(mapping.keys()):
        raise ValueError(
            f"Unexpected gender values: {set(df['gender'].unique()) - set(mapping.keys())}. "
            "Expected only 'M' or 'F' for ML-100K."
        )
    df["gender_code"] = df["gender"].map(mapping)
    return df


def add_age_group(
    user_attrs: pd.DataFrame, bins: list = AGE_BINS, labels: list = AGE_LABELS
) -> pd.DataFrame:
    """Add an 'age_group' column (categorical) and 'age_group_code' (0-indexed int)."""
    df = user_attrs.copy()
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    df["age_group_code"] = df["age_group"].cat.codes
    return df


def build_intersectional_groups(
    user_attrs: pd.DataFrame,
    attribute_cols: list = ("gender_code", "age_group_code"),
) -> pd.DataFrame:
    """
    Combine the given attribute columns into a single intersectional group id.
    Requires add_gender_code() and add_age_group() to have been called first.

    Adds:
        - 'group_id'    : integer id, one per unique combination of attribute_cols
        - 'group_label' : human-readable label, e.g. "F_25-34"
    """
    df = user_attrs.copy()
    for col in attribute_cols:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. Call add_gender_code()/add_age_group() first."
            )

    # Build a readable label per row (uses age_group text label if available, else raw values)
    def _row_label(row):
        parts = []
        for col in attribute_cols:
            if col == "gender_code":
                parts.append("M" if row[col] == 0 else "F")
            elif col == "age_group_code":
                parts.append(str(row.get("age_group", row[col])))
            else:
                parts.append(str(row[col]))
        return "_".join(parts)

    df["group_label"] = df.apply(_row_label, axis=1)

    # Assign a stable integer id per unique combination (sorted for reproducibility)
    unique_labels = sorted(df["group_label"].unique())
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    df["group_id"] = df["group_label"].map(label_to_id)

    return df


def compute_group_sizes(
    user_attrs: pd.DataFrame, group_col: str, label_col: str = None
) -> pd.DataFrame:
    """
    Return a DataFrame with one row per group: group id/label, size, and
    a 'small_group' flag (size < MIN_GROUP_SIZE) that downstream causal
    diagnosis code should treat conservatively.

    label_col: which column to use as the human-readable label for each
    group (e.g. 'gender', 'age_group', 'group_label'). Defaults to group_col
    itself if not given.
    """
    if label_col is None:
        label_col = group_col
    sizes = (
        user_attrs.groupby(group_col)
        .agg(group_label=(label_col, "first"), size=(group_col, "count"))
        .reset_index()
        .sort_values("size", ascending=False)
        .reset_index(drop=True)
    )
    sizes["small_group"] = sizes["size"] < MIN_GROUP_SIZE
    return sizes


def prepare_all_attributes(user_attrs: pd.DataFrame) -> dict:
    """
    Convenience pipeline: run all steps and return a dict with the
    enriched DataFrame plus individual and intersectional group-size summaries.
    """
    df = add_gender_code(user_attrs)
    df = add_age_group(df)
    df = build_intersectional_groups(df, attribute_cols=("gender_code", "age_group_code"))

    gender_sizes = compute_group_sizes(df, group_col="gender_code", label_col="gender")
    age_sizes = compute_group_sizes(df, group_col="age_group_code", label_col="age_group")
    intersectional_sizes = compute_group_sizes(df, group_col="group_id", label_col="group_label")

    return {
        "user_attrs": df,
        "gender_group_sizes": gender_sizes,
        "age_group_sizes": age_sizes,
        "intersectional_group_sizes": intersectional_sizes,
    }


if __name__ == "__main__":
    # Quick sanity check when run directly, chained with load_data.py:
    #   python preprocessing/attributes.py
    from load_data import load_and_prepare

    data = load_and_prepare()
    result = prepare_all_attributes(data["user_attrs"])

    print("Enriched user_attrs sample:")
    print(result["user_attrs"].head())
    print()
    print("Gender group sizes:")
    print(result["gender_group_sizes"])
    print()
    print("Age group sizes:")
    print(result["age_group_sizes"])
    print()
    print("Intersectional (gender x age) group sizes:")
    print(result["intersectional_group_sizes"])
