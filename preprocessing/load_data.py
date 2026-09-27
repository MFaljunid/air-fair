"""
load_data.py
------------
Loads the MovieLens-100K dataset (ratings + user demographic attributes)
and produces train/test splits for the CD-AFMS pilot.

Expected raw files (place them under data/raw/ml-100k/):
    u.data   -> user_id \t item_id \t rating \t timestamp
    u.user   -> user_id | age | gender | occupation | zip_code

Download source (do this manually, network access is restricted here):
    https://files.grouplens.org/datasets/movielens/ml-100k.zip
"""

import os
import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ml-100k")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_ratings(raw_dir: str = RAW_DIR) -> pd.DataFrame:
    """
    Load ratings into a DataFrame with columns: user_id, item_id, rating, timestamp.
    Auto-detects the format present in raw_dir:
        - ML-100K: u.data                          (tab-separated)
        - ML-1M:   ratings.dat                      ("::"-separated)
        - LastFM (360K): usersha1-artmbid-artname-plays.tsv
                   (tab-separated: user, artist-mbid, artist-name, plays;
                   NO per-interaction timestamp -- 'timestamp' is set to NaN
                   and train_test_split_leave_one_out() automatically falls
                   back to a random hold-out for this dataset, since a true
                   temporal "most recent interaction" cannot be determined.
                   'rating' here is the raw play count, not a 1-5 rating.)
    """
    path_100k = os.path.join(raw_dir, "u.data")
    path_1m = os.path.join(raw_dir, "ratings.dat")
    path_lastfm = os.path.join(raw_dir, "usersha1-artmbid-artname-plays.tsv")

    if os.path.exists(path_100k):
        return pd.read_csv(
            path_100k, sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
            engine="python",
        )
    elif os.path.exists(path_1m):
        return pd.read_csv(
            path_1m, sep="::",
            names=["user_id", "item_id", "rating", "timestamp"],
            engine="python",
        )
    elif os.path.exists(path_lastfm):
        df = pd.read_csv(
            path_lastfm, sep="\t",
            names=["user_id", "item_mbid", "item_name", "rating"],
            engine="python",
            na_filter=False,  # some artist-mbid fields are legitimately empty strings
            quoting=3,  # csv.QUOTE_NONE -- some artist names contain a bare " character
                        # that isn't a real quote/escape; without this, pandas'
                        # default CSV quote-handling misparses those rows entirely.
        )
        # Use artist name as the item identifier when mbid is missing/empty
        # (a nontrivial fraction of LastFM-360K rows have a blank mbid).
        df["item_id"] = df["item_mbid"].where(df["item_mbid"].astype(bool), df["item_name"])
        df = df[["user_id", "item_id", "rating"]].copy()
        df["timestamp"] = np.nan  # no per-interaction time in this file -- see docstring
        return df
    else:
        raise FileNotFoundError(
            f"Could not find u.data, ratings.dat, or usersha1-artmbid-artname-plays.tsv "
            f"in {raw_dir}. Download ml-100k.zip / ml-1m.zip from "
            "https://files.grouplens.org/datasets/movielens/ "
            "or the LastFM-360K dataset from "
            "http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html "
            "and extract it there."
        )


def load_user_attributes(raw_dir: str = RAW_DIR) -> pd.DataFrame:
    """
    Load user demographic attributes into a DataFrame with columns:
    user_id, age, gender, occupation, zip_code (normalized regardless of source format).
    Auto-detects the format present in raw_dir:
        - ML-100K: u.user    ("|"-separated, columns: user_id|age|gender|occupation|zip_code)
        - ML-1M:   users.dat ("::"-separated, columns: UserID::Gender::Age::Occupation::Zip-code
                   -- note ML-1M's column ORDER differs (gender before age); age is a
                   pre-binned code {1,18,25,35,45,50,56} rather than a raw integer, but
                   these codes fall inside attributes.py's existing AGE_BINS edges
                   [0,25,35,50,200] correctly by construction, so no downstream change
                   is needed for age_group assignment.)
        - LastFM (360K): usersha1-profile.tsv (tab-separated, header row:
                   #id, gender, age, country, registered). Gender is lowercase
                   'm'/'f' in the source file -- normalized to 'M'/'F' here to
                   match attributes.py's expected encoding. 'occupation' has no
                   LastFM equivalent (filled as "unknown"); 'zip_code' is filled
                   with the source 'country' field as the closest available
                   proxy. Ages in this dataset are known to be noisy (some
                   missing or implausible, e.g. negative) -- these are left
                   as-is rather than silently cleaned, so attributes.py's
                   age-binning will surface them as NaN age_group rather than
                   hiding a data-quality issue.
    """
    path_100k = os.path.join(raw_dir, "u.user")
    path_1m = os.path.join(raw_dir, "users.dat")
    path_lastfm = os.path.join(raw_dir, "usersha1-profile.tsv")

    if os.path.exists(path_100k):
        return pd.read_csv(
            path_100k, sep="|",
            names=["user_id", "age", "gender", "occupation", "zip_code"],
            engine="python",
        )
    elif os.path.exists(path_1m):
        df = pd.read_csv(
            path_1m, sep="::",
            names=["user_id", "gender", "age", "occupation", "zip_code"],
            engine="python",
        )
        # Reorder columns to the normalized schema so downstream code (attributes.py,
        # everything else) never needs to know which source dataset this came from.
        return df[["user_id", "age", "gender", "occupation", "zip_code"]]
    elif os.path.exists(path_lastfm):
        # LastFM-360K mirrors are inconsistent about whether this file has a
        # header row ("#id\tgender\tage\tcountry\tregistered") or starts
        # directly with data. Peek at the first line to detect which.
        with open(path_lastfm, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
        first_field = first_line.split("\t")[0].strip()
        has_header = first_field.lstrip("#").lower() in ("id", "userid", "user_id")

        column_names = ["user_id", "gender", "age", "country", "registered"]
        if has_header:
            df = pd.read_csv(path_lastfm, sep="\t", header=0, engine="python", quoting=3)
            df.columns = [c.strip().lstrip("#").lower() for c in df.columns]
            df = df.rename(columns={"id": "user_id"})
        else:
            df = pd.read_csv(
                path_lastfm, sep="\t", header=None, names=column_names,
                engine="python", quoting=3,
            )

        if "user_id" not in df.columns or "gender" not in df.columns or "age" not in df.columns:
            raise ValueError(
                f"Unexpected columns in {path_lastfm}: {list(df.columns)}. "
                "Expected user_id, gender, age, country, registered (in some order)."
            )
        df["gender"] = df["gender"].astype(str).str.upper()
        # LastFM-360K has many users who left gender blank/unspecified (a
        # documented characteristic of this dataset, not a parsing error).
        # Drop those rows here rather than letting attributes.py's stricter
        # M/F-only validation crash -- print the count so it's visible.
        valid_gender_mask = df["gender"].isin(["M", "F"])
        n_dropped = (~valid_gender_mask).sum()
        if n_dropped > 0:
            print(f"(load_user_attributes) dropping {n_dropped} users with missing/invalid "
                  f"gender (LastFM-360K commonly has unspecified gender)")
        df = df[valid_gender_mask].reset_index(drop=True)

        # LastFM-360K also has many users with missing, zero, negative, or
        # implausibly large ages (a second documented data-quality issue,
        # independent of the gender one above). Left unfiltered, these
        # produce a spurious "nan" age bucket downstream (attributes.py's
        # age-binning turns an invalid age into NaN, which then silently
        # gets misclassified as "not the target age group" rather than
        # excluded) -- caught via a diagnostic group-size check, not by
        # design, so we fix it at the source here.
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        valid_age_mask = df["age"].notna() & (df["age"] > 0) & (df["age"] < 100)
        n_age_dropped = (~valid_age_mask).sum()
        if n_age_dropped > 0:
            print(f"(load_user_attributes) dropping {n_age_dropped} users with missing/invalid "
                  f"age (LastFM-360K commonly has blank, zero, or implausible ages)")
        df = df[valid_age_mask].reset_index(drop=True)

        df["occupation"] = "unknown"
        df["zip_code"] = df.get("country", pd.NA)
        return df[["user_id", "age", "gender", "occupation", "zip_code"]]
    else:
        raise FileNotFoundError(
            f"Could not find u.user, users.dat, or usersha1-profile.tsv in {raw_dir}. "
            "Download ml-100k.zip / ml-1m.zip from https://files.grouplens.org/datasets/movielens/ "
            "or the LastFM-360K dataset from "
            "http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html "
            "and extract it there."
        )


def k_core_filter(
    ratings: pd.DataFrame, k: int = None,
    min_user_interactions: int = None, min_item_interactions: int = None,
) -> pd.DataFrame:
    """
    Iterative k-core filtering: repeatedly drops users and items below their
    respective minimum interaction counts until the dataset stabilizes
    (removing one side can push counts on the other side below threshold
    too, so a single pass isn't enough). Standard preprocessing step in CF
    papers to control density -- e.g. FairIR's own reported LastFM stats
    (139,371 users, 60,081 items, 4,017,311 interactions) come from this
    kind of filtering on the raw LastFM-360K data (~360K users otherwise).

    Pass either a single k (applies the same threshold to both sides), or
    separate min_user_interactions / min_item_interactions when the two
    sides need very different thresholds to hit a target size (e.g. LastFM's
    user and item activity distributions are shaped very differently, so a
    single shared k can't independently hit both target counts at once).
    """
    min_user = min_user_interactions if min_user_interactions is not None else k
    min_item = min_item_interactions if min_item_interactions is not None else k
    if min_user is None or min_item is None:
        raise ValueError("Provide either k, or both min_user_interactions and min_item_interactions.")

    df = ratings.copy()
    round_num = 0
    while True:
        round_num += 1
        before = len(df)
        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user].index
        valid_items = item_counts[item_counts >= min_item].index
        df = df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]
        after = len(df)
        print(f"(k_core_filter, min_user={min_user}, min_item={min_item}) round {round_num}: "
              f"{before} -> {after} interactions, "
              f"{df['user_id'].nunique()} users, {df['item_id'].nunique()} items")
        if after == before:
            break
        if after == 0:
            print(f"(k_core_filter) WARNING: filtering collapsed to 0 interactions -- "
                  f"min_user={min_user}/min_item={min_item} are too aggressive for this "
                  f"dataset's density (mutual dependency between the two thresholds caused "
                  f"a cascading collapse). Try lower values.")
            break
    return df.reset_index(drop=True)


def reindex_ids(ratings: pd.DataFrame):
    """
    Map raw user_id/item_id (1-indexed, possibly non-contiguous after filtering)
    to contiguous 0-indexed ids. Returns the remapped ratings DataFrame plus
    the two mapping dictionaries (needed to join back to user attributes later).
    """
    user_ids = sorted(ratings["user_id"].unique())
    item_ids = sorted(ratings["item_id"].unique())

    user_id_map = {raw_id: new_id for new_id, raw_id in enumerate(user_ids)}
    item_id_map = {raw_id: new_id for new_id, raw_id in enumerate(item_ids)}

    ratings = ratings.copy()
    ratings["user_id"] = ratings["user_id"].map(user_id_map)
    ratings["item_id"] = ratings["item_id"].map(item_id_map)

    return ratings, user_id_map, item_id_map


def train_test_split_leave_one_out(
    ratings: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leave-Last-Out (temporal) split: for each user, their single MOST RECENT
    interaction (by timestamp) goes to the test set, all earlier interactions
    go to train. This is the classic protocol used in NCF/BPR-style papers
    (He et al. 2017 and follow-ups) for top-K evaluation.

    If the 'timestamp' column is entirely missing (NaN) -- as is the case for
    the LastFM-360K play-count file, which has no per-interaction time --
    this automatically falls back to Leave-RANDOM-Out (one interaction per
    user chosen uniformly at random, seeded) and prints a warning, since a
    true "most recent" cannot be determined without timestamps.

    Users with only 1 interaction are dropped entirely (can't have both a
    train and a test interaction) -- this matches standard LOO practice and
    is reported so it's visible rather than silently changing user counts.
    """
    single_interaction_users = (
        ratings.groupby("user_id").size().loc[lambda s: s < 2].index
    )
    if len(single_interaction_users) > 0:
        print(
            f"(leave-one-out) dropping {len(single_interaction_users)} users "
            f"with only 1 interaction (can't form both train and test)"
        )
    ratings = ratings[~ratings["user_id"].isin(single_interaction_users)]

    has_timestamps = ratings["timestamp"].notna().any()
    if not has_timestamps:
        print(
            "(leave-one-out) WARNING: no per-interaction timestamps found in this "
            "dataset -- falling back to Leave-RANDOM-Out instead of the temporal "
            "Leave-Last-Out protocol used for datasets that do have timestamps "
            "(e.g. ML-1M). Results on this dataset are therefore not a true "
            "'most recent interaction' holdout."
        )
        rng = np.random.default_rng(seed)
        train_rows, test_rows = [], []
        for _, group in ratings.groupby("user_id"):
            idx = rng.permutation(len(group))
            group_shuffled = group.iloc[idx]
            test_rows.append(group_shuffled.iloc[[0]])
            train_rows.append(group_shuffled.iloc[1:])
        train_df = pd.concat(train_rows).reset_index(drop=True)
        test_df = pd.concat(test_rows).reset_index(drop=True)
        return train_df, test_df

    train_rows, test_rows = [], []
    for _, group in ratings.groupby("user_id"):
        # Sort by timestamp; ties broken deterministically by the given seed
        # so repeated runs are reproducible even when timestamps tie.
        group = group.sort_values("timestamp", kind="stable")
        test_rows.append(group.iloc[[-1]])   # most recent interaction
        train_rows.append(group.iloc[:-1])   # everything before it

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)
    return train_df, test_df


def train_test_split_per_user(
    ratings: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    [LEGACY -- kept for reference / comparison only, not used by default now]
    Random per-user split: for each user, test_size fraction of their
    interactions go to the test set, the rest to train. Guarantees every
    user appears in both splits (needed for per-user BPR sampling).
    """
    rng = np.random.default_rng(seed)
    train_rows, test_rows = [], []

    for _, group in ratings.groupby("user_id"):
        group = group.sample(frac=1.0, random_state=seed)  # shuffle
        n_test = max(1, int(len(group) * test_size))
        test_rows.append(group.iloc[:n_test])
        train_rows.append(group.iloc[n_test:])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)
    return train_df, test_df


def load_and_prepare(
    raw_dir: str = RAW_DIR, split_method: str = "leave_one_out",
    test_size: float = 0.2, positive_threshold: float = None,
    k_core: int = None, min_user_interactions: int = None, min_item_interactions: int = None,
    subsample_users: int = None, seed: int = 42,
) -> dict:
    """
    Full pipeline: load ratings + user attributes, reindex ids, split train/test.
    Returns a dict with everything downstream modules need.

    k_core / min_user_interactions / min_item_interactions: if given, applies
    iterative core filtering (see k_core_filter) BEFORE positive_threshold
    and attribute alignment. Pass k_core for a single shared threshold, or
    min_user_interactions/min_item_interactions separately when the two
    sides need different thresholds to hit a target dataset size. None
    (default, all three) skips this step entirely.

    subsample_users: if given, randomly samples this many users (with all
    their interactions kept) AFTER k-core filtering and attribute alignment,
    but BEFORE reindexing. Purely a compute-budget control for very large
    datasets (e.g. LastFM-360K's ~289K-user filtered core is too slow for
    FairIR's per-user contrastive step to train in practical time) -- not
    part of matching any paper's own methodology. None (default) skips it.

    split_method: "leave_one_out" (default -- classic NCF/BPR protocol, each
    user's most recent interaction held out as test) or "random" (legacy
    80/20-style per-user split, kept for comparison only). test_size is only
    used when split_method="random".

    positive_threshold: if given, only interactions with rating > this value
    are kept as "positive feedback" BEFORE reindexing/splitting -- everything
    else is dropped entirely (not used as train, not used as test). This
    matches FairIR's own preprocessing exactly for MovieLens-1M (Section 4.1:
    "considering items with user ratings greater than 3 as positive feedback"
    -> pass positive_threshold=3 to reproduce their 513,112-interaction,
    2.150%-density setup). Leave as None (default) to keep every interaction
    as positive, which is the right choice for datasets whose 'rating' column
    isn't a 1-5 preference scale to begin with (e.g. LastFM's play counts).
    """
    ratings = load_ratings(raw_dir)
    user_attrs = load_user_attributes(raw_dir)

    if k_core is not None or min_user_interactions is not None or min_item_interactions is not None:
        ratings = k_core_filter(
            ratings, k=k_core,
            min_user_interactions=min_user_interactions, min_item_interactions=min_item_interactions,
        )
        if len(ratings) == 0:
            raise ValueError(
                "k-core filtering removed all interactions -- the given thresholds are too "
                "aggressive for this dataset's density. Try lower min_user_interactions / "
                "min_item_interactions values."
            )

    if positive_threshold is not None:
        before = len(ratings)
        ratings = ratings[ratings["rating"] > positive_threshold].reset_index(drop=True)
        print(f"(positive_threshold={positive_threshold}) kept {len(ratings)} of {before} "
              f"interactions as positive feedback")

    # Restrict ratings to users who have valid demographic attributes BEFORE
    # reindexing. Some datasets (LastFM-360K) have users with missing/invalid
    # gender that load_user_attributes already drops -- if we reindexed on
    # ALL users in ratings and only filtered user_attrs afterward, the two
    # tables would no longer align by position (row i of user_attrs would
    # NOT correspond to user_id i in the model), silently corrupting every
    # downstream group-based computation (DP/EO, the instance gate, etc.).
    # Filtering here first guarantees n_users == len(user_attrs) and perfect
    # positional alignment after reindexing.
    valid_user_ids = set(user_attrs["user_id"])
    before_users = ratings["user_id"].nunique()
    ratings = ratings[ratings["user_id"].isin(valid_user_ids)].reset_index(drop=True)
    after_users = ratings["user_id"].nunique()
    if after_users < before_users:
        print(f"(attribute alignment) restricting to {after_users} of {before_users} users "
              f"with valid demographic attributes (dropping users with missing gender/age)")

    if subsample_users is not None:
        remaining_users = np.array(sorted(valid_user_ids & set(ratings["user_id"])))
        if len(remaining_users) > subsample_users:
            rng_sub = np.random.default_rng(seed)
            sampled_users = rng_sub.choice(remaining_users, size=subsample_users, replace=False)
            before_sub = len(ratings)
            ratings = ratings[ratings["user_id"].isin(sampled_users)].reset_index(drop=True)
            print(f"(subsample_users) randomly sampled {subsample_users} of {len(remaining_users)} "
                  f"users -- {before_sub} -> {len(ratings)} interactions "
                  f"(a compute-budget choice, not part of any published methodology being matched)")

    ratings, user_id_map, item_id_map = reindex_ids(ratings)

    # Remap user_attrs to the same contiguous ids; drop users that never rated anything.
    user_attrs = user_attrs[user_attrs["user_id"].isin(user_id_map)].copy()
    user_attrs["user_id"] = user_attrs["user_id"].map(user_id_map)
    user_attrs = user_attrs.sort_values("user_id").reset_index(drop=True)

    if split_method == "leave_one_out":
        train_df, test_df = train_test_split_leave_one_out(ratings, seed=seed)
    elif split_method == "random":
        train_df, test_df = train_test_split_per_user(ratings, test_size=test_size, seed=seed)
    else:
        raise ValueError(f"Unknown split_method: {split_method!r}. Use 'leave_one_out' or 'random'.")

    n_users = len(user_id_map)
    n_items = len(item_id_map)

    return {
        "train": train_df,
        "test": test_df,
        "user_attrs": user_attrs,
        "n_users": n_users,
        "n_items": n_items,
    }


if __name__ == "__main__":
    # Quick sanity check when run directly: python preprocessing/load_data.py
    data = load_and_prepare()
    print(f"n_users={data['n_users']}, n_items={data['n_items']}")
    print(f"train interactions={len(data['train'])}, test interactions={len(data['test'])}")
    print(data["user_attrs"].head())