"""
Add context and soft-labels to given datasets
"""

import os

import numpy as np
import pandas as pd
from scipy.special import softmax

from utils import config


def add_soft(data, labels, annotator_suffix="_a"):
    _data = data.copy()
    if isinstance(labels, str):
        labels = (labels,)
    for label in labels:
        cols = [c for c in data.columns.to_list() if label + annotator_suffix in c]
        if not cols:
            continue
        df = _data[cols].copy()

        df["positive"] = df.sum(axis=1)
        df["negative"] = 3 - df["positive"]
        pos_neg = df[["positive", "negative"]].to_numpy()
        _data[label + "_soft"] = np.round(softmax(pos_neg, axis=1)[:, 0], 4)
    return _data


def fill_context(df: pd.DataFrame, text, contexts) -> pd.DataFrame:
    """
    - fill missing contexts with level above
    - remove contexts equal to text itself
    - concatenate all contexts into "concat_contexts"

    text: text column
    contexts: list of contexts"""
    _df = df.copy()
    _df["concat_contexts"] = _df[contexts[-1]]

    for i, cont in enumerate(contexts[:-1][::-1]):
        # remove context equal to text
        _df.loc[_df[text] == _df[cont], cont] = ""

        no_cont = _df[cont].isin(("", "0", "[]"))
        _df.loc[no_cont, cont] = ""
        print(
            f"missing {cont}:",
            len(_df[no_cont]),
            f"({len(_df[no_cont]) / len(_df) * 100:.0f}%)",
        )

        # concat context if it's not repated (same as context above)
        same_as_above = _df[cont] == _df[contexts[len(contexts) - 1 - i]]
        _df.loc[~same_as_above, "concat_contexts"] = (
            _df.loc[~same_as_above, cont] + " " + _df.loc[~same_as_above, "concat_contexts"]
        ).str.strip()

        # fill missing context
        _df.loc[no_cont, cont] = _df.loc[no_cont, contexts[len(contexts) - 1 - i]]
    return _df


def detests_news(df: pd.DataFrame, tokenized=False) -> pd.DataFrame:
    """Add news title
    If tokenized is True:  use tokenized_news_title column
    """
    news = pd.read_csv("data/detests/news_DETESTS.csv", sep=";")
    column = "tokenized_news_title" if tokenized else "news_title"
    news = news[["file_id", column]].rename(columns={column: "news_title"})

    df = df.merge(news, on="file_id", how="left")
    return df


def find_first_comment(row, df, comment_id="comment_id", reply="reply_to", no_reply=None):
    """Returns the `comment_id` for the first comment"""
    while row[comment_id] != row[reply]:
        filtered_df = df[df[comment_id] == row[reply]]
        if row[reply] == no_reply:
            break
        if filtered_df.empty:
            return "UNAVAILABLE"
        row = filtered_df.iloc[0]
    return row[comment_id]


def detests_context(df: pd.DataFrame) -> pd.DataFrame:
    """Create context columns for DETESTS:
    1. previous_sentences
    2. previous_comment
    3. first_comment
    4. news_title

    If tokenized is True: use tokenized_news_title column
    """
    det = df.copy()

    def concatenate_previous_sentences(group, separator=" "):
        # without `cumsum` it would return just the previous sentence
        group["previous_sentences"] = group["sentence"].shift().fillna("") + separator
        group["previous_sentences"] = group["previous_sentences"].cumsum()
        return group

    det["previous_sentences"] = ""
    det[["sentence", "previous_sentences"]] = det.groupby("comment_id", group_keys=False)[
        ["sentence", "previous_sentences"]
    ].apply(concatenate_previous_sentences)

    det["previous_sentences"] = det["previous_sentences"].str.strip()
    det["first_comment_id"] = det.apply(lambda row: find_first_comment(row, det), axis=1)

    # Secondary DataFrame with full comments
    comments = (
        det[["comment_id", "reply_to", "sentence", "previous_sentences"]]
        .groupby("comment_id")
        .tail(1)
        .set_index("comment_id")
    )

    comments["comment"] = (comments["previous_sentences"] + " " + comments["sentence"]).str.strip()

    # Add previous comment and first comment as contexts
    det = pd.merge(
        det, comments["comment"], how="left", left_on=["reply_to"], right_index=True
    ).rename(columns={"comment": "previous_comment"})
    det = pd.merge(
        det,
        comments["comment"],
        how="left",
        left_on=["first_comment_id"],
        right_index=True,
    ).rename(columns={"comment": "first_comment"})

    # Remove previous_comment and first_comment when is itself
    det.loc[det.reply_to == det.comment_id, "previous_comment"] = ""
    det.loc[det.first_comment_id == det.comment_id, "first_comment"] = ""
    det = det.fillna("")
    return det


def clean_stereocom2(df):
    df = df.rename(
        columns={
            "FILE_ID": "file_id",
            "USER_ID": "user_id",
            "COMMENT_ID": "sentence_id",
            "THREAD": "reply_to",
            "COMMENT": "sentence",
        }
    )
    df["file_id"] = df["file_id"].str.strip()

    df["comment_id"] = df["file_id"] + df["sentence_id"].str.split("_").str[0]
    df["sentence_pos"] = df["sentence_id"].str.split("_").str[1]
    df["reply_to"] = df["file_id"] + df["reply_to"].str[:-1]

    # Fix non-breaking spaces "\xa0"
    df["sentence"] = df["sentence"].str.split().str.join(" ")
    df = df[
        [
            "file_id",
            "sentence_id",
            "comment_id",
            "sentence_pos",
            "reply_to",
            "user_id",
            "sentence",
        ]
    ]
    return df


# -----------------------------------------------------
# Main programs
# -----------------------------------------------------


def main_detests():
    print("DATA: detests")
    conf = config.get_conf("detests")
    tokenized = [True, True, False, False]
    for i, file in enumerate(
        (
            "train.csv",
            "test.csv",
            "train_with_disagreement.csv",
            "test_with_disagreement.csv",
        )
    ):
        print("FILE - ", file)
        file = os.path.join(conf.path, file)
        sep = "\t" if file.endswith(".tsv") else ","
        df = pd.read_csv(file, sep=sep)
        df = (
            df.pipe(detests_context)
            .pipe(detests_news, tokenized=tokenized[i])
            .pipe(add_soft, (conf.target, "implicit"))
        )
        df["sentence"] = df["sentence"].str.strip()
        # df.to_csv(file[:-4] + "_context_no_fill_soft.csv", index=False)
        df = fill_context(df, "sentence", conf.contexts)
        df.to_csv(file[:-4] + "_context_soft.csv", index=False)


def main_stereohoax():
    print("DATA: stereohoax")
    conf = config.get_conf("stereohoax")
    for file in (
        "train_val_split.csv",
        "train_split.csv",
        "val_split.csv",
        "test_split.csv",
    ):
        print("FILE - ", file)
        file = os.path.join(conf.path, file)
        df = pd.read_csv(file)
        df = add_soft(df, (conf.target,))
        # df.to_csv(file[:-4] + "_context_no_fill_soft.csv", index=False)
        df = fill_context(df, "text", conf.contexts)
        df.to_csv(file[:-4] + "_context_soft.csv", index=False)


if __name__ == "__main__":
    main_detests()
    main_stereohoax()
