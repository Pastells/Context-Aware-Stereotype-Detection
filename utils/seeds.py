"""
Analysis of results with multiple seeds
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind

from utils import config
from utils.plots import barplot, savefig

st_cols = (
    [
        "text",
        "stereo",
        "stereo_soft",
        "implicit",
        "implicit_soft",
        "contextual",
        "url",
        "txt_father",
        "stereo_father",
        "implicit_father",
        "contextual_father",
        "txt_head",
        "stereo_head",
        "implicit_head",
        "contextual_head",
        "rh_text",
        "rh_id",
        "stereo_pred",
        # "softmax_confidence",
        # "softmax_confidence_std",
    ]
    + config.common["topics"]
    + config.common["implicit_categories"]
)
st_cols_res = [
    # "data",
    # "stereo",
    "model",
    "index",
    "2_txt_father",
    "2 %",
    "2 implicit",
    "2 contextual",
    "stereo_father",
    "3_txt_head",
    "3 %",
    "3 implicit",
    "3 contextual",
    "stereo_head",
    "4_rh_text",
    "4 %",
    "4 implicit",
    "4 contextual",
    "total",
]
st_cols_tex = [
    "index",
    "2_txt_father",
    "2 %",
    "3_txt_head",
    "3 %",
    "4_rh_text",
    "4 %",
    "total",
]
det_cols = (
    [
        "sentence",
        "previous_sentences",
        "stereotype_previous_sentences",
        "implicit_previous_sentences",
        "previous_comment",
        "stereotype_previous_comment",
        "implicit_previous_comment",
        "first_comment",
        "stereotype_first_comment",
        "implicit_first_comment",
        "news_title",
        "file_id",
        "stereotype",
        "stereotype_soft",
        "implicit",
        "implicit_soft",
        # "stereo_pred",
        # "softmax_confidence",
        # "softmax_confidence_std",
    ]
    + config.common["topics"]
    # + config.common["implicit_categories"]
)
det_cols_res = [
    "index",
    "data",
    "model",
    "stereo",
    "1_previous_sentences",
    "1 %",
    "1 implicit",
    "2_previous_comment",
    "2 %",
    "2 implicit",
    "3_first_comment",
    "3 %",
    "3 implicit",
    "4_news_title",
    "4 %",
    "4 implicit",
    "total",
]
det_cols_tex = [
    "index",
    "1_previous_sentences",
    "1 %",
    "2_previous_comment",
    "2 %",
    "3_first_comment",
    "3 %",
    "4_news_title",
    "4 %",
    "total",
]


def group_metrics_seeds(
    data: str,
    model: str,
    context: str,
    hard_or_soft: str,
    suffixes: tuple = ("max_tokens_512",),
    n_seeds: int = -1,
) -> pd.DataFrame:
    """
    reads the metrics file generated with create_metrics and aggregates the different seeds
    """
    df = pd.read_csv(
        os.path.join(config.BASE_DIR, "results/metrics", f"metrics_{data}-task1.csv"),
    )
    preffix = f"{data}-{model}-{context}_{hard_or_soft}"

    masks = []
    for suf in suffixes:
        masks.append(df.model.str.contains(suf))

    masks = np.logical_or.reduce(masks)

    if n_seeds != -1:
        upper_limit = 42 + n_seeds - 1
        if n_seeds < 8:
            pattern = rf"s(?:4[2-{upper_limit % 10}]|"
        else:
            pattern = r"s(?:4[2-9]|"

        for i in range(5, upper_limit // 10):
            pattern += f"{i}[0-9]|"
        pattern += rf"{upper_limit // 10}[0-{upper_limit % 10}])"
        masks = masks & (df.model.str.contains(pattern))

    df = df[masks & df.model.str.contains(preffix)]
    return df


def avg_metrics_seeds(
    data: str,
    model: str,
    context: str,
    hard_or_soft: str,
    suffixes: tuple = ("max_tokens_512",),
    n_seeds: int = -1,
    df_ttest=None,
) -> pd.DataFrame:
    """
    reads the metrics file generated with create_metrics and aggregates the different seeds
    returns the median for each metric, as well as the std, and quartiles 1 and 3

    if df_ttest is given, a _ttest column is added for each metric
    """
    df = group_metrics_seeds(data, model, context, hard_or_soft, suffixes, n_seeds)

    if df.empty:
        print(data, model, context, hard_or_soft, suffixes, n_seeds, "empty")
        return "empty"

    cols = df.columns[1:]
    if df_ttest is not None:
        ttest = {}
        for metric in cols:
            test = ttest_ind(df[metric], df_ttest[metric], equal_var=False)
            ttest[metric + "--ttest"] = test.statistic
            ttest[metric + "--pvalue"] = test.pvalue
        ds_ttest = pd.Series(ttest)

    df = df[cols].agg(["median", "std", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
    new_dfs = []
    for metric in cols:
        col_std = pd.DataFrame({metric + "--std": df.iloc[1][metric]}, index=df.index)
        col_q1 = pd.DataFrame({metric + "--q1": df.iloc[2][metric]}, index=df.index)
        col_q3 = pd.DataFrame({metric + "--q3": df.iloc[3][metric]}, index=df.index)
        new_dfs.extend([col_std, col_q1, col_q3])

    df = pd.concat([df] + new_dfs, axis=1).iloc[0]
    if df_ttest is not None:
        df = pd.concat([df, ds_ttest])

    preffix = f"{data}-{model}-{context}_{hard_or_soft}"
    df["model"] = preffix + "_sMean_" + "max_tokens_512"

    return df


def get_avg_seeds_df(
    data,
    suffixes: tuple = ("max_tokens_512",),
    n_seeds: int = -1,
    contexts=None,
    hard_or_soft: tuple = ("hard", "soft"),
    models: tuple = ("beto", "roberta_bne", "mbert"),
) -> pd.DataFrame:
    conf = config.get_conf(data)
    dfs = []
    if contexts is None:
        contexts = conf.contexts_results
    for model in models:
        for h_s in hard_or_soft:
            df_no_cont = group_metrics_seeds(data, model, "0_no_context", h_s, suffixes, n_seeds)
            for context in contexts:
                _df = avg_metrics_seeds(
                    data, model, context, h_s, suffixes, n_seeds, df_ttest=df_no_cont
                )
                if isinstance(_df, str) and _df == "empty":
                    continue
                dfs.append(_df)
    df = pd.concat(dfs, axis=1).T
    return df


def barplot_from_metrics_seeds(
    data="stereohoax",
    models: tuple = ("beto", "roberta_bne", "mbert"),
    hard_or_soft: tuple = ("hard",),
    metrics=None,
    std: bool = False,
    quartiles: bool = False,
    ttest: bool = False,
    save: bool = False,
    suffixes: tuple = ("max_tokens_512",),
    contexts=None,
    n_seeds: int = -1,
    rotation=0,
    **kwargs,
):
    """
    if ttest is True, adds an * on top of the bars that have p-values < 0.05
    """
    conf = config.get_conf(data)
    hard_or_soft_labels = {x: x.capitalize() + " labels" for x in hard_or_soft}

    if contexts is None:
        contexts = conf.contexts_results[:-1]

    if metrics is None:
        metrics = [
            "f1_neg",
            "f1_pos",
            "f1_explicit",
            "f1_implicit",
            "roc_auc",
        ]

    columns = ["model"] + metrics
    if std:
        cols = columns + [c + "--std" for c in metrics]
        y_std = "std"
        y_minmax = ""
    elif quartiles:
        cols = columns + [c + "--q1" for c in metrics] + [c + "--q3" for c in metrics]
        y_std = ""
        y_minmax = ("q1", "q3")
    else:
        cols = columns
        y_std = ""
        y_minmax = ""

    if ttest:
        cols = cols + [c + "--ttest" for c in metrics]
        cols = cols + [c + "--pvalue" for c in metrics]

    def scale_ce(df, cols):
        if "cross_entropy" in cols:
            for col in (c for c in cols if "cross_entropy" in c):
                df[col] = df[col] / df[col].max()
        return df

    def metric_cols(df):
        df["metric_type"] = df["metric"].str.split("--").str[-1]
        df["metric_type"] = df["metric_type"].apply(
            lambda x: (
                "median" if x not in ["min", "max", "std", "q1", "q3", "ttest", "pvalue"] else x
            )
        )
        df["metric"] = df["metric"].apply(lambda x: x[: x.rfind("--")] if "--" in x else x)
        return df

    df = (
        get_avg_seeds_df(data, suffixes, n_seeds, contexts, hard_or_soft, models)[cols]
        .pipe(scale_ce, cols)
        .rename(columns={col: config.metrics_dict[col] for col in metrics})
        .rename(columns={col + "--std": config.metrics_dict[col] + "--std" for col in metrics})
        .rename(columns={col + "--q1": config.metrics_dict[col] + "--q1" for col in metrics})
        .rename(columns={col + "--q3": config.metrics_dict[col] + "--q3" for col in metrics})
        .rename(columns={col + "--ttest": config.metrics_dict[col] + "--ttest" for col in metrics})
        .rename(
            columns={col + "--pvalue": config.metrics_dict[col] + "--pvalue" for col in metrics}
        )
        .melt("model", var_name="metric")
        .pipe(metric_cols)
        .pivot_table(index=["model", "metric"], columns="metric_type", values="value")
        .reset_index()
    )

    fig, ax = plt.subplots(
        len(models),
        len(hard_or_soft),
        figsize=(15, 2 + 2 * len(models)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    palette = sns.color_palette(config.PALETTE)[:10]
    if data == "stereohoax":
        palette = palette[:1] + palette[2:]
    palette = palette[: len(contexts)]

    for i, model in enumerate(models):
        for j, h_s in enumerate(hard_or_soft):
            _df = df[
                df.model.str.contains(data)
                & df.model.str.contains(model)
                & df.model.str.contains(h_s)
                & df.model.str.contains("max_tokens_512")
            ]
            if _df.empty:
                continue
            barplot(
                fig,
                ax[i, j],
                _df,
                "metric",
                "median",
                "model",
                y_std=y_std,
                y_minmax=y_minmax,
                ttest=ttest,
                legend=False,
                palette=palette,
                order=[config.metrics_dict[col] for col in metrics],
                **kwargs,
            )
            ax[i, j].set_xlabel("")
            ax[i, j].set_ylabel("")
            # ax[i, j].set_ylim([0.3, 1.05])
            ax[i, j].set_title(f"{config.models_dict[model]}", fontsize=config.SUBTITLE_FONTSIZE)
            # ax[i, j].set_title(f"{hard_or_soft_labels[h_s]} {model}", fontsize=config.SUBTITLE_FONTSIZE)
            ax[i, j].set_yticks([0, 0.5, 1])
            if rotation != 0:
                ax[i, j].set_xticklabels(ax[i, j].get_xticklabels(), rotation=rotation, ha="right")
    # fig.suptitle(config.data_dict[data], fontsize=config.TITLE_FONTSIZE)
    fig.legend(
        loc="upper center",
        ncol=len(contexts),
        bbox_to_anchor=(0.5, 1.05),
        fontsize=config.LEGEND_FONTSIZE,
        labels=["_no_legend_"] * len(contexts) * len(metrics)
        + [config.contexts_dict[c] for c in contexts],
    )
    if save is not False:
        savefig(save)


def avg_results_seeds(
    data, model, context, hard_or_soft, folder=None, n_seeds: int = -1
) -> pd.DataFrame:
    """
    Average of the predictions at text level
    """
    folder = folder if folder is not None else "fine_tuning_512"
    dfs = []
    conf = config.get_conf(data)
    upper_limit = 92 if n_seeds == -1 else 42 + n_seeds
    for seed in range(42, upper_limit):
        f = f"{data}-{model}-{context}_{hard_or_soft}_s{seed}_max_tokens_512-task1.csv"
        try:
            _df = pd.read_csv(os.path.join(config.BASE_DIR, "results", data, folder, f))
            dfs.append(_df)
        except FileNotFoundError:
            print(f, "not found. No more seeds will be added")
            break

    target_pred = conf.target + "_pred"
    if hard_or_soft == "hard":
        cols1 = [target_pred, "softmax_confidence"] + conf.indexes
        cols2 = conf.indexes + [
            conf.target,
            f"{conf.target}_pred_mean",
            "softmax_confidence_mean",
            "softmax_confidence_std",
        ]
        cols_rename = {
            f"{conf.target}_pred_mean": target_pred,
            "softmax_confidence_mean": "softmax_confidence",
        }
        df = pd.concat(dfs)[cols1].groupby(conf.indexes).agg(["mean", "std"])
        df.columns = [f"{col[0]}_{col[1]}" for col in df.columns.to_flat_index()]
        df = df.merge(_df[conf.indexes + [conf.target]], how="left", on=conf.indexes)
        df = df[cols2].rename(columns=cols_rename)

    else:
        cols1 = [f"{conf.target}_pred_soft"] + conf.indexes
        df = pd.concat(dfs)[cols1].groupby(conf.indexes).agg(["mean", "std"])
        df.columns = [f"{col[0]}_{col[1]}" for col in df.columns.to_flat_index()]
        df[target_pred] = df[f"{conf.target}_pred_soft_mean"].round(0).astype(int)

        cols_rename = {f"{conf.target}_pred_soft_mean": f"{conf.target}_pred_soft"}
        df = df.merge(
            _df[conf.indexes + [conf.target, f"{conf.target}_soft"]],
            how="left",
            on=conf.indexes,
        )
        cols2 = conf.indexes + [
            conf.target,
            target_pred,
            f"{conf.target}_soft",
            f"{conf.target}_pred_soft_mean",
            f"{conf.target}_pred_soft_std",
        ]
        df = df[cols2].rename(columns=cols_rename)

    return df.set_index(conf.indexes)


def mejora_empeora(df, dfs, conf, columns, thres: float = 0.3, pos_neg: int = 1):
    """
    Changes between no context and context for FN, TP, FP, TN
    """
    filt = df[conf.target] == pos_neg
    pos = (df[conf.target] == 1) & filt
    neg = (df[conf.target] == 0) & filt

    no_cont_acierto_pos = pos & (df[conf.target_pred] >= 1 - thres)
    no_cont_fallo_pos = pos & (df[conf.target_pred] <= thres)
    no_cont_acierto_neg = neg & (df[conf.target_pred] <= thres)
    no_cont_fallo_neg = neg & (df[conf.target_pred] >= 1 - thres)

    for cont in conf.contexts_results[1:]:
        df[cont] = ""
        cont_fallo_pos = dfs[cont][conf.target_pred] <= thres
        cont_acierto_pos = dfs[cont][conf.target_pred] >= 1 - thres
        cont_fallo_neg = dfs[cont][conf.target_pred] >= 1 - thres
        cont_acierto_neg = dfs[cont][conf.target_pred] <= thres

        df.loc[no_cont_acierto_pos & cont_fallo_pos, cont] = "empeora (FN)"
        df.loc[no_cont_fallo_pos & cont_acierto_pos, cont] = "mejora (TP)"
        df.loc[no_cont_acierto_neg & cont_fallo_neg, cont] = "empeora (FP)"
        df.loc[no_cont_fallo_neg & cont_acierto_neg, cont] = "mejora (TN)"

    mask = df[conf.contexts_results[1:]] != ""
    df = df[mask.any(axis=1)][conf.contexts_results[1:] + columns]
    return df


def get_no_cont_counts(
    data,
    thres: float = 0.3,
    folder=None,
    n_seeds: int = -1,
    models: tuple = ("beto", "roberta_bne", "mbert"),
):
    no_cont_counts = {}
    conf = config.get_conf(data)
    for model in models:
        dfs = {
            cont: avg_results_seeds(data, model, cont, "hard", folder, n_seeds)
            for cont in conf.contexts_results
        }
        df = dfs["0_no_context"]
        pos = df[conf.target] == 1
        neg = df[conf.target] == 0

        no_cont_acierto_pos = pos & (df[conf.target_pred] >= 1 - thres)
        no_cont_fallo_pos = pos & (df[conf.target_pred] <= thres)
        no_cont_acierto_neg = neg & (df[conf.target_pred] <= thres)
        no_cont_fallo_neg = neg & (df[conf.target_pred] >= 1 - thres)

        TP = len(df[no_cont_acierto_pos])
        FP = len(df[no_cont_fallo_pos])
        FN = len(df[no_cont_fallo_neg])
        TN = len(df[no_cont_acierto_neg])
        no_cont_counts[model] = (
            FP,
            TN,
            FN,
            TP,
            FP / len(df[df[conf.target] == 0]) * 100,
            TN / len(df[df[conf.target] == 0]) * 100,
            FN / len(df[df[conf.target] == 1]) * 100,
            TP / len(df[df[conf.target] == 1]) * 100,
        )

    divide_by = []
    for model in models:
        divide_by.extend(no_cont_counts[model][:2])
    for model in models:
        divide_by.extend(no_cont_counts[model][2:4])
    divide_by = np.array(divide_by)
    return no_cont_counts, divide_by


def get_analysis_dfs(
    data,
    test,
    cols,
    thres: float = 0.3,
    folder=None,
    n_seeds: int = -1,
    models: tuple = ("beto", "roberta_bne", "mbert"),
):
    conf = config.get_conf(data)
    hard_or_soft = "hard"
    pos_neg = ("negativos", "positivos")
    analysis_dfs = []
    for j, stereo in enumerate(range(2)):
        for model in models:
            dfs = {
                cont: avg_results_seeds(data, model, cont, hard_or_soft, folder, n_seeds)
                for cont in conf.contexts_results
            }
            df = dfs["0_no_context"]
            columns = [c for c in df.columns if (c in test.columns) and (c not in conf.indexes)]

            if data == "stereohoax":
                df = df.merge(
                    test.drop(columns=columns),
                    how="left",
                    on=conf.indexes,
                    validate="1:1",
                )
                _df = mejora_empeora(df, dfs, conf, cols, thres, pos_neg=stereo)
                _df.loc[_df["txt_father"] == _df["txt_head"], "txt_head"] = "igual al father"
                _df.loc[_df["txt_father"] == _df["rh_text"], "txt_father"] = "igual al rh_text"
                _df.loc[_df["txt_head"] == _df["rh_text"], "txt_head"] = "igual al rh_text"
            else:
                df = df.join(
                    test.set_index(conf.indexes).drop(columns=columns),
                    validate="1:1",
                )
                _df = mejora_empeora(df, dfs, conf, cols, thres, pos_neg=stereo)
                _df.loc[_df["previous_comment"] == _df["first_comment"], "first_comment"] = (
                    "igual al father"
                )
                _df.loc[_df["previous_comment"] == _df["news_title"], "previous_comment"] = (
                    "igual al news_title"
                )
                _df.loc[_df["first_comment"] == _df["news_title"], "first_comment"] = (
                    "igual al news_title"
                )

            _df.to_excel(
                f"../results/analysis/{data}_{model}_{hard_or_soft}_context_analysis_{pos_neg[j]}.xlsx"
            )
            # _df.to_csv(
            #     f"../results/analysis/{data}_{model}_{hard_or_soft}_context_analysis_{pos_neg[j]}.csv"
            # )
            columns = _df.columns.tolist()
            _df["model"] = model
            analysis_dfs.append(_df[["model"] + columns].fillna(0))

    return analysis_dfs


def tabla_resumen(analysis_dfs, data, models=("beto", "roberta_bne", "mbert")):
    conf = config.get_conf(data)
    dfs = []
    models = models * 2
    pos_neg = [0] * len(models) + [1] * len(models)
    mej_emp = [["empeora (FP)", "mejora (TN)"], ["empeora (FN)", "mejora (TP)"]]
    for i, _df in enumerate(analysis_dfs):
        _df = _df[conf.contexts_results[1:]].apply(pd.Series.value_counts)
        _df["data"] = data
        _df["model"] = models[i]
        _df["stereo"] = pos_neg[i]

        # Missing empeora but not mejora -> need to add it and flip order
        if (
            not _df.empty
            and not _df.index.str.contains("empeora").any()
            and _df.index.str.contains("mejora").any()
        ):
            new_row = pd.DataFrame(
                {col: [0] for col in conf.contexts_results[1:]}, index=[mej_emp[pos_neg[i]][0]]
            )
            new_row["data"] = data
            new_row["model"] = models[i]
            new_row["stereo"] = pos_neg[i]
            _df = pd.concat([_df.iloc[:-1], new_row, _df.iloc[-1:]])
        if _df.empty or not _df.index.str.contains("empeora").any():
            _df.loc[mej_emp[pos_neg[i]][0]] = [0] * len(conf.contexts_results[1:]) + [
                data,
                models[i],
                pos_neg[i],
            ]
        if not _df.index.str.contains("mejora").any():
            _df.loc[mej_emp[pos_neg[i]][1]] = [0] * len(conf.contexts_results[1:]) + [
                data,
                models[i],
                pos_neg[i],
            ]

        dfs.append(_df)

    cols = ["data", "model", "stereo"] + conf.contexts_results[1:-1]
    df = (
        pd.concat(dfs)[cols]
        .reset_index()
        .query("index != ''")
        .fillna(0)
        .convert_dtypes()
        .reset_index(drop=True)
    )
    df["arrows"] = np.where(df["index"].str.contains("mejora"), "↑", "↓")
    return df


def get_st_test():
    st_test = pd.read_csv(
        os.path.join(config.BASE_DIR, "data/stereohoax/test_split_context_soft.csv"),
        index_col=0,
    )
    st_unaggregated = pd.read_csv(
        os.path.join(config.BASE_DIR, "data/stereohoax", "stereohoax_unaggregated.csv"),
        index_col=0,
    )
    st_test_crit = pd.read_csv(
        os.path.join(
            config.BASE_DIR,
            "data/stereohoax",
            "stereohoax_implicit_criteria_5anotadors.csv",
        ),
        index_col=0,
    )

    st_test = (
        st_test.join(st_unaggregated[["implicit_soft", "contextual_soft", "url_soft"]])
        .merge(
            st_test_crit[config.common["implicit_categories"]],
            how="left",
            left_index=True,
            right_index=True,
        )
        .fillna(0)
    )
    conf = config.get_conf("stereohoax")
    st_test = (
        st_test.reset_index()
        .merge(
            st_test[["tweet_id", "stereo", "implicit", "contextual"]],
            how="left",
            left_on="in_reply_to_tweet_id",
            right_on="tweet_id",
            suffixes=("", "_father"),
        )
        .merge(
            st_test[["tweet_id", "stereo", "implicit", "contextual"]],
            how="left",
            left_on="conversation_id",
            right_on="tweet_id",
            suffixes=("", "_head"),
        )
        .set_index(conf.indexes)
    )
    return st_test


def cut_previous_sentences_labels(row):
    row["stereotype_previous_sentences"] = row["stereotype_previous_sentences"][
        : row["sentence_pos"] - 1
    ]
    row["implicit_previous_sentences"] = row["implicit_previous_sentences"][
        : row["sentence_pos"] - 1
    ]
    return row


def get_det_test():
    det_test = pd.read_csv(
        os.path.join(config.BASE_DIR, "data/detests/test_with_disagreement_context_soft.csv")
    )

    comment_id_stereotype = [
        det_test[det_test["comment_id"] == i]["stereotype"].to_numpy()
        for i in sorted(set(det_test.comment_id))
    ]
    comment_id_implicit = [
        det_test[det_test["comment_id"] == i]["implicit"].to_numpy()
        for i in sorted(set(det_test.comment_id))
    ]
    comment_id = pd.DataFrame(
        {"stereotype": comment_id_stereotype, "implicit": comment_id_implicit}
    )

    previous_sentences_stereotype = [
        det_test[det_test["comment_id"] == i]["stereotype"].to_numpy() for i in det_test.comment_id
    ]
    previous_sentences_implicit = [
        det_test[det_test["comment_id"] == i]["implicit"].to_numpy() for i in det_test.comment_id
    ]
    previous_sentences = pd.DataFrame(
        {
            "stereotype_previous_sentences": previous_sentences_stereotype,
            "implicit_previous_sentences": previous_sentences_implicit,
        }
    )

    det_test = (
        (
            det_test.merge(previous_sentences, how="left", left_index=True, right_index=True).apply(
                cut_previous_sentences_labels, axis=1
            )
        )
        .merge(
            comment_id,
            how="left",
            left_on="reply_to",
            right_index=True,
            suffixes=("", "_previous_comment"),
        )
        .merge(
            comment_id,
            how="left",
            left_on="first_comment_id",
            right_index=True,
            suffixes=("", "_first_comment"),
        )
    )
    return det_test


def add_col_resumen(df, i, resumen, cont, col):
    _df = df.groupby(cont)[col].sum()
    if not _df[_df.index.str.contains("empeora")].empty:
        resumen.loc[2 * i, col] = _df[_df.index.str.contains("empeora")].to_numpy()
    if not _df[_df.index.str.contains("mejora")].empty:
        resumen.loc[2 * i + 1, col] = _df[_df.index.str.contains("mejora")].to_numpy()


def paper_names(resumen):
    df = resumen.copy()
    df["model"] = df["model"].map(config.models_dict)
    df["index"] = df["index"].str[-3:-1]
    return df


def add_perc(resumen, divide_by, conf):
    """
    Add percentage of change respect to the number of cases with no context
    """
    for context in conf.contexts_results[1:-1]:
        resumen[context[0] + " %"] = np.round(resumen[context] / divide_by * 100, 0)
        resumen[context[0] + " %"] = resumen[context[0] + " %"].astype(int)
        resumen[context[0] + " %"] = np.where(
            resumen[context] > 0,
            (("(" + resumen[context[0] + " %"].astype(str)) + "% " + resumen["arrows"] + ")"),
            "",
        )
    resumen["total"] = divide_by
    return resumen


def casos_comun_beto_roberta(data, analysis_dfs, models):
    conf = config.get_conf(data)
    neg_pos = ("negative class", "positive class")
    for cont in conf.contexts_results[1:]:
        for i, j in enumerate((0, len(models))):
            df = analysis_dfs[j]
            a = set(df[df[cont] != ""].index)
            df = analysis_dfs[j + 1]
            b = set(df[df[cont] != ""].index)
            c = a.intersection(b)
            print(f"{cont}, {neg_pos[i]}:", c)
