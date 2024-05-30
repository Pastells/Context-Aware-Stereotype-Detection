import os
import shutil
from argparse import Namespace

SEED = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Figures
DPI = 600
PALETTE = "colorblind"
SUBTITLE_FONTSIZE = 18
TITLE_FONTSIZE = 20
LEGEND_FONTSIZE = 16
FIG_PARAMS = {
    "text.usetex": shutil.which("latex") is not None,  # check if latex is installed
    "font.size": 10,
    "font.family": "serif",
    "font.serif": "Computer Modern",
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "figure.figsize": (5, 4),
    "lines.linewidth": 2,
}

# ---------------------------------------
# DICTS for FIGURES
# ---------------------------------------

data_dict = {"detests": "DETESTS", "stereohoax": "StereoHoax-ES"}
models_dict = {
    "beto": "BETO",
    "roberta_bne": "MarIA",
    "bertin": "BERTIN",
    "mbert": "M-BERT",
}
contexts_dict = {
    "0_no_context": "No Context",
    "1_previous_sentences": "Level 1",  # "Previous Sentences",
    "2_previous_comment": "Level 2",  # "Parent Comment",
    "3_first_comment": "Level 3",  # "Root Comment",
    "4_news_title": "News Title",
    "5_concat_contexts": "Concatenated Contexts",
    "2_txt_father": "Level 2",  # "Parent Tweet",
    "3_txt_head": "Level 3",  # "Root Tweet",
    "4_rh_text": "Racial Hoax",
    "1 %": "(%)",
    "2 %": "(%)",
    "3 %": "(%)",
    "4 %": "(%)",
    "5 %": "(%)",
    "stereo_father": "Father Stereotype",
    "stereo_head": "Head Stereotype",
    "model": "Model",
}
metrics_dict = {
    "f1_neg": "$F_1$\nNegative Class",
    "f1_pos": "$F_1$\nPositive Class",
    "accuracy_explicit": "Accuracy\nExplicit",
    "accuracy_implicit": "Accuracy\nImplicit",
    "accuracy_no_contextual": "Accuracy\nNonContextual",
    "accuracy_contextual": "Accuracy\nContextual",
    "roc_auc": "Roc AUC",
    "accuracy": "Accuracy",
    "precision_neg": "Precision\nNegative Class",
    "recall_neg": "Recall\nNegative Class",
    "precision_pos": "Precision\nPositive Class",
    "recall_pos": "Recall\nPositive Class",
    "softmax_confidence": "Confidence",
    "TP_softmax_confidence": "TP Confidence",
    "FP_softmax_confidence": "FP Confidence",
    "FN_softmax_confidence": "FN Confidence",
    "TN_softmax_confidence": "TN Confidence",
}

# available corpora, baselines
data_choices = ["detests", "stereohoax"]
model_choices = ["all", "zeros", "ones", "random", "tfidf", "fast"]


# ---------------------------------------
# CORPORA
# ---------------------------------------

# Common labels
common = {
    "topics": [
        "xenophobia",
        "suffering",
        "economic",
        "migration",
        "culture",
        "benefits",
        "health",
        "security",
        "dehumanisation",
        "others",
    ],
    "implicit_feature": "implicit",
    "implicit_categories": [
        "Context",
        "Entailment/Evaluation",
        "Extrapolation",
        "Figures of speech",
        "Humor/Jokes",
        "Imperative/Exhortative",
        "Irony/Sarcasm",
        "World knowledge",
        "Others",
    ],
}

# StereoHoax corpus
stereohoax = {
    **common,
    "x_columns": [
        "text",
        "txt_father",
        "txt_head",
        "rh_text",
        "rh_type",
        "rh_id",
        "tweet_id",
        "conversation_id",
        "in_reply_to_tweet_id",
        "b",
        "du",
        "dd",
        "c",
        "ac",
        "p",
        "contextual",
        "implicit",
        "url",
    ],
    "indexes": ["index"],
    "text_columns": ["text", "txt_father", "txt_head", "rh_text"],
    "contexts": ["txt_father", "txt_head", "rh_text"],
    "contexts_results": [
        "0_no_context",
        "2_txt_father",
        "3_txt_head",
        "4_rh_text",
        "5_concat_contexts",
    ],
    "feature": "text",
    "target": "stereo",
    "target_pred": "stereo_pred",
    "data": "stereohoax",
    "path": "data/stereohoax",
    "datafile": "stereoHoax-ES_goldstandard.csv",
}
stereohoax["y_columns"] = [stereohoax["target"]] + stereohoax["topics"]
stereohoax["path"] = os.path.join(BASE_DIR, stereohoax["path"])


# DETESTS corpus
detests = {
    **common,
    "x_columns": [
        "comment_id",
        "sentence_pos",
        "reply_to",
        "sentence",
        "racial_target",
        "other_target",
        "implicit",
    ],
    "indexes": ["comment_id", "sentence_pos"],
    "text_columns": ["sentence"],
    "contexts": [
        "previous_sentences",
        "previous_comment",
        "first_comment",
        "news_title",
    ],
    "contexts_results": [
        "0_no_context",
        "1_previous_sentences",
        "2_previous_comment",
        "3_first_comment",
        "4_news_title",
        "5_concat_contexts",
    ],
    "feature": "sentence",
    "target": "stereotype",
    "target_pred": "stereotype_pred",
    "data": "detests",
    "path": "data/detests",
    "datafile": "train_original.csv",
}
detests["y_columns"] = [detests["target"]] + detests["topics"]
detests["path"] = os.path.join(BASE_DIR, detests["path"])


def get_conf(data: str) -> Namespace:
    """Convert the config dictionary for the wanted corpus into a Namespace"""
    if data.lower() == "detests":
        conf = Namespace(**detests)
    elif data.lower() == "stereohoax":
        conf = Namespace(**stereohoax)
    else:
        raise ValueError("Data must be 'detests' or 'stereohoax'")

    return conf
