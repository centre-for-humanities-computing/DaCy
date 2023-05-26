from functools import partial
from typing import Any, Dict

import altair as alt
import pandas as pd
from spacy.language import Language
from spacy.training import Example

import dacy

from .generalization_utils import bootstrap, compute_mean_and_ci, dansk
from .ner_bias_utils import DACY_MODELS_FINE
from .ner_bias_utils import MDL_GETTER_DICT as _MDL_GETTER_DICT
from .ner_bias_utils import highlight_max, underline_second_max

MDL_GETTER_DICT = {
    mdl_name: mdl
    for mdl_name, mdl in _MDL_GETTER_DICT.items()
    if mdl_name not in DACY_MODELS_FINE
}

MDL_FINE_GETTER_DICT = {
    "da_dacy_large_ner_fine_grained-0.1.0": partial(
        dacy.load,
        "da_dacy_large_ner_fine_grained-0.1.0",
    ),
    "da_dacy_medium_ner_fine_grained-0.1.0": partial(
        dacy.load,
        "da_dacy_medium_ner_fine_grained-0.1.0",
    ),
    "da_dacy_small_ner_fine_grained-0.1.0": partial(
        dacy.load,
        "da_dacy_small_ner_fine_grained-0.1.0",
    ),
}


def score_to_string(score: Dict[str, Any], decimals: int = 1) -> str:
    if score["mean"] == 0:
        return " "
    return f"{100*score['mean']:.{decimals}f} ({100*score['ci'][0]:.{decimals}f}, {100*score['ci'][1]:.{decimals}f})"


def apply_models(
    mdl_name: str, nlp: Language, examples: list[Example], decimals=1
) -> pd.DataFrame:
    texts = [example.reference.text for example in examples]
    docs = nlp.pipe(texts)
    for doc, example in zip(docs, examples):
        example.predicted = doc
    score = bootstrap(examples, getter=None)
    score = compute_mean_and_ci(score)

    row = {
        "Models": mdl_name,
    }
    for key, value in score.items():
        row[key] = score_to_string(value, decimals=decimals)
    return pd.DataFrame([row])


def create_table(
    df: pd.DataFrame,
    caption="F1 score with 95% confidence interval calculated using bootstrapping with 100 samples.",
):
    # replace index with range
    df.index = range(len(df))  # type: ignore

    col_names = [("", "Models")] + [("F1", col) for col in df.columns[1:]]
    super_header = pd.MultiIndex.from_tuples(col_names)
    df.columns = super_header

    s = df.style.apply(highlight_max, axis=0, subset=df.columns[1:])
    s = s.apply(underline_second_max, axis=0, subset=df.columns[1:])

    # Add a caption
    s = s.set_caption(caption)

    # Center the header and left align the model names
    s = s.set_properties(subset=df.columns[1:], **{"text-align": "right"})

    super_header_style = [
        {"selector": ".level0", "props": [("text-align", "center")]},
        {"selector": ".col_heading", "props": [("text-align", "center")]},
    ]
    # Apply the CSS style to the styler
    s = s.set_table_styles(super_header_style)
    s = s.set_properties(subset=[("", "Models")], **{"text-align": "left"})
    # remove the index
    s = s.hide(axis="index")
    return s


def create_dansk_viz(df: pd.DataFrame):
    plot_df = df.melt(
        id_vars=["Models"],
        var_name="Label",
        value_name="F1 string",
    )

    # Convert the score value to a float
    plot_df["F1"] = plot_df["F1 string"].apply(
        lambda x: float(x.split()[0]) if not isinstance(x, float) else x
    )
    plot_df["CI Lower"] = plot_df["F1 string"].apply(
        lambda x: float(x.split("(")[1].split(",")[0])
    )
    plot_df["CI Upper"] = plot_df["F1 string"].apply(
        lambda x: float(x.split(",")[1].split(")")[0])
    )

    selection = alt.selection_point(
        fields=["Label"],
        bind="legend",
        value=[{"Label": "Average"}],
    )

    base = (
        alt.Chart(plot_df)
        .mark_point(filled=True, size=100)
        .encode(
            x=alt.X("F1", title="F1"),
            y="Models",
            color="Label",
            tooltip=[
                "Models",
                "Label",
                alt.Tooltip("F1 string", title="F1"),
            ],
            opacity=alt.condition(selection, alt.value(1), alt.value(0.0)),
            # only show the tooltip when when the label is selected
        )
    )
    error_bars = (
        alt.Chart(plot_df)
        .mark_errorbar(ticks=False)
        .encode(
            x=alt.X("CI Lower", title="F1"),
            x2="CI Upper",
            y="Models",
            color="Label",
            opacity=alt.condition(selection, alt.value(1), alt.value(0.0)),
        )
    )

    chart = error_bars + base

    return chart.add_params(selection).properties(width=800, height=400)
