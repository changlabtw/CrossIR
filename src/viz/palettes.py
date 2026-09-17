"""Colour-blind-safe palettes.

The Okabe-Ito palette, chosen because its eight hues stay mutually
distinguishable under all three common forms of colour vision deficiency
(protanopia, deuteranopia and tritanopia) while still reading as ordinary
colours to everyone else.

The pairs that cause trouble elsewhere -- red with green, and red with blue --
do not appear here. Where a figure in this project needs colour to carry
meaning, its accessible variant is built from these values.
"""

from matplotlib.colors import LinearSegmentedColormap

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}
"""The Okabe-Ito qualitative palette, by name."""

NEUTRAL_GREY = "#999999"
"""Grey for points that carry no category, distinguishable from every hue above."""

COLOR_BLIND_CMAP = LinearSegmentedColormap.from_list(
    "okabe_ito_blue_orange", [OKABE_ITO["blue"], OKABE_ITO["orange"]]
)
"""Two-ended low-to-high scale, blue for low and orange for high.

Replaces the red-to-blue default of SHAP's summary plot.
"""

COLOR_BLIND_DIVERGING = LinearSegmentedColormap.from_list(
    "okabe_ito_diverging",
    [OKABE_ITO["blue"], "#F7F7F7", OKABE_ITO["orange"]],
)
"""Diverging scale through near-white, for correlations and other signed values.

Replaces ``coolwarm``, whose red and blue ends are the hardest pair to separate
under protanopia and deuteranopia.
"""

COLOR_BLIND_CATEGORICAL = [
    OKABE_ITO["blue"],
    OKABE_ITO["orange"],
    OKABE_ITO["bluish_green"],
    OKABE_ITO["reddish_purple"],
]
"""Four categorical hues, in the order to assign them.

Replaces ``tab10``, which pairs red with green within its first four entries.
"""
