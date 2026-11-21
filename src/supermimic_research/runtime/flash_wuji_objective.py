"""Bind the tagged FlashCHORD objective to the single Wuji right hand."""

from flash_chord.objectives.config import ObjectiveConfig


def make_right_objective(**terms):
    # FlashCHORD v0.4 compares this value directly with ContactLayout.sides (a tuple).
    return ObjectiveConfig(sides=("right",), **terms)
