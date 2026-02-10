from __future__ import annotations
from .config import TARGET_BRANDS

BRAND_PROMPTS: dict[str, list[str]] = {
    "cocacola": [
        "coca cola logo",
        "coca-cola logo",
        "the logo of coca cola",
        "a photo of the coca cola logo",
    ],
    "disney": [
        "disney logo",
        "walt disney logo",
        "the logo of disney",
        "a photo of the disney logo",
    ],
    "mcdonalds": [
        "mcdonalds logo",
        "mcDonald's logo",
        "the golden arches logo",
        "a photo of the mcdonalds logo",
    ],
    "starbucks": [
        "starbucks logo",
        "the starbucks siren logo",
        "the logo of starbucks",
        "a photo of the starbucks logo",
    ],
}

BACKGROUND_PROMPTS: list[str] = [
    "no logo",
    "an image without a logo",
    "background without the target logos",
    "a random photo without brand logos",
    "a logo of a different brand",
]

def sample_prompt(label: str, rng) -> str:
    if label == "background":
        return rng.choice(BACKGROUND_PROMPTS)
    return rng.choice(BRAND_PROMPTS[label])

def get_brand_prompt_map() -> dict[str, list[str]]:
    return {k: BRAND_PROMPTS[k] for k in TARGET_BRANDS}
