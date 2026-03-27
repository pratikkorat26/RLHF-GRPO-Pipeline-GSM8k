"""
Shared numeric normalization utilities used by pipeline and reward modules.

Handles: integers, negatives, comma-separated thousands, currency ($),
percentages (%), fractions (a/b), decimals, and English word numbers (zero–twelve).
"""

import re

NUMERIC_RE = re.compile(
    r"""
    [\$\s]*
    (-?)
    ([\d]{1,3}(?:,\d{3})*(?:\.\d+)?
    |
    \d+\.\d+
    |
    \d+)
    \s*%?
    """,
    re.VERBOSE,
)
FRACTION_RE = re.compile(r"(-?\d+)\s*/\s*(-?\d+)")
WORD_NUMBERS: dict[str, str] = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}


def normalise_numeric(raw: str) -> str | None:
    """Return a canonical numeric string from *raw*, or None if not parseable.

    Resolution order:
    1. Fraction  (e.g. ``1/2`` → ``"0.5"``)
    2. Numeric   (e.g. ``$1,200`` → ``"1200"``, ``35%`` → ``"35"``)
    3. Word      (e.g. ``twelve`` → ``"12"``)
    """
    raw = raw.strip()

    frac = FRACTION_RE.search(raw)
    if frac:
        num, den = int(frac.group(1)), int(frac.group(2))
        if den == 0:
            return None
        val = num / den
        return str(int(val)) if val == int(val) else f"{val:.6g}"

    m = NUMERIC_RE.search(raw)
    if m:
        normalised = m.group(1) + m.group(2).replace(",", "")
        try:
            f = float(normalised)
            return str(int(f)) if f == int(f) else str(f)
        except ValueError:
            return normalised

    low = raw.lower().strip(".,!? ")
    return WORD_NUMBERS.get(low)
