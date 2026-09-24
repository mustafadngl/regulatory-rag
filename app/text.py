"""Typographic normalisation.

Both the Official Journal and the language model emit typographic whitespace and hyphens:
`Article\u00a06`, `Article\u202f99`, `high\u2011risk`. They look identical to a reader and
are invisible in most terminals, but they break exact string matching, which is how citation
checks in the evaluation harness decide whether an answer is grounded.
"""

TRANSLATIONS = str.maketrans(
    {
        "\u00a0": " ",  # no-break space
        "\u202f": " ",  # narrow no-break space
        "\u2009": " ",  # thin space
        "\u2011": "-",  # non-breaking hyphen
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
    }
)


def normalise_typography(text: str) -> str:
    return text.translate(TRANSLATIONS)
