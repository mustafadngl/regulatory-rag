from app.text import normalise_typography


def test_no_break_spaces_become_ordinary_spaces() -> None:
    assert normalise_typography("Article\u00a06") == "Article 6"


def test_narrow_no_break_spaces_become_ordinary_spaces() -> None:
    """The model emits these, and they silently break citation matching."""
    assert normalise_typography("Article\u202f99(4)") == "Article 99(4)"


def test_non_breaking_hyphens_become_ordinary_hyphens() -> None:
    assert normalise_typography("high\u2011risk") == "high-risk"


def test_dashes_are_normalised() -> None:
    assert normalise_typography("5(1)(a)\u2013(b)") == "5(1)(a)-(b)"


def test_plain_text_is_unchanged() -> None:
    assert normalise_typography("Article 6(2) applies") == "Article 6(2) applies"


def test_normalised_text_is_pure_ascii() -> None:
    messy = "Article\u202f5(1)(a)\u2013(b) on high\u2011risk\u2009systems"

    assert normalise_typography(messy).isascii()
