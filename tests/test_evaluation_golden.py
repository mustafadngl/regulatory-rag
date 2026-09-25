from pathlib import Path

import pytest

from app.evaluation.golden import load_golden_set

REAL_GOLDEN_SET = Path("evaluation/golden_set.yaml")


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "golden.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_questions_are_loaded(tmp_path: Path) -> None:
    path = write(tmp_path, "- id: a\n  question: why?\n  expected_articles: ['6']\n")

    questions = load_golden_set(path)

    assert questions[0].id == "a"
    assert questions[0].answerable is True


def test_out_of_scope_questions_are_marked(tmp_path: Path) -> None:
    path = write(tmp_path, "- id: a\n  question: why?\n  answerable: false\n")

    assert load_golden_set(path)[0].answerable is False


def test_an_answerable_question_must_expect_articles(tmp_path: Path) -> None:
    path = write(tmp_path, "- id: a\n  question: why?\n")

    with pytest.raises(ValueError, match="needs expected articles"):
        load_golden_set(path)


def test_an_unanswerable_question_cannot_expect_articles(tmp_path: Path) -> None:
    path = write(
        tmp_path, "- id: a\n  question: why?\n  answerable: false\n  expected_articles: ['6']\n"
    )

    with pytest.raises(ValueError, match="cannot expect articles"):
        load_golden_set(path)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    body = "- id: a\n  question: one?\n  expected_articles: ['6']\n"
    path = write(tmp_path, body + "- id: a\n  question: two?\n  expected_articles: ['7']\n")

    with pytest.raises(ValueError, match="duplicate"):
        load_golden_set(path)


def test_an_empty_file_yields_no_questions(tmp_path: Path) -> None:
    assert load_golden_set(write(tmp_path, "")) == []


@pytest.mark.skipif(not REAL_GOLDEN_SET.exists(), reason="run from the repository root")
def test_the_projects_golden_set_is_valid() -> None:
    questions = load_golden_set(REAL_GOLDEN_SET)

    assert len(questions) >= 20
    assert any(not q.answerable for q in questions), "out-of-scope questions detect hallucination"
