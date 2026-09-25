import pytest

from app.evaluation.metrics import (
    QuestionOutcome,
    Report,
    Thresholds,
    cited_articles,
    first_hit_rank,
)


def outcome(**overrides) -> QuestionOutcome:
    defaults = dict(
        id="q",
        question="?",
        answerable=True,
        expected_articles=["6"],
        retrieved_articles=["6"],
        answer="The answer (Article 6(2)).",
        grounded=True,
    )
    return QuestionOutcome(**{**defaults, **overrides})


class TestCitationExtraction:
    def test_long_form_is_recognised(self) -> None:
        assert cited_articles("see Article 99(4)") == ["99"]

    @pytest.mark.parametrize("text", ["Art 14(1-2)", "Art. 14", "Art14", "Article 14"])
    def test_abbreviations_are_recognised(self, text: str) -> None:
        """Missing these scored correctly-cited answers as uncited on the first real run."""
        assert cited_articles(text) == ["14"]

    def test_multiple_articles_keep_first_appearance_order(self) -> None:
        assert cited_articles("Art 53(1), then Article 23, then Art 53(3)") == ["53", "23"]

    def test_text_without_citations_yields_nothing(self) -> None:
        assert cited_articles("No references at all.") == []


class TestRanking:
    def test_the_first_expected_article_sets_the_rank(self) -> None:
        assert first_hit_rank(["6"], ["99", "6", "16"]) == 2

    def test_a_missing_article_has_no_rank(self) -> None:
        assert first_hit_rank(["6"], ["99", "16"]) is None

    def test_any_expected_article_counts(self) -> None:
        assert first_hit_rank(["12", "19"], ["19"]) == 1


class TestOutcome:
    def test_a_correct_answer_passes(self) -> None:
        assert outcome().passed is True

    def test_a_missing_citation_fails(self) -> None:
        result = outcome(answer="The answer, uncited.")

        assert result.passed is False
        assert "citation" in result.failure_reason

    def test_retrieving_the_wrong_articles_fails(self) -> None:
        result = outcome(retrieved_articles=["99"], answer="uncited")

        assert result.passed is False
        assert "retrieved" in result.failure_reason

    def test_refusing_an_answerable_question_fails(self) -> None:
        result = outcome(grounded=False)

        assert result.passed is False
        assert "refused an answerable question" in result.failure_reason

    def test_refusing_an_out_of_scope_question_passes(self) -> None:
        result = outcome(answerable=False, expected_articles=[], grounded=False, answer="")

        assert result.passed is True

    def test_answering_an_out_of_scope_question_fails(self) -> None:
        result = outcome(answerable=False, expected_articles=[], grounded=True)

        assert result.passed is False
        assert "cannot support" in result.failure_reason

    def test_an_errored_question_never_passes(self) -> None:
        result = outcome(error="HTTP 503")

        assert result.passed is False
        assert "not evaluated" in result.failure_reason


class TestReport:
    def test_perfect_results_score_one(self) -> None:
        report = Report([outcome(id="a"), outcome(id="b")])

        assert report.retrieval_recall == 1.0
        assert report.citation_accuracy == 1.0

    def test_recall_counts_only_answerable_questions(self) -> None:
        report = Report(
            [
                outcome(id="a"),
                outcome(id="b", retrieved_articles=["99"]),
                outcome(id="c", answerable=False, expected_articles=[], grounded=False),
            ]
        )

        assert report.retrieval_recall == 0.5

    def test_reciprocal_rank_rewards_higher_placement(self) -> None:
        report = Report(
            [
                outcome(id="a", retrieved_articles=["6"]),
                outcome(id="b", retrieved_articles=["99", "6"]),
            ]
        )

        assert report.mean_reciprocal_rank == 0.75

    def test_refusal_accuracy_counts_only_out_of_scope_questions(self) -> None:
        report = Report(
            [
                outcome(id="a"),
                outcome(id="b", answerable=False, expected_articles=[], grounded=False),
                outcome(id="c", answerable=False, expected_articles=[], grounded=True),
            ]
        )

        assert report.refusal_accuracy == 0.5

    def test_false_refusal_rate_tracks_wrongly_refused_questions(self) -> None:
        report = Report([outcome(id="a"), outcome(id="b", grounded=False)])

        assert report.false_refusal_rate == 0.5

    def test_errors_are_excluded_from_every_metric(self) -> None:
        """An outage must not be scored as a regression."""
        report = Report([outcome(id="a"), outcome(id="b", error="HTTP 503")])

        assert report.retrieval_recall == 1.0
        assert report.error_rate == 0.5
        assert len(report.evaluated) == 1

    def test_failures_exclude_errored_questions(self) -> None:
        report = Report([outcome(id="a", error="HTTP 503")])

        assert report.failures == []

    def test_an_empty_report_does_not_divide_by_zero(self) -> None:
        report = Report([])

        assert report.retrieval_recall == 0.0
        assert report.mean_reciprocal_rank == 0.0

    def test_the_serialised_report_carries_metrics_and_questions(self) -> None:
        data = Report([outcome()]).as_dict()

        assert data["totals"]["questions"] == 1
        assert data["metrics"]["retrieval_recall"] == 1.0
        assert data["questions"][0]["cited"] == ["6"]


class TestThresholds:
    def test_meeting_every_threshold_reports_no_violations(self) -> None:
        report = Report(
            [outcome(), outcome(answerable=False, expected_articles=[], grounded=False)]
        )

        assert Thresholds().violations(report) == []

    def test_low_recall_is_reported(self) -> None:
        report = Report([outcome(retrieved_articles=["99"], answer="uncited")])

        assert any("retrieval_recall" in v for v in Thresholds().violations(report))

    def test_a_hallucinated_answer_breaches_refusal_accuracy(self) -> None:
        report = Report([outcome(answerable=False, expected_articles=[], grounded=True)])

        assert any("refusal_accuracy" in v for v in Thresholds().violations(report))

    def test_excessive_false_refusals_are_reported(self) -> None:
        report = Report([outcome(grounded=False)])

        assert any("false_refusal_rate" in v for v in Thresholds().violations(report))
