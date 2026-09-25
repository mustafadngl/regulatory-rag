"""Run the golden set against the live service and gate on the results.

    python -m app.evaluation.run --output evaluation/report.json

Exits non-zero when any threshold is breached, which is what makes this a gate rather than
a report. Thresholds live in `Thresholds` and can be overridden on the command line.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from app.config import get_settings
from app.evaluation.golden import GoldenQuestion, load_golden_set
from app.evaluation.metrics import QuestionOutcome, Report, Thresholds
from app.main import build_rag_service
from app.rag import RagService
from app.upstream import UpstreamError

DEFAULT_GOLDEN_SET = Path("evaluation/golden_set.yaml")


def rescore(report_path: Path) -> Report:
    """Recompute metrics from a stored report.

    Reports keep the full answer text, so a change to the scoring rules can be applied to past
    runs without paying for the model calls again.
    """
    stored = json.loads(report_path.read_text(encoding="utf-8"))
    return Report(
        [
            QuestionOutcome(
                id=q["id"],
                question=q["question"],
                answerable=q["answerable"],
                expected_articles=q["expected_articles"],
                retrieved_articles=q["retrieved_articles"],
                answer=q["answer"],
                grounded=q["grounded"],
                error=q.get("error"),
            )
            for q in stored["questions"]
        ]
    )


def evaluate_question(service: RagService, question: GoldenQuestion) -> QuestionOutcome:
    try:
        result = service.answer(question.question)
    except UpstreamError as exc:
        return QuestionOutcome(
            id=question.id,
            question=question.question,
            answerable=question.answerable,
            expected_articles=question.expected_articles,
            retrieved_articles=[],
            answer="",
            grounded=False,
            error=str(exc)[:200],
        )

    retrieved = []
    for citation in result.citations:
        if citation.article and citation.article not in retrieved:
            retrieved.append(citation.article)

    return QuestionOutcome(
        id=question.id,
        question=question.question,
        answerable=question.answerable,
        expected_articles=question.expected_articles,
        retrieved_articles=retrieved,
        answer=result.answer,
        grounded=result.grounded,
    )


def run(service: RagService, questions: list[GoldenQuestion]) -> Report:
    outcomes = []
    for index, question in enumerate(questions, start=1):
        started = time.perf_counter()
        outcome = evaluate_question(service, question)
        elapsed = time.perf_counter() - started
        status = "ERROR" if outcome.error else ("pass" if outcome.passed else "FAIL")
        progress = f"[{index:>2}/{len(questions)}]"
        print(f"{progress} {status:<5} {outcome.id}  ({elapsed:.1f}s)", flush=True)
        if not outcome.passed:
            print(f"          {outcome.failure_reason}", flush=True)
        outcomes.append(outcome)
    return Report(outcomes)


def print_summary(report: Report) -> None:
    totals = report.as_dict()["totals"]
    print()
    if report.errored:
        print(f"{totals['errored']} question(s) could not be evaluated and are excluded")
    print(f"Passed {totals['passed']}/{totals['evaluated']} evaluated questions")
    print(f"  retrieval recall      {report.retrieval_recall:.2f}")
    print(f"  mean reciprocal rank  {report.mean_reciprocal_rank:.2f}")
    print(f"  citation accuracy     {report.citation_accuracy:.2f}")
    print(f"  refusal accuracy      {report.refusal_accuracy:.2f}")
    print(f"  false refusal rate    {report.false_refusal_rate:.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET)
    parser.add_argument("--output", type=Path, help="write the full report as JSON")
    parser.add_argument("--min-retrieval-recall", type=float, default=0.90)
    parser.add_argument("--min-citation-accuracy", type=float, default=0.85)
    parser.add_argument("--min-refusal-accuracy", type=float, default=1.00)
    parser.add_argument("--max-false-refusal-rate", type=float, default=0.10)
    parser.add_argument(
        "--max-error-rate",
        type=float,
        default=0.20,
        help="above this share of unevaluable questions the run is inconclusive, not failing",
    )
    parser.add_argument(
        "--rescore",
        type=Path,
        help="recompute metrics from an existing report instead of calling the model",
    )
    args = parser.parse_args()

    if args.rescore:
        report = rescore(args.rescore)
        print(f"Rescoring {len(report.outcomes)} questions from {args.rescore}")
    else:
        settings = get_settings()
        if not settings.llm_api_key:
            print("LLM_API_KEY is not set; cannot evaluate", file=sys.stderr)
            return 2

        questions = load_golden_set(args.golden_set)
        print(f"Evaluating {len(questions)} questions against {settings.llm_model}\n")
        report = run(build_rag_service(settings), questions)

    print_summary(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
        print(f"\nReport written to {args.output}")

    if report.error_rate > args.max_error_rate:
        print(
            f"\nInconclusive: {report.error_rate:.0%} of questions could not be evaluated. "
            "This is an availability problem, not a quality regression.",
            file=sys.stderr,
        )
        return 2

    thresholds = Thresholds(
        retrieval_recall=args.min_retrieval_recall,
        citation_accuracy=args.min_citation_accuracy,
        refusal_accuracy=args.min_refusal_accuracy,
        max_false_refusal_rate=args.max_false_refusal_rate,
    )
    violations = thresholds.violations(report)
    if violations:
        print("\nQuality gate failed:")
        for violation in violations:
            print(f"  - {violation}")
        return 1

    print("\nQuality gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
