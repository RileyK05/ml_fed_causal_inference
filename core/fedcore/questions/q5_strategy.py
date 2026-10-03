from fedcore.questions.base import QuestionSpec

SPEC = QuestionSpec(
    id="Q5",
    slug="strategy",
    title="The strategy (stretch)",
    question="Does acting on the gap signal earn abnormal returns after costs?",
    estimand="Abnormal return of a gap-signal policy, after costs.",
    currency="Meetings",
    method="Simple constrained gap portfolios; chronological held-out backtest with executable timing and costs.",
    owner="Stretch goal",
    depends_on=("Q4",),
    datasets=("fomc_meetings", "etf_returns"),
    gate="Built only if Q4's held-out relation survives benchmark/placebo comparisons. Direction fixed on validation data.",
    notes=("Default entry: next trading-day close, after the gap is observable; exit at t+20 close.",),
)
