from fedci.questions.base import QuestionSpec

SPEC = QuestionSpec(
    id="Q2",
    slug="channels",
    title="The mechanism (parked)",
    question="Through which transmitted channels does the effect flow -- rates, dollar, oil, credit, volatility?",
    estimand=(
        "Deferred channel decomposition; causal mediation requires a separate identification design."
    ),
    currency="Meetings; channels are correlated outcomes, not independent shocks",
    method="Parked: no active channel modeling or additional data collection.",
    owner="Deferred",
    depends_on=("Q1",),
    datasets=("fomc_meetings", "mps_surprises", "etf_returns", "fred_controls", "event_study_table"),
    notes=(
        "Outside the active Q1 -> Q3 -> Q4 -> Q5 chain. Sector and exposure comparisons remain in Q3.",
    ),
)
