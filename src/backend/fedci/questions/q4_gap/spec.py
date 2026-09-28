from fedci.questions.base import QuestionSpec

SPEC = QuestionSpec(
    id="Q4",
    slug="gap",
    title="The gap (the hinge)",
    question=(
        "Does actual response minus Q3's out-of-sample expected response predict subsequent "
        "abnormal returns (reversal or continuation)?"
    ),
    estimand=(
        "Signed gap = actual event-day return minus Q3's historical-only prediction. "
        "Relation to factor-adjusted returns from t+1 close to t+20 close."
    ),
    currency="(a) events x firms; (b) meetings -- drift-regression errors cluster by meeting.",
    method="Per-meeting gap regressions; independent ordinary-day benchmark and matched-placebo comparisons.",
    owner="ML workstream",
    depends_on=("Q1", "Q3"),
    datasets=("fomc_meetings", "mps_surprises", "etf_returns", "fred_controls"),
    data_needed=(
        "Firm daily bars 1998-present",
        "Placebo harness: non-FOMC days matched on pre-event conditions",
        "Saved out-of-sample Q3 predictions; historical factor data for future-return adjustment",
    ),
    gate="Select Q3 on earlier response-prediction performance; freeze it before Q4 held-out evaluation.",
    notes=(
        "Prediction errors include omitted information and estimation error; reversal does not prove mispricing.",
        "Q3 remains a valid contribution if Q4 is null. The ordinary-day residual is a comparison signal.",
    ),
)
