from fedci.questions.base import QuestionSpec

SPEC = QuestionSpec(
    id="Q3",
    slug="susceptibility",
    title="The susceptibility map",
    question="Given firm state z and shock s, what is the expected response mu(z, s)? Which firms respond more?",
    estimand="Expected firm event-day return given pre-event state and realized shock; predictive, not automatically causal.",
    currency="Firm-meeting panel; cross-sectional width helps, but independent shocks remain meeting-level.",
    method=(
        "Ridge/hierarchical interactions -> boosting -> TCN or masked-patch encoder with shock conditioning. "
        "Cross-sectional and graph layers are later candidates; promote on held-out response prediction."
    ),
    owner="ML workstream",
    depends_on=("Q1",),
    datasets=("fomc_meetings", "mps_surprises", "usmpd_minutes", "mps_minutes_surprises", "bls_cpi_release_dates"),
    data_needed=(
        "Firm daily bars 1998-present (S&P 100 first, then broaden)",
        "SEC EDGAR as-filed XBRL fundamentals with filing dates (point-in-time)",
        "Auxiliary event calendar: CPI, payrolls, ECB, minutes, press conferences (training only)",
    ),
    notes=(
        "Scheduled FOMC meetings are the test population; auxiliary training must precede each cutoff.",
        "Q3 is the main contribution and does not require Q4 reversal. Its out-of-sample errors define the Q4 gap.",
        "Inputs end at the previous close, apart from the realized shock; target is event-day daily return.",
    ),
)
