from fedci.questions.base import QuestionSpec

SPEC = QuestionSpec(
    id="Q1",
    slug="total_effect",
    title="The causal base",
    question="Does an unanticipated Fed policy shock move equity values at all, and by how much?",
    estimand=(
        "Total effect of the monetary-policy surprise on returns over the announcement window. "
        "The treatment is the surprise (high-frequency rate-sensitive instruments), not the announced target."
    ),
    currency=(
        "Meetings: 245 scheduled statement events with announcement-window SP500 and a constructed "
        "surprise (1996-2026); 30 of them also have daily ETF outcomes and richer pre-event controls."
    ),
    method=(
        "Partially linear DML with pre-event controls and simple nuisance learners; OLS baseline. "
        "Audit USMPD announcement-window SP500 outcomes and preselect shock robustness variants."
    ),
    owner="DML workstream",
    datasets=("usmpd_statements", "mps_surprises", "treasury_1y", "event_study_table"),
    data_needed=("Sector ETF returns extended back to 1998",),
    notes=(
        "Killer: the information effect (a cut can be bad news).",
        "Killer: meetings confounded by same-day macro releases.",
        "Unscheduled actions are a separate treatment -- never pooled.",
    ),
)
