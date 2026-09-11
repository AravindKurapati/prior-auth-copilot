from pa_copilot.memory import working


def test_remember_is_immutable_and_recallable():
    wm0 = {}
    wm1 = working.remember(wm0, "prior_denial_reason", "step therapy not documented")
    assert wm0 == {}
    assert working.recall(wm1, "prior_denial_reason") == "step therapy not documented"
    assert working.recall(wm1, "missing", default="?") == "?"


def test_search_working_matches_key_or_value():
    wm = working.remember(working.remember({}, "pt_weeks", 8), "dx", "M54.16")
    assert working.search_working(wm, "weeks") == [("pt_weeks", 8)]
    assert working.search_working(wm, "m54") == [("dx", "M54.16")]
