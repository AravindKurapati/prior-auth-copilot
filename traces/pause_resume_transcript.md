# AC-05 pause/resume -- two python processes, one shared checkpoint db
[pause pid=<pid>] driving case case-ac05-2proc to human_review's interrupt
[pause pid=<pid>] interrupt payload: {"case_id": "case-ac05-2proc", "decision": {"cited_criteria": [], "confidence": 0.92, "disposition": "approve", "human_review_required": true, "reviewer_summary": "auto-approved pending human sign-off"}, "necessity": null, "reason": "human review required", "request": null}
[pause pid=<pid>] paused=True
[resume pid=<pid>] built a fresh graph object against the same state db
[resume pid=<pid>] completed=True next=FINISH disposition=approve
RESULT: PASS
