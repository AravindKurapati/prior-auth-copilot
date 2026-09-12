"""Worker ReAct loops (design.md §3.3) — each emits one validated Pydantic
object across the node boundary. _react.py is the shared loop; intake.py /
benefit_check.py (this PR) and medical_necessity.py / decision_draft.py /
human_review.py (PR5b) each build on it."""
