from data.synthetic import generators as g
from pa_copilot.rag import corpus


def test_every_policy_has_a_guidance_file():
    mapped = {pid for pid, _ in corpus.POLICY_BY_FILE.values()}
    assert mapped == {s["policy_id"] for s in g.SERVICES}


def test_load_is_deterministic_and_tagged():
    a = corpus.load_guidance()
    b = corpus.load_guidance()
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len(a) >= 24  # 6 docs, >= 4 clauses each
    psg = [c for c in a if c.policy_id == "PA-PSG"]
    assert psg and all(c.service_code == "95810" for c in psg)
    assert {c.section for c in psg} >= {"Indications", "Step therapy", "Exclusions"}


def test_chunk_ids_unique():
    ids = [c.chunk_id for c in corpus.load_guidance()]
    assert len(ids) == len(set(ids))
