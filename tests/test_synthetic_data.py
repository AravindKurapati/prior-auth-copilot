from data.synthetic import generators as g


def test_generators_deterministic():
    assert g.build_benefits() == g.build_benefits()
    assert g.build_criteria() == g.build_criteria()


def test_every_service_has_a_policy_and_criteria():
    criteria = g.build_criteria()
    policy_ids = {s["policy_id"] for s in g.SERVICES}
    assert policy_ids == set(criteria)
    for pol in criteria.values():
        assert pol["required_conditions"]
        assert pol["evidence_requirements"]


def test_benefits_reference_real_services():
    codes = {s["service_code"] for s in g.SERVICES}
    for member in g.build_benefits().values():
        assert set(member["covered_services"]).issubset(codes)


def test_samples_cover_the_required_scenarios():
    samples = g.build_samples()
    assert set(samples) >= {
        "mri_lumbar_clearcut", "knee_scope_missing_info", "egd_not_covered",
        "psg_indeterminate", "injection_prompt_injection",
    }
    inj = samples["injection_prompt_injection"]["raw_provider_text"].lower()
    assert "ignore your instructions" in inj
    assert "diagnosis_codes" not in samples["knee_scope_missing_info"]["structured"]


def test_clinical_guidance_one_doc_per_policy():
    docs = g.build_clinical_guidance()
    assert len(docs) == len(g.SERVICES)
