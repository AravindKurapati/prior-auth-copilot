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
        "no_pa_required", "unknown_member",
    }
    inj = samples["injection_prompt_injection"]["raw_provider_text"].lower()
    assert "ignore your instructions" in inj
    assert "diagnosis_codes" not in samples["knee_scope_missing_info"]["structured"]


def test_no_pa_required_sample_resolves_to_a_no_pa_benefit():
    samples = g.build_samples()
    benefits = g.build_benefits()
    s = samples["no_pa_required"]
    svc = benefits[s["member_id"]]["covered_services"][s["structured"]["service_code"]]
    assert svc["covered"] is True
    assert svc["requires_pa"] is False


def test_unknown_member_sample_misses_every_corpus():
    samples = g.build_samples()
    s = samples["unknown_member"]
    assert s["member_id"] not in g.build_benefits()
    assert s["structured"]["provider_npi"] not in g.build_providers()


def test_clinical_guidance_one_doc_per_policy():
    docs = g.build_clinical_guidance()
    assert len(docs) == len(g.SERVICES)


def test_criteria_have_structured_diagnosis_exclusions():
    criteria = g.build_criteria()
    for pol in criteria.values():
        assert isinstance(pol["excluded_diagnoses"], list)
    assert "M54.5" in criteria["PA-MRI-LUMBAR"]["excluded_diagnoses"]
    assert criteria["PA-PSG"]["excluded_diagnoses"] == []
