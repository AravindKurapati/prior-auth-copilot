from pa_copilot.mcp_server import data_access as da


def test_benefit_hit_and_miss():
    hit = da.benefit_lookup("M100001", "72148")
    assert hit["found"] and hit["covered"] and hit["requires_pa"] is True
    assert hit["plan_id"]

    no_pa = da.benefit_lookup("M100001", "J0178")
    assert no_pa["found"] and no_pa["requires_pa"] is False

    miss = da.benefit_lookup("M999999", "72148")
    assert miss["found"] is False and miss["covered"] is False

    not_covered = da.benefit_lookup("M100001", "99999")
    assert not_covered["covered"] is False

    # genuine found-but-not-covered branch (M100003's EGD entry has covered: false)
    listed_not_covered = da.benefit_lookup("M100003", "43239")
    assert listed_not_covered["found"] is True and listed_not_covered["covered"] is False


def test_provider_hit_and_miss():
    hit = da.provider_lookup("1093817465")
    assert hit["found"] and hit["specialty"]
    assert da.provider_lookup("0000000000")["found"] is False


def test_criteria_indeterminate_excluded_and_not_found():
    ind = da.criteria_check("72148", ["M54.16"])
    assert ind["status"] == "indeterminate" and ind["policy_id"] == "PA-MRI-LUMBAR"
    assert ind["required_conditions"]

    exc = da.criteria_check("72148", ["M54.5"])
    assert exc["status"] == "excluded"

    nf = da.criteria_check("00000", [])
    assert nf["status"] == "not_found" and nf["found"] is False

    # a None diagnosis list must not raise (knee_scope_missing_info sample omits it)
    assert da.criteria_check("72148", None)["status"] == "indeterminate"


def test_policy_resource_helpers():
    assert da.get_policy("PA-PSG")["service_code"] == "95810"
    assert da.get_policy("NOPE")["found"] is False
    ids = {p["policy_id"] for p in da.list_policies()["policies"]}
    assert "PA-EGD" in ids and len(ids) == 6
