# MCP tool-call transcript (Gemini agent -> pa MCP server)

> NOTE: tool calls and tool results below are REAL (captured from the pa MCP server via langchain-mcp-adapters). The agent's natural-language turns are representative — regenerate with a live GEMINI_API_KEY via `pytest -m slow -k transcript`.

## user

A provider requests prior auth for service 72148, diagnosis M54.16, member M100001.
Use the available tools to check the benefit and the coverage criteria, then summarize
what you found.

## assistant (reasoning)

I need two facts before I can say anything useful: (1) does this member's plan cover
CPT 72148 and does it require prior authorization, and (2) what does the prior-auth
policy for 72148 require, and is M54.16 an exclusion. I'll call `benefit_lookup` first,
then `criteria_check`.

## tool call: `benefit_lookup`

```json
{
  "member_id": "M100001",
  "service_code": "72148"
}
```

## tool result

```json
{
  "found": true,
  "member_id": "M100001",
  "service_code": "72148",
  "plan_id": "PLAN-GOLD-PPO",
  "covered": true,
  "requires_pa": true,
  "network_status": "in_network",
  "note": "service covered under plan"
}
```

## assistant (reasoning)

The service is covered under PLAN-GOLD-PPO, in-network, and it does require prior auth
(`requires_pa: true`) — so this request cannot be auto-approved on benefits alone. Now I
need the medical-necessity policy for 72148 and whether M54.16 is excluded.

## tool call: `criteria_check`

```json
{
  "service_code": "72148",
  "diagnosis_codes": [
    "M54.16"
  ]
}
```

## tool result

```json
{
  "found": true,
  "policy_id": "PA-MRI-LUMBAR",
  "title": "MRI Lumbar Spine - Medical Necessity",
  "required_conditions": [
    {
      "id": "MRI-LS-1",
      "text": "At least 6 weeks of conservative therapy (PT, NSAIDs, or activity modification) documented."
    },
    {
      "id": "MRI-LS-2",
      "text": "Persistent or progressive radicular pain, neurologic deficit, or red-flag findings."
    }
  ],
  "exclusions": [
    "Uncomplicated acute low back pain under 6 weeks without red flags.",
    "Repeat MRI within 12 months without a clinical change."
  ],
  "evidence_requirements": [
    "Dates and duration of conservative care.",
    "Neurologic exam findings or imaging red flags."
  ],
  "excluded_diagnoses": [
    "M54.5"
  ],
  "status": "indeterminate",
  "note": "policy criteria require clinical assessment; mechanical check cannot verify conditions"
}
```

## assistant (reasoning)

M54.16 (radiculopathy, lumbar region) is not in `excluded_diagnoses` (only M54.5 is), so
the mechanical screen does not exclude the request. But `status` is `indeterminate`: the
policy needs documented conservative therapy of at least 6 weeks (MRI-LS-1) and
persistent/progressive radicular findings (MRI-LS-2), and the evidence requirements call
for dates/duration of conservative care and neurologic exam findings. Only a human
reviewer can confirm those from the chart. I have enough to summarize.

## final answer

Benefit: CPT 72148 (MRI lumbar spine) is covered for member M100001 under PLAN-GOLD-PPO,
in-network, and it **requires prior authorization** — it cannot be auto-approved on
benefits alone.

Coverage criteria (policy PA-MRI-LUMBAR): the diagnosis M54.16 is **not** a documented
exclusion, so the request is not auto-denied. However the mechanical check returns
**indeterminate** — a reviewer must verify from the clinical record:

- at least 6 weeks of documented conservative therapy (PT, NSAIDs, or activity
  modification), with dates and duration (MRI-LS-1); and
- persistent or progressive radicular pain, a neurologic deficit, or red-flag findings,
  with the supporting neurologic exam / imaging notes (MRI-LS-2).

Recommendation: route to clinical review and request the conservative-care history and
neurologic exam findings from the ordering provider.
