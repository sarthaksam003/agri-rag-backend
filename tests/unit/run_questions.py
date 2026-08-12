import json
import uuid
from pathlib import Path

import requests
from tqdm import tqdm

API_URL = "http://localhost:38779/api/v1/multiquery-chat/message"
HEADERS = {
    "accept": "application/json",
    "x-tenant-id": "demo-tenant",
    "Content-Type": "application/json",
}
ANSWER_FILE = Path("answer.md")
CROSS_REF_FILE = Path("cross_reference.md")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def load_questions() -> list[str]:
    """Return list of questions (strip blank lines)."""
    return [
        "Whether the latest version of the CCA audit criteria been used for conducting the audit?",
        "Are all control points listed in the CCA audit criteria covered in the audit report?",
        "Does the audit report contain any missing control points compared with the CCA audit criteria?",
        "Has any change, modification, or manipulation been made to the control descriptions from the published audit criteria?",
        "Are all auditors who performed the audit named in the report, and are they eligible and empanelled as per CCA requirements?",
        "Has the auditor listed their qualifications and experience as required under CCA empanelment?",
        "Was the audit conducted by the eligible manpower as per the approved empanelment?",
        "Are the CPS, CP, and audit templates consistent with the CA’s current licensed services and operational setup?",
        "List all mandatory documents that were required to be submitted along with the audit report as per CCA audit criteria, and indicate their submission status (submitted, not submitted, or incomplete).",
        "Does the audit report include the 500 sample certificates along with their Issuance and Operation Guidelines (IOG) compliance details?",
        "Are there any typographical or copy–paste errors such as incorrect CA names or unrelated details?",
        "Are there any inconsistencies between the CA’s declared scope of services (such as eSign, SSL, TSA, etc.) and the services for which compliance has been reported?",
        "Are there discrepancies between the auditor’s narrative explanation and the compliance marking for any control point?",
        "Does the report show incomplete data entry, or unlinked annexures and evidence documents?",
        "Has the auditor validated and verified each control point as described under the official audit criteria?",
        "Are there any control points where compliance has been marked but the justification or evidence provided does not support it?",
        "Are there any mandatory controls marked as “Not Applicable” without proper justification or explanation?",
        "Are there unlinked or missing annexures or references to required supporting evidence?",
        "Does the audit evidence confirm that all mandatory internal reviews, such as ISMS, risk, or DR drills, were actually performed and documented?",
        "Are there any inconsistencies or contradictions between auditor responses in different sections (e.g., DR mentioned as both tested and untested)?",
        "Is there evidence of repetitive or template-style text across unrelated sections of the report?",
        "Has the CA provided proper explanation or annexures for each “Not Applicable” control point?",
        "Are there deviations or mismatches between the CA’s actual infrastructure (e.g., DC/DR details, hardware, systems) and what is described in the audit report?",
        "Are all non-compliances mentioned in the detailed report also included in the summary of non-compliances, as per the required audit format?",
        "Are there any repeat non-compliances from previous audits indicating unresolved issues?",
        "Has the CA’s management reviewed, approved, and signed the audit report, taking accountability for its content?",
        "Has the auditor provided evidence-based verification, or is the report largely descriptive without supporting proof?",
        "Has the auditor carried out independent testing or verification rather than relying only on CA declarations?",
        "Are there signs of incomplete coverage, bias, or uniform compliance markings suggesting inconsistency in auditing?",
        "Was the audit timeline and scope (covering DC, DR, eSign, TSA, ESP etc.) reasonable for the CA’s operations?",
        "Does the report distinguish between procedural deviations and actual non-compliances, or are all issues treated equally?",
        "Does the report clearly identify which findings could impact the reliability or trustworthiness of certificate issuance?",
        "Can any operational or governance issues be inferred from the audit observations?",
        "If corrective actions are mentioned, are they clear, assigned to responsible officials, and time-bound?",
        "Are the corrective actions proposed by the CA consistent with the non-compliances identified by the auditor?",
        "Based on the report and supporting documents, should the CCA issue an advisory, seek clarification, or initiate a license review for the CA?",
    ]


def build_payload(question: str) -> dict:
    return {
        "message": question,
        "session_id": str(uuid.uuid4()),
        "use_multiquery": True,
        "max_queries": 6,
    }


def fetch_answer(question: str) -> dict:
    """Return parsed JSON from the API."""
    payload = build_payload(question)
    resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    questions = load_questions()
    all_responses = []

    for q in tqdm(questions, desc="Querying API"):
        try:
            data = fetch_answer(q)
        except Exception as exc:
            # Build a stub so the loop continues
            data = {"error": str(exc), "message": q}
        all_responses.append(data)

    # ---------------- cross_reference.md ----------------
    CROSS_REF_FILE.write_text(
        "# Raw JSON responses\n\n" + json.dumps(all_responses, indent=2)
    )

    # ---------------- answer.md ----------------
    lines = ["# Question & Answer Summary\n"]
    for idx, data in enumerate(all_responses, 1):
        q = data.get("message", "")
        a = data.get("response", data.get("error", "No answer received"))
        lines.append(f"\n## Q{idx}. {q}\n\n{a}\n")
    ANSWER_FILE.write_text("\n".join(lines))

    print(f"Finished. Wrote {CROSS_REF_FILE} and {ANSWER_FILE}")


if __name__ == "__main__":
    main()
