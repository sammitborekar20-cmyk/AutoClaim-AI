from datetime import datetime, timezone


def assess_responsible_ai(
    detected_damages,
    text_analysis,
    claim_estimate,
    policy_analysis=None,
):
    """
    Responsible AI checks for the AutoClaim AI demonstration.

    This is a checklist and risk-flagging component, not a formal
    fairness certification or legal compliance assessment.
    """

    flags = []
    safeguards = []

    # 1. Human oversight
    safeguards.append(
        "A human assessor must verify the damage and make the final claim decision."
    )

    # 2. Confidence and uncertainty
    if not detected_damages:
        flags.append(
            "UNCERTAINTY: No damage was detected. A missed detection is possible."
        )
    else:
        low_confidence = [
            item for item in detected_damages
            if float(item.get("confidence", 0)) < 0.50
        ]

        if low_confidence:
            flags.append(
                f"UNCERTAINTY: {len(low_confidence)} detection(s) have confidence "
                "below 50%. Manual inspection is recommended."
            )

        safeguards.append(
            "Detection confidence is displayed as model confidence, not the "
            "probability that an insurance claim is valid."
        )

    # 3. Text-image consistency
    reported_parts = set(
        part.lower()
        for part in text_analysis.get("reported_parts", [])
    )
    detected_types = set(
        item.get("type", "").lower()
        for item in detected_damages
    )

    if reported_parts and not detected_types:
        flags.append(
            "CONSISTENCY: The description mentions damage, but the image model "
            "did not detect a damage class."
        )
    elif reported_parts and detected_types:
        safeguards.append(
            "Text and image outputs are compared using simple keyword matching. "
            "A mismatch does not prove that a user is dishonest."
        )

    # 4. Fairness and bias
    flags.append(
        "FAIRNESS: Dataset performance across vehicle makes, models, lighting, "
        "weather, damage severity, and other relevant groups has not been "
        "established by this application."
    )

    safeguards.append(
        "Do not infer a person's protected characteristics or use them to "
        "determine claim outcomes."
    )

    # 5. Privacy
    safeguards.append(
        "Use only information needed for assessment. Avoid collecting "
        "unnecessary personal identifiers, faces, registration plates, "
        "or unrelated policy information."
    )

    # 6. Robustness and security
    safeguards.append(
        "Poor lighting, blur, unusual viewpoints, image manipulation, and "
        "unseen damage types may affect predictions. Verify suspicious inputs."
    )

    # 7. Accountability and transparency
    safeguards.append(
        "Record the model version, assessment timestamp, detected classes, "
        "confidence scores, and human review outcome when maintaining an "
        "assessment audit trail."
    )

    # 8. Estimate limitations
    if claim_estimate:
        flags.append(
            "COST: The displayed repair cost is illustrative unless validated "
            "against reliable repair-price data. It does not establish coverage."
        )

    # 9. Document extraction
    if policy_analysis:
        if not policy_analysis.get("success"):
            flags.append(
                "DOCUMENT: Policy text could not be extracted reliably. "
                "Inspect the original document manually."
            )
        else:
            safeguards.append(
                "Extracted policy details must be checked against the original "
                "policy wording before being used."
            )

    return {
        "framework": "Responsible AutoClaim AI",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "human_review_required": True,
        "automated_claim_decision": False,
        "risk_flags": flags,
        "safeguards": safeguards,
        "fairness_status": "Not yet empirically evaluated",
        "privacy_status": "Data minimization recommended",
        "accountability_status": "Human review and audit trail recommended",
        "limitations": [
            "This checklist does not certify legal or regulatory compliance.",
            "Fairness requires testing on representative, appropriately collected data.",
            "Human review is required before any real-world insurance decision.",
        ],
    }