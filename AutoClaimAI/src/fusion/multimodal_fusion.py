def fuse_claim_modalities(
    detected_damages,
    text_analysis,
    claim_estimate,
    policy_analysis=None,
):
    """Combine modality outputs into one explainable claim assessment."""

    detected_types = {
        item.get("type", "").strip().lower()
        for item in detected_damages
        if item.get("type")
    }

    reported_parts = {
        part.strip().lower()
        for part in text_analysis.get("reported_parts", [])
    }

    # Basic keyword-level comparison, not semantic verification.
    matches = []
    differences = []

    for part in reported_parts:
        related = [
            damage
            for damage in detected_types
            if part in damage or damage in part
        ]

        if related:
            matches.append({
                "reported_part": part,
                "detected_damage": sorted(related),
            })
        else:
            differences.append(part)

    review_flags = []

    if not detected_damages:
        review_flags.append(
            "No damage was detected. Check the image and model output."
        )

    if differences:
        review_flags.append(
            "The description mentions parts not matched by the detector: "
            + ", ".join(differences)
        )

    if not text_analysis.get("description"):
        review_flags.append("No accident description was provided.")

    if policy_analysis and not policy_analysis.get("success"):
        review_flags.append(
            "The uploaded policy document could not be read successfully."
        )

    if not review_flags:
        review_flags.append(
            "No basic consistency issue was found. This does not verify the claim."
        )

    return {
        "detected_damages": detected_damages,
        "text_analysis": text_analysis,
        "claim_estimate": claim_estimate,
        "policy_analysis": policy_analysis,
        "text_image_matches": matches,
        "unmatched_reported_parts": differences,
        "review_flags": review_flags,
        "assessment_status": "Requires human verification",
        "disclaimer": (
            "This is a decision-support demonstration. It does not establish "
            "accident authenticity, liability, fraud, policy coverage, or "
            "insurance claim approval."
        ),
    }