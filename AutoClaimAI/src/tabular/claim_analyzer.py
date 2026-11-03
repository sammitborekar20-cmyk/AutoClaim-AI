# Demonstration rates in Indian rupees.
# Replace these with locally validated repair-price data before real use.

DAMAGE_REPAIR_RATES = {
    "bonnet-dent": 4500,
    "boot-dent": 4000,
    "doorouter-dent": 3500,
    "fender-dent": 3500,
    "front-bumper-dent": 5000,
    "Front-windscreen-damage": 8000,
    "Headlight-damage": 4500,
    "quaterpanel-dent": 4000,
    "rear-bumper-dent": 5000,
    "Rear-windscreen-Damage": 7500,
    "roof-dent": 6000,
    "Runningboard-Damage": 3500,
    "Sidemirror-Damage": 2500,
    "Taillight-Damage": 3000,
}


def estimate_claim(
    detected_damages,
    vehicle_age,
    previous_claims,
    deductible,
    declared_repair_cost=0,
):
    """
    Build an illustrative claim estimate.

    detected_damages: list of dictionaries containing 'type' and optionally
                       'confidence'.
    vehicle_age: age of the vehicle in years.
    previous_claims: user-reported previous claim count.
    deductible: user-entered deductible in INR.
    declared_repair_cost: optional user-provided repair estimate in INR.
    """
    vehicle_age = max(0, int(vehicle_age))
    previous_claims = max(0, int(previous_claims))
    deductible = max(0.0, float(deductible))
    declared_repair_cost = max(0.0, float(declared_repair_cost))

    detected_types = list(
        dict.fromkeys(
            item.get("type", "")
            for item in detected_damages
            if item.get("type")
        )
    )

    # Estimate one repair item per unique detected damage class.
    component_estimate = sum(
        DAMAGE_REPAIR_RATES.get(damage_type, 3000)
        for damage_type in detected_types
    )

    # Prefer the user's declared repair estimate when it is greater than zero.
    base_estimate = (
        declared_repair_cost
        if declared_repair_cost > 0
        else component_estimate
    )

    # Illustrative adjustment only; not a validated pricing model.
    age_adjustment = min(vehicle_age * 0.01, 0.15)
    estimated_repair_cost = round(base_estimate * (1 + age_adjustment), 2)

    indicative_payout = round(
        max(0.0, estimated_repair_cost - deductible), 2
    )

    return {
        "detected_damage_types": detected_types,
        "component_estimate_inr": round(component_estimate, 2),
        "estimated_repair_cost_inr": estimated_repair_cost,
        "deductible_inr": deductible,
        "indicative_payout_inr": indicative_payout,
        "vehicle_age_years": vehicle_age,
        "previous_claims_reported": previous_claims,
        "pricing_method": (
            "User-declared estimate with illustrative age adjustment"
            if declared_repair_cost > 0
            else "Demonstration rates with illustrative age adjustment"
        ),
        "notice": (
            "Illustrative estimate only. Actual repair prices, policy terms, "
            "depreciation, exclusions, and claim eligibility are not verified."
        ),
    }