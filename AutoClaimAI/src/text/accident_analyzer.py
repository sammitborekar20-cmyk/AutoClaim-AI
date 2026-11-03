import re

PART_KEYWORDS = {
    "bonnet": ["bonnet", "hood"],
    "boot": ["boot", "trunk"],
    "door": ["door", "side door"],
    "fender": ["fender", "wing"],
    "front bumper": ["front bumper", "front bump"],
    "front windscreen": ["front windscreen", "windshield", "windscreen"],
    "headlight": ["headlight", "head lamp"],
    "quarter panel": ["quarter panel", "quaterpanel"],
    "rear bumper": ["rear bumper", "back bumper"],
    "rear windscreen": ["rear windscreen", "rear windshield"],
    "roof": ["roof"],
    "running board": ["running board", "side step"],
    "side mirror": ["side mirror", "side-mirror", "wing mirror"],
    "taillight": ["taillight", "tail light", "rear light"],
}

ACCIDENT_KEYWORDS = {
    "collision": ["collision", "collided", "crashed", "crash", "hit", "struck"],
    "parking incident": ["parking", "reversing", "reverse", "pillar"],
    "weather-related incident": ["rain", "hail", "storm", "flood", "wind"],
    "falling object": ["falling object", "tree branch", "debris"],
}


def analyze_accident_text(description: str) -> dict:
    """Extract simple, explainable keywords from a claim description."""
    text = re.sub(r"\s+", " ", (description or "").lower()).strip()

    if not text:
        return {
            "description": "",
            "reported_parts": [],
            "incident_keywords": [],
            "word_count": 0,
            "summary": "No accident description provided.",
        }

    reported_parts = [
        part
        for part, keywords in PART_KEYWORDS.items()
        if any(keyword in text for keyword in keywords)
    ]

    incident_keywords = [
        incident
        for incident, keywords in ACCIDENT_KEYWORDS.items()
        if any(keyword in text for keyword in keywords)
    ]

    if reported_parts:
        summary = "Reported vehicle parts: " + ", ".join(reported_parts)
    else:
        summary = "No recognized vehicle part was found in the description."

    return {
        "description": description.strip(),
        "reported_parts": reported_parts,
        "incident_keywords": incident_keywords,
        "word_count": len(text.split()),
        "summary": summary,
    }