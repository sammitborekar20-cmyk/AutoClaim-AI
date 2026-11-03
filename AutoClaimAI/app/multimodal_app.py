from pathlib import Path
import sys
import json

import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# --------------------------------------------------
# 1. PROJECT SETUP
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.text.accident_analyzer import analyze_accident_text
from src.tabular.claim_analyzer import estimate_claim
from src.documents.policy_extractor import extract_policy_text
from src.fusion.multimodal_fusion import fuse_claim_modalities
from src.rai.rai_assessment import assess_responsible_ai
from src.xai.explainer import generate_occlusion_explanation


MODEL_CANDIDATES = [
    PROJECT_ROOT
    / "runs/detect/runs/damage/autoclaim_detector_test/weights/best.pt",

    PROJECT_ROOT
    / "runs/damage/autoclaim_mps_test/weights/best.pt",

    PROJECT_ROOT / "models/best.pt",
    PROJECT_ROOT / "best.pt",
]


@st.cache_resource
def load_model():
    """Load the trained YOLO model."""

    for model_path in MODEL_CANDIDATES:
        if model_path.is_file():
            return YOLO(str(model_path)), model_path

    return None, None


# --------------------------------------------------
# 2. STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AutoClaim AI | Multimodal RAI + XAI",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 AutoClaim AI")

st.subheader(
    "Multimodal Vehicle Insurance Claim Assessment"
)

st.markdown(
    """
    AutoClaim AI combines image detection, accident-description
    analysis, structured claim information, and optional policy
    documents.

    It also provides Explainable AI (XAI) visualizations and
    Responsible AI (RAI) safeguards.
    """
)

st.warning(
    "Research demonstration only. Results require human verification "
    "and must not automatically determine claim approval, rejection, "
    "liability, or fraud."
)


# --------------------------------------------------
# 3. LOAD YOLO MODEL
# --------------------------------------------------

model, model_path = load_model()

if model is None:
    st.error("YOLO model weights were not found.")

    st.write("Update MODEL_CANDIDATES with your actual best.pt path.")

    for candidate in MODEL_CANDIDATES:
        st.code(str(candidate))

    st.stop()

st.success(f"YOLO model loaded: {model_path}")


# --------------------------------------------------
# 4. IMAGE MODALITY
# --------------------------------------------------

st.divider()
st.header("1. Image Modality")

uploaded_image = st.file_uploader(
    "Upload a vehicle damage image",
    type=["jpg", "jpeg", "png", "webp"],
)


# --------------------------------------------------
# 5. TEXT MODALITY
# --------------------------------------------------

st.header("2. Accident Description")

accident_description = st.text_area(
    "Describe how the accident happened and what damage you observed",
    placeholder=(
        "Example: While reversing, my car hit a concrete pillar, "
        "damaging the front bumper and right headlight."
    ),
    height=120,
)


# --------------------------------------------------
# 6. TABULAR MODALITY
# --------------------------------------------------

st.header("3. Vehicle and Claim Information")

col1, col2 = st.columns(2)

with col1:

    vehicle_make_model = st.text_input(
        "Vehicle make and model",
        value="Hyundai i20",
    )

    vehicle_age = st.number_input(
        "Vehicle age (years)",
        min_value=0,
        max_value=60,
        value=4,
        step=1,
    )

    previous_claims = st.number_input(
        "Previous claims reported",
        min_value=0,
        max_value=100,
        value=1,
        step=1,
    )


with col2:

    deductible = st.number_input(
        "Policy deductible / excess (₹)",
        min_value=0.0,
        max_value=10_000_000.0,
        value=5000.0,
        step=500.0,
    )

    declared_cost = st.number_input(
        "Declared repair estimate (₹; enter 0 if unknown)",
        min_value=0.0,
        max_value=10_000_000.0,
        value=28000.0,
        step=500.0,
    )


# --------------------------------------------------
# 7. DOCUMENT MODALITY
# --------------------------------------------------

st.header("4. Insurance Policy Document (Optional)")

uploaded_pdf = st.file_uploader(
    "Upload an insurance policy PDF",
    type=["pdf"],
    help=(
        "The application extracts selectable PDF text. "
        "Scanned PDFs may require OCR."
    ),
)


# --------------------------------------------------
# 8. XAI SETTINGS
# --------------------------------------------------

st.divider()

run_xai = st.checkbox(
    "Generate XAI ",
    value=False,
    help=(
        "Runs additional YOLO predictions to estimate which "
        "image regions affect the selected detection confidence."
    ),
)


# --------------------------------------------------
# 9. PROCESS CLAIM
# --------------------------------------------------

if st.button(
    "🔎 Assess Claim",
    type="primary",
    use_container_width=True,
):

    if uploaded_image is None:
        st.error("Please upload a vehicle damage image first.")
        st.stop()

    try:

        with st.spinner(
            "Processing image, text, tabular data, and document..."
        ):

            # ==========================================
            # IMAGE PROCESSING
            # ==========================================

            image = Image.open(uploaded_image).convert("RGB")

            prediction = model.predict(
                source=image,
                conf=0.25,
                verbose=False,
            )[0]

            detected_damages = []

            if prediction.boxes is not None:

                for box in prediction.boxes:

                    class_id = int(box.cls[0].item())

                    confidence = float(
                        box.conf[0].item()
                    )

                    damage_type = prediction.names.get(
                        class_id,
                        f"class_{class_id}",
                    )

                    coordinates = [
                        round(float(value), 2)
                        for value in box.xyxy[0].tolist()
                    ]

                    detected_damages.append(
                        {
                            "type": damage_type,
                            "class_id": class_id,
                            "confidence": round(
                                confidence,
                                4,
                            ),
                            "bounding_box_xyxy": coordinates,
                        }
                    )

            # ==========================================
            # TEXT PROCESSING
            # ==========================================

            text_analysis = analyze_accident_text(
                accident_description
            )

            # ==========================================
            # TABULAR CLAIM ANALYSIS
            # ==========================================

            claim_estimate = estimate_claim(
                detected_damages=detected_damages,
                vehicle_age=vehicle_age,
                previous_claims=previous_claims,
                deductible=deductible,
                declared_repair_cost=declared_cost,
            )

            claim_estimate["vehicle_make_model"] = (
                vehicle_make_model
            )

            # ==========================================
            # DOCUMENT PROCESSING
            # ==========================================

            policy_analysis = None

            if uploaded_pdf is not None:

                policy_analysis = extract_policy_text(
                    uploaded_pdf.getvalue()
                )

            # ==========================================
            # MULTIMODAL FUSION
            # ==========================================

            fusion_report = fuse_claim_modalities(
                detected_damages=detected_damages,
                text_analysis=text_analysis,
                claim_estimate=claim_estimate,
                policy_analysis=policy_analysis,
            )

            # ==========================================
            # RESPONSIBLE AI
            # ==========================================

            rai_report = assess_responsible_ai(
                detected_damages=detected_damages,
                text_analysis=text_analysis,
                claim_estimate=claim_estimate,
                policy_analysis=policy_analysis,
            )

            # ==========================================
            # EXPLAINABLE AI
            # ==========================================

            xai_result = None

            if run_xai and detected_damages:

                xai_result = (
                    generate_occlusion_explanation(
                        model=model,
                        image=image,
                        grid_size=3,
                    )
                )

        # --------------------------------------------------
        # 10. ASSESSMENT STATUS
        # --------------------------------------------------

        st.success("Assessment processing completed.")

        st.info(
            "Assessment status: Requires human verification."
        )

        # --------------------------------------------------
        # 11. DISPLAY IMAGE DETECTIONS
        # --------------------------------------------------

        st.divider()
        st.header("5. Vehicle Damage Detection")

        left, right = st.columns(2)

        with left:

            annotated_rgb = prediction.plot()[:, :, ::-1]

            st.image(
                annotated_rgb,
                caption="YOLO11 detections and bounding boxes",
                use_container_width=True,
            )

        with right:

            if detected_damages:

                detection_table = pd.DataFrame(
                    [
                        {
                            "Damage type": item["type"],
                            "Confidence": (
                                f"{item['confidence'] * 100:.1f}%"
                            ),
                            "Bounding box": str(
                                item["bounding_box_xyxy"]
                            ),
                        }
                        for item in detected_damages
                    ]
                )

                st.dataframe(
                    detection_table,
                    use_container_width=True,
                )

            else:

                st.warning(
                    "No damage was detected at the selected "
                    "confidence threshold. This does not prove "
                    "the vehicle is undamaged."
                )

        # --------------------------------------------------
        # 12. DISPLAY TEXT ANALYSIS
        # --------------------------------------------------

        st.divider()
        st.header("6. Accident Description Analysis")

        st.write(text_analysis["summary"])

        incident_keywords = (
            text_analysis["incident_keywords"]
        )

        st.write(
            "**Incident keywords:** "
            + (
                ", ".join(incident_keywords)
                if incident_keywords
                else "None recognized"
            )
        )

        st.write(
            f"**Word count:** {text_analysis['word_count']}"
        )

        # --------------------------------------------------
        # 13. DISPLAY CLAIM ESTIMATE
        # --------------------------------------------------

        st.divider()
        st.header("7. Indicative Claim Estimate")

        cost1, cost2, cost3 = st.columns(3)

        cost1.metric(
            "Estimated repair cost",
            (
                "₹"
                f"{claim_estimate['estimated_repair_cost_inr']:,.2f}"
            ),
        )

        cost2.metric(
            "Deductible",
            (
                "₹"
                f"{claim_estimate['deductible_inr']:,.2f}"
            ),
        )

        cost3.metric(
            "Illustrative amount after deductible",
            (
                "₹"
                f"{claim_estimate['indicative_payout_inr']:,.2f}"
            ),
        )

        st.caption(
            claim_estimate["pricing_method"]
        )

        st.warning(
            claim_estimate["notice"]
        )

        # --------------------------------------------------
        # 14. DISPLAY DOCUMENT EXTRACTION
        # --------------------------------------------------

        st.divider()
        st.header("8. Policy Document Extraction")

        if policy_analysis is not None:

            st.write(
                policy_analysis["message"]
            )

            st.json(
                policy_analysis.get("fields", {})
            )

            if policy_analysis.get("text"):

                with st.expander(
                    "View extracted policy text"
                ):

                    st.text(
                        policy_analysis["text"][:15000]
                    )

        else:

            st.info(
                "No insurance policy document was uploaded."
            )

        # --------------------------------------------------
        # 15. DISPLAY MULTIMODAL FUSION
        # --------------------------------------------------

        st.divider()
        st.header("9. Multimodal Fusion")

        st.write(
            "**Assessment status:**",
            fusion_report["assessment_status"],
        )

        if fusion_report["text_image_matches"]:

            st.subheader(
                "Possible Text-Image Matches"
            )

            st.json(
                fusion_report["text_image_matches"]
            )

        if fusion_report["unmatched_reported_parts"]:

            st.subheader(
                "Reported Parts Not Matched by Image Labels"
            )

            st.write(
                ", ".join(
                    fusion_report[
                        "unmatched_reported_parts"
                    ]
                )
            )

        st.subheader("Consistency and Review Flags")

        for flag in fusion_report["review_flags"]:

            st.warning(flag)

        st.error(
            fusion_report["disclaimer"]
        )

        # --------------------------------------------------
        # 16. DISPLAY XAI
        # --------------------------------------------------

        st.divider()
        st.header("10. Explainable AI (XAI)")

        st.write(
            """
            The bounding boxes and confidence scores show the model's
            predictions. The optional occlusion experiment covers
            image regions and observes how the selected detection's
            confidence changes.
            """
        )

        if not run_xai:

            st.info(
                "XAI heatmap was not requested. Enable the XAI "
                "checkbox and assess the claim again to generate it."
            )

        elif (
            xai_result
            and xai_result.get("success")
        ):

            st.subheader(
                "Occlusion Explanation: "
                + xai_result["class_name"]
            )

            st.write(
                "Baseline confidence: "
                f"{xai_result['baseline_confidence'] * 100:.2f}%"
            )

            st.image(
                xai_result["overlay"],
                caption=(
                    "Occlusion sensitivity heatmap overlay"
                ),
                use_container_width=True,
            )

            st.info(
                xai_result["message"]
            )

            with st.expander(
                "View occlusion importance grid"
            ):

                st.dataframe(
                    pd.DataFrame(
                        xai_result["importance_grid"]
                    ),
                    use_container_width=True,
                )

        elif xai_result:

            st.warning(
                xai_result.get(
                    "message",
                    "XAI explanation unavailable.",
                )
            )

        else:

            st.warning(
                "No detection was available to explain, "
                "so no heatmap was generated."
            )

        # --------------------------------------------------
        # 17. DISPLAY RAI
        # --------------------------------------------------

        st.divider()
        st.header("11. Responsible AI (RAI) Assessment")

        st.warning(
            """
            Human review is required. This application must not
            automatically approve or reject claims or accuse
            claimants of fraud.
            """
        )

        st.subheader(
            "Responsible AI Safeguards"
        )

        for safeguard in rai_report["safeguards"]:

            st.markdown(
                f"- {safeguard}"
            )

        st.subheader(
            "Risk Flags"
        )

        for flag in rai_report["risk_flags"]:

            st.warning(flag)

        status1, status2, status3 = st.columns(3)

        status1.metric(
            "Fairness",
            rai_report["fairness_status"],
        )

        status2.metric(
            "Privacy",
            rai_report["privacy_status"],
        )

        status3.metric(
            "Accountability",
            rai_report["accountability_status"],
        )

        st.subheader(
            "RAI Limitations"
        )

        for limitation in rai_report["limitations"]:

            st.markdown(
                f"- {limitation}"
            )

        # --------------------------------------------------
        # 18. GENERATE DOWNLOADABLE REPORTS
        # --------------------------------------------------

        claim_report = {
            "project": "AutoClaim AI",
            "vehicle_make_model": vehicle_make_model,
            "detected_damages": detected_damages,
            "text_analysis": text_analysis,
            "claim_estimate": claim_estimate,
            "fusion_report": fusion_report,
            "rai_report": rai_report,
            "policy_fields": (
                policy_analysis.get("fields", {})
                if policy_analysis
                else None
            ),
            "policy_extraction_message": (
                policy_analysis.get("message")
                if policy_analysis
                else None
            ),
            "xai_summary": (
                {
                    "success": xai_result.get(
                        "success", False
                    ),
                    "class_name": xai_result.get(
                        "class_name"
                    ),
                    "baseline_confidence": (
                        xai_result.get(
                            "baseline_confidence"
                        )
                    ),
                    "message": xai_result.get(
                        "message"
                    ),
                }
                if xai_result
                else None
            ),
            "human_review_required": True,
        }

        st.divider()
        st.header("12. Download Reports")

        st.download_button(
            label="📥 Download Full Claim Assessment (JSON)",
            data=json.dumps(
                claim_report,
                indent=2,
                ensure_ascii=False,
            ),
            file_name="autoclaim_assessment.json",
            mime="application/json",
        )

        st.download_button(
            label="📥 Download RAI Assessment (JSON)",
            data=json.dumps(
                rai_report,
                indent=2,
                ensure_ascii=False,
            ),
            file_name="autoclaim_rai_assessment.json",
            mime="application/json",
        )

    except Exception as exc:

        st.error(
            "An error occurred while processing the assessment."
        )

        st.exception(exc)