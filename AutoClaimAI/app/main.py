
import streamlit as st
from pathlib import Path
from datetime import date

st.set_page_config(
    page_title="AutoClaim AI",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 AutoClaim AI")
st.subheader("Multimodal Vehicle Insurance Claim Assessment")
st.caption(
    "AI-assisted preliminary assessment for human review. "
    "This prototype does not approve or reject claims."
)

st.divider()

st.header("1. Claim Information")

col1, col2 = st.columns(2)

with col1:
    claim_id = st.text_input("Claim ID", placeholder="CLM-1001")
    claimant_name = st.text_input("Claimant Name")
    vehicle_number = st.text_input("Vehicle Registration Number")

with col2:
    vehicle_make = st.text_input("Vehicle Make and Model")
    accident_date = st.date_input(
        "Accident Date",
        value=date.today(),
        max_value=date.today()
    )
    claim_amount = st.number_input(
        "Claimed Amount (₹)",
        min_value=0,
        step=1000
    )

st.divider()

st.header("2. Accident Description")

accident_description = st.text_area(
    "Describe how the accident happened",
    placeholder=(
        "Example: The vehicle was involved in a "
        "front-side collision at an intersection..."
    ),
    height=140
)

st.divider()

st.header("3. Upload Vehicle Images")

images = st.file_uploader(
    "Upload accident or damage photographs",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if images:
    cols = st.columns(min(len(images), 3))
    for i, uploaded_image in enumerate(images):
        with cols[i % len(cols)]:
            st.image(
                uploaded_image,
                caption=uploaded_image.name,
                use_container_width=True
            )

st.divider()

st.header("4. Upload Claim Documents")

documents = st.file_uploader(
    "Upload policy, registration or repair-estimate documents",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key="documents"
)

if documents:
    st.write(f"{len(documents)} document(s) uploaded.")

st.divider()

st.header("5. Preliminary Claim Intake")

if st.button("Submit Claim for Assessment", type="primary"):
    if not claim_id.strip():
        st.error("Please enter a Claim ID.")
    elif not accident_description.strip():
        st.error("Please provide an accident description.")
    elif not images:
        st.error("Please upload at least one vehicle image.")
    else:
        st.success("Claim information captured in this session.")

        st.info(
            "AI assessment modules are not connected yet. "
            "No damage classification, cost prediction, "
            "fraud assessment or claim decision has been made."
        )

        st.subheader("Submitted Claim Summary")
        st.write({
            "Claim ID": claim_id,
            "Claimant": claimant_name,
            "Vehicle": vehicle_make,
            "Registration": vehicle_number,
            "Accident Date": str(accident_date),
            "Claimed Amount (₹)": claim_amount,
            "Images": len(images),
            "Documents": len(documents or []),
            "Assessment Status": "Awaiting AI module integration"
        })

st.caption(
    "Prototype only. Do not upload real personal or "
    "insurance information until secure storage, "
    "access controls and privacy safeguards are implemented."
)