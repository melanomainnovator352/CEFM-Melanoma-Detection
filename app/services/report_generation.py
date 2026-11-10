def generate_structured_report(
    validation,
    classification=None,
    abcde=None,
    evolution=None,
    explanation=None,
    rag_evidence=None,
):
    """
    Generate the structured melanoma assessment report.

    RAG provides supporting medical reference information.
    It does not change or override the model prediction.
    """

    # =========================================================
    # NORMALIZE INPUTS
    # =========================================================

    validation = validation or {}
    classification = classification or {}
    abcde = abcde or {}
    evolution = evolution or {}
    explanation = explanation or {}
    rag_evidence = rag_evidence or {}

    # =========================================================
    # RAG DOCUMENTS
    # =========================================================

    rag_documents = rag_evidence.get(
        "documents",
        []
    ) or []

    # =========================================================
    # ABCDE DATA
    # =========================================================

    # Current analysis.py uses flattened ABCDE fields.
    # Nested fallback is retained for backward compatibility.

    color = abcde.get(
        "color",
        {}
    ) or {}

    color_index = abcde.get(
        "color_index",
        color.get("index", 0.0)
    )

    diameter = abcde.get(
        "diameter",
        {}
    ) or {}

    diameter_mm = abcde.get(
        "diameter_mm",
        diameter.get("max_diameter_mm")
    )

    diameter_pixels = abcde.get(
        "diameter_pixels",
        diameter.get("max_diameter_pixels")
    )

    diameter_available = abcde.get(
        "diameter_mm_available",
        diameter.get("available", False)
    )

    # =========================================================
    # CLASSIFICATION & MULTI-MODEL DATA
    # =========================================================

    probability = _number(
        classification.get(
            "melanoma_probability"
        ),
        0.0,
    )

    risk = str(
        classification.get(
            "risk_level",
            "not available",
        )
    ).replace("_", " ")

    predicted_class = str(
        classification.get(
            "predicted_class",
            "not available",
        )
    ).replace("_", " ")

    decision_threshold = _number(
        classification.get("decision_threshold"),
        0.80,
    )

    # Multimodal Fusion
    mm_prob = classification.get("multimodal_melanoma_probability")
    mm_pred = str(classification.get("multimodal_predicted_class", "not available")).replace("_", " ")
    mm_thresh = _number(classification.get("multimodal_decision_threshold"), 0.36)

    # Challenger M6 Dual-Scale
    ch_info = classification.get("challenger") or {}
    m6_prob = classification.get("challenger_melanoma_probability", ch_info.get("challenger_melanoma_probability"))
    m6_pred = str(classification.get("challenger_predicted_class", ch_info.get("challenger_predicted_class", "not available"))).replace("_", " ")
    m6_thresh = _number(classification.get("challenger_decision_threshold", ch_info.get("challenger_decision_threshold")), 0.690)
    m6_mode = str(classification.get("challenger_operating_point", ch_info.get("challenger_operating_point", "higher_specificity_review")))

    # Three-Way Discordance & Quality Audit
    discordance = classification.get("three_way_discordance") or classification.get("discordance") or {}
    quality_audit = classification.get("quality_audit") or {}

    # =========================================================
    # DIAMETER
    # =========================================================

    exceeds_6mm = (
        diameter_mm is not None
        and _number(diameter_mm) > 6
    )

    if (
        diameter_available
        and diameter_mm is not None
    ):
        diameter_text = (
            f"The measured maximum diameter is "
            f"{_number(diameter_mm):.1f} mm. "
            f"{'This is above' if exceeds_6mm else 'This is not above'} "
            "the conventional 6 mm reference "
            "used in ABCDE screening."
        )

    elif diameter_pixels is not None:
        diameter_text = (
            f"The lesion measures approximately "
            f"{_number(diameter_pixels):.0f} pixels "
            "in the image. A physical millimetre "
            "measurement was not available because "
            "the calibration reference could not "
            "be validated."
        )

    else:
        diameter_text = (
            "A reliable diameter measurement "
            "was not available."
        )

    # =========================================================
    # IMAGE VALIDATION
    # =========================================================

    validation_text = (
        "The image passed the available "
        "image-quality checks."
    )

    if not validation.get("valid", True):
        validation_text = (
            "The image did not pass all "
            "image-quality checks."
        )

    # =========================================================
    # EVOLUTION
    # =========================================================

    evolution_text = _evolution_text(
        evolution
    )

    # =========================================================
    # MODEL EXPLANATION
    # =========================================================

    summary = explanation.get(
        "summary",
        "",
    )

    # =========================================================
    # RAG MEDICAL EVIDENCE
    # =========================================================

    rag_text = _format_rag_evidence(
        rag_documents
    )

    # =========================================================
    # FULL REPORT
    # =========================================================

    full_report_text = "\n\n".join(
        [
            "CEFM Melanoma Assessment Report",

            "What this report says",

            (
                f"The image analysis estimates a "
                f"{risk} level of concern. The computer "
                f"classified the image as "
                f"'{predicted_class}' with an estimated "
                f"melanoma probability of "
                f"{probability:.0%}. This is an "
                "image-based estimate, not a diagnosis."
            ),

            "Multi-Model Research Assessment",

            (
                f"1. Primary ViT Baseline: '{predicted_class}' "
                f"(Melanoma Probability: {probability:.0%}, Threshold: {decision_threshold:.2f})\n"
                + (
                    f"2. Phase 1 Multimodal Fusion: '{mm_pred}' "
                    f"(Melanoma Probability: {_number(mm_prob):.0%}, Threshold: {mm_thresh:.2f})\n"
                    if mm_prob is not None else "2. Phase 1 Multimodal Fusion: Not evaluated\n"
                )
                + (
                    f"3. Challenger M6 (Dual-Scale RCAG): '{m6_pred}' "
                    f"(Melanoma Probability: {_number(m6_prob):.0%}, Threshold: {m6_thresh:.3f}, Mode: {m6_mode})"
                    if m6_prob is not None else "3. Challenger M6: Not evaluated"
                )
            ),

            "Consensus and Model Agreement",

            (
                str(discordance.get("message"))
                if discordance.get("message")
                else "Evaluated models reflect consistent baseline screening outputs."
            ),

            "ABCDE findings",

            (
                f"A - Asymmetry: "
                f"{_number(abcde.get('asymmetry')):.2f} "
                "on a 0 to 1 scale. Higher values mean "
                "the two sides of the measured lesion "
                "look less alike."
            ),

            (
                f"B - Border: "
                f"{_number(abcde.get('border_irregularity')):.2f} "
                "on a 0 to 1 scale. Higher values mean "
                "the outline appears more irregular."
            ),

            (
                f"C - Color variation: "
                f"{_number(color_index):.2f} "
                "on a 0 to 1 scale. Higher values mean "
                "more variation in the colors inside "
                "the lesion."
            ),

            f"D - Diameter: {diameter_text}",

            f"E - Evolution: {evolution_text}",

            f"Image quality: {validation_text}",

            (
                f"Overall explanation: {summary}"
                if summary
                else
                "Overall explanation: The available "
                "measurements are shown above."
            ),

            # =================================================
            # RAG SECTION
            # =================================================

            "Medical evidence from RAG",

            rag_text,

            # =================================================
            # RECOMMENDATION
            # =================================================

            (
                "What to do next: This result should be "
                "reviewed by a qualified healthcare "
                "professional, especially if the lesion "
                "is changing, bleeding, itching, painful, "
                "or looks different from your other moles."
            ),

            # =================================================
            # DISCLAIMER
            # =================================================

            (
                "Important: This tool supports research "
                "and clinical decision-making. It cannot "
                "confirm or rule out melanoma and should "
                "not replace an in-person medical "
                "examination."
            ),
        ]
    )

    # =========================================================
    # RETURN REPORT
    # =========================================================

    return {
        "validation": validation,
        "classification": classification,
        "abcde": abcde,
        "evolution": evolution,
        "explanation": explanation,

        # RAG information
        "rag_evidence": rag_evidence,

        # Complete report
        "full_report_text": full_report_text,

        "recommendation": (
            "This system is a research "
            "decision-support prototype. "
            "Clinical assessment must be performed "
            "by a qualified professional."
        ),
    }


# =============================================================
# RAG EVIDENCE FORMATTER
# =============================================================

def _format_rag_evidence(rag_documents):
    """
    Format retrieved RAG documents for the report.
    """

    if not rag_documents:
        return (
            "No relevant medical reference was "
            "retrieved from the RAG knowledge base."
        )

    sections = []

    for index, document in enumerate(
        rag_documents,
        start=1,
    ):
        document = document or {}

        source = document.get(
            "source",
            "Medical Reference",
        )

        text = document.get(
            "text",
            "",
        )

        score = _number(
            document.get("score"),
            0.0,
        )

        if not text:
            text = (
                "No text was available for "
                "this retrieved reference."
            )

        section = (
            f"{index}. {source}\n"
            f"Similarity score: {score:.2f}\n"
            f"{text}"
        )

        sections.append(section)

    return "\n\n".join(sections)


# =============================================================
# NUMBER HELPER
# =============================================================

def _number(value, default=0.0):
    """
    Safely convert a value to float.
    """

    try:
        return (
            float(value)
            if value is not None
            else default
        )

    except (TypeError, ValueError):
        return default


# =============================================================
# EVOLUTION TEXT
# =============================================================

def _evolution_text(evolution):
    """
    Generate human-readable evolution information.
    """

    if not evolution.get("available"):
        return evolution.get(
            "reason",
            "There is no previous image "
            "available for comparison.",
        )

    changes = [
        (key, item)
        for key, item in (
            evolution.get("changes") or {}
        ).items()
        if item.get("change_detected")
    ]

    if not changes:
        return (
            "The current image was compared "
            "with the previous uploaded image "
            "in the same case. No meaningful "
            "difference was detected in the "
            "measured features."
        )

    labels = {
        "diameter_mm": "diameter",
        "asymmetry": "asymmetry",
        "border_irregularity": "border shape",
        "color_h": "color hue",
        "color_s": "color saturation",
        "color_v": "color brightness",
        "color_index": "overall color variation",
    }

    changed_labels = ", ".join(
        labels.get(key, key)
        for key, _ in changes
    )

    return (
        "The current image was compared "
        "with the previous uploaded image "
        "in the same case. A meaningful "
        f"difference was detected in: "
        f"{changed_labels}. "

        "Differences can result from real "
        "change, lighting, camera angle, "
        "focus, or calibration, so a "
        "healthcare professional should "
        "review them."
    )