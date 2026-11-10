"""
Seed the RAG knowledge base with melanoma / ABCDE clinical knowledge.
Run once:  python scripts/seed_rag.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import rag_documents_collection
from app.rag.embeddings import create_embedding

DOCUMENTS = [
    {
        "source": "ABCDE Rule - Clinical Dermatology Guidelines",
        "text": (
            "The ABCDE rule is a clinical mnemonic used to identify warning signs of melanoma. "
            "A stands for Asymmetry: one half of the lesion does not match the other. "
            "B stands for Border: edges are irregular, ragged, notched, or blurred. "
            "C stands for Color: variation in colour from one area to another, including shades of tan, brown, black, red, white, or blue. "
            "D stands for Diameter: the lesion is larger than 6 mm (about the size of a pencil eraser). "
            "E stands for Evolution: any change in size, shape, colour, or new symptoms such as bleeding, itching, or crusting. "
            "Any lesion that meets one or more of these criteria warrants prompt evaluation by a dermatologist."
        ),
    },
    {
        "source": "Melanoma Risk Factors - American Cancer Society",
        "text": (
            "Risk factors for melanoma include: ultraviolet (UV) light exposure (sun or tanning beds), "
            "fair skin that burns easily, history of severe sunburns, large numbers of moles (>50), "
            "unusual or atypical moles (dysplastic nevi), personal or family history of melanoma, "
            "weakened immune system, and older age. Men are at slightly higher risk than women. "
            "People with red or blond hair and blue or green eyes also carry elevated risk."
        ),
    },
    {
        "source": "Dermoscopy and Lesion Classification - Journal of Investigative Dermatology",
        "text": (
            "Dermoscopy improves clinical diagnosis of melanoma compared to naked-eye examination. "
            "Key dermoscopic features associated with melanoma include atypical pigment network, regression structures, "
            "blue-white veil, irregular streaks, and atypical vascular patterns. "
            "The 7-point checklist assigns scores to major features (atypical network, blue-white veil, atypical vascular pattern) "
            "and minor features (irregular streaks, irregular pigmentation, irregular dots/globules, regression structures). "
            "A score of 3 or more is considered suspicious for melanoma."
        ),
    },
    {
        "source": "Melanoma Staging - AJCC Cancer Staging Manual 8th Edition",
        "text": (
            "Melanoma is staged from 0 to IV. Stage 0 (melanoma in situ) is confined to the epidermis. "
            "Stage I and II are localised to the skin, differentiated by tumour thickness (Breslow depth) and ulceration. "
            "Stage III indicates spread to regional lymph nodes or in-transit metastases. "
            "Stage IV is distant metastatic disease. "
            "The 5-year survival rate is approximately 98% for localised melanoma but drops to 35% for distant metastases. "
            "Early detection is the single most important factor in improving outcomes."
        ),
    },
    {
        "source": "High Melanoma Probability - Clinical Management Guidelines",
        "text": (
            "A high model-estimated melanoma probability (above 60%) combined with multiple elevated ABCDE features "
            "indicates that urgent clinical assessment by a board-certified dermatologist is recommended. "
            "The patient should be advised to avoid UV exposure and not to manipulate the lesion. "
            "A full-body skin examination and dermoscopic evaluation are standard next steps. "
            "If malignancy is suspected after clinical evaluation, excisional biopsy with appropriate margins is the standard of care."
        ),
    },
    {
        "source": "Lesion Evolution and Follow-Up - British Journal of Dermatology",
        "text": (
            "Lesion evolution (the E in ABCDE) is one of the most sensitive indicators of malignant transformation. "
            "Any mole or skin lesion that changes in size, shape, colour, or develops new symptoms "
            "such as bleeding, itching, ulceration, or pain over weeks to months requires urgent evaluation. "
            "Digital dermoscopy follow-up at 3 to 6 month intervals is recommended for high-risk patients with multiple atypical nevi. "
            "Quantitative change detection algorithms can assist in early identification of evolving lesions."
        ),
    },
    {
        "source": "Color Variation in Melanoma - Pathological Basis",
        "text": (
            "Color heterogeneity in a pigmented skin lesion reflects variable depths of melanin deposition and different histological components. "
            "Tan and light brown represent superficial melanin. Dark brown and black suggest dense melanin or melanocyte proliferation. "
            "Red areas indicate inflammation or neo-vascularisation. White areas represent regression or fibrosis. "
            "Blue-grey colour corresponds to deep dermal melanin (blue-white veil in dermoscopy). "
            "The presence of more than two distinct colours within a single lesion increases the probability of melanoma."
        ),
    },
    {
        "source": "Border Irregularity - Diagnostic Significance",
        "text": (
            "A well-defined, regular border is characteristic of benign melanocytic lesions such as common nevi. "
            "Irregular, poorly defined, or notched borders suggest disordered radial growth phase and are a hallmark of melanoma. "
            "Border irregularity quantified computationally (e.g., compactness ratio, fractal dimension) correlates with malignancy grade. "
            "Values above 0.5 on a normalised 0 to 1 irregularity scale are considered clinically significant and warrant dermatological review."
        ),
    },
    {
        "source": "Diameter Threshold - ABCDE Screening Reference",
        "text": (
            "The conventional 6 mm diameter threshold in the ABCDE rule was established because most melanomas at clinical presentation "
            "exceed 6 mm in greatest diameter, while most benign nevi remain smaller. "
            "However, small melanomas (less than 6 mm) do exist, particularly nodular melanoma and amelanotic melanoma. "
            "A lesion below 6 mm but meeting other ABCDE criteria still requires clinical evaluation. "
            "Calibrated digital imaging provides more accurate diameter measurement than naked-eye estimation."
        ),
    },
    {
        "source": "AI-Assisted Melanoma Detection - Systematic Review",
        "text": (
            "Deep learning models trained on large dermoscopic image datasets (e.g., ISIC, HAM10000) have achieved sensitivity and specificity "
            "comparable to board-certified dermatologists in controlled test conditions. "
            "Vision Transformer (ViT) architectures have shown state-of-the-art performance in skin lesion classification tasks. "
            "AI-assisted tools should be used as decision-support instruments alongside clinical expertise, not as replacements. "
            "A model probability above 70% combined with elevated ABCDE scores significantly increases pre-test probability of malignancy."
        ),
    },
]


def seed():
    existing = rag_documents_collection.count_documents({})
    if existing >= len(DOCUMENTS):
        print(f"RAG collection already has {existing} documents. Skipping seed.")
        print("To reseed, drop the 'rag_documents' collection manually and re-run.")
        return

    print(f"Seeding {len(DOCUMENTS)} melanoma knowledge documents into RAG collection...")
    inserted = 0
    for i, doc in enumerate(DOCUMENTS):
        try:
            embedding = create_embedding(doc["text"])
            rag_documents_collection.update_one(
                {"source": doc["source"], "chunk_index": i},
                {"$set": {
                    "source":      doc["source"],
                    "text":        doc["text"],
                    "chunk_index": i,
                    "embedding":   embedding,
                }},
                upsert=True,
            )
            inserted += 1
            print(f"  [{i+1}/{len(DOCUMENTS)}] OK  {doc['source'][:70]}")
        except Exception as e:
            print(f"  [{i+1}/{len(DOCUMENTS)}] FAIL: {e}")

    print(f"\nDone. {inserted}/{len(DOCUMENTS)} documents seeded.")
    print("RAG evidence will now appear in analysis reports.")


if __name__ == "__main__":
    seed()

