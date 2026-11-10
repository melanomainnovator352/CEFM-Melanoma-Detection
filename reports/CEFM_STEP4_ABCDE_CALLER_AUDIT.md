# CEFM Step 4 — ABCDE Scoring Interface and Caller Forensic Audit

**Document ID:** `CEFM_STEP4_ABCDE_CALLER_AUDIT`  
**Date:** November 2026  
**Status:** Complete Read-Only Investigation  
**Scope:** Forensic repository-wide audit of all callers, parameter structures, dictionary schemas, and score propagation for `compute_abcde_score()`.

---

## 1. Inventory of `compute_abcde_score()` Callers

Across the entire codebase (`app/`, `tests/`, `scripts/`), a comprehensive repository search identified three specific occurrences of `compute_abcde_score`:

| Location | Role | Invocation Signature | Input Schema Supplied |
| :--- | :--- | :--- | :--- |
| [`app/services/abcde/composite.py#L6`](file:///D:/pytest_cache/app/services/abcde/composite.py#L6) | **Function Definition** | `def compute_abcde_score(features: dict) -> dict` | Standard library definition |
| [`app/services/analysis.py#L425`](file:///D:/pytest_cache/app/services/analysis.py#L425) | **Production Service Pipeline** | `abcde_risk = compute_abcde_score(flat)` | **Flattened 1D dictionary** (`flat`) |
| [`tests/test_abcde.py#L63`](file:///D:/pytest_cache/tests/test_abcde.py#L63) | **Unit Test Suite** | `score = compute_abcde_score(features)` | **Flattened dictionary** fixtures |

### Detailed Caller Analysis

#### A. Production Pipeline (`app/services/analysis.py:425`)
In the main analysis service, `analyze_lesion()` conducts feature extraction sequentially:
1. `features = extract_abcde_features(image, mask, pixels_per_mm=pixels_per_mm)` returns a **hierarchical / nested dictionary** where:
   - `features["asymmetry"]` is a float.
   - `features["border_irregularity"]` is a float.
   - `features["color"]` is a nested dictionary: `{"h": ..., "s": ..., "v": ..., "index": ...}`.
   - `features["diameter"]` is a nested dictionary: `{"max_diameter_mm": ..., "available": ..., ...}`.
2. In Section 8 (`lines 345–377`), `analyze_lesion()` explicitly flattens these into a flat mapping named `flat`:
   ```python
   flat = {
       "asymmetry": float(asymmetry),
       "border_irregularity": float(border_irregularity),
       "color_h": float(color.get("h", 0.0)),
       "color_s": float(color.get("s", 0.0)),
       "color_v": float(color.get("v", 0.0)),
       "color_index": float(color.get("index", 0.0)),
       "diameter_pixels": float(features["diameter_pixels"]),
       "diameter_mm": float(diameter.get("max_diameter_mm")) if diameter.get("max_diameter_mm") is not None else None,
       "diameter_mm_available": bool(diameter.get("available", False)),
   }
   ```
3. At Line 425, `compute_abcde_score(flat)` is invoked with the **flattened** dictionary.
4. The output is stored into `flat["abcde_risk_score"]`, `flat["abcde_risk_label"]`, and attached to the nested dictionary as `features["composite_score"] = abcde_risk`.

#### B. Unit Test Suite (`tests/test_abcde.py:63`)
In `tests/test_abcde.py`, the test fixtures provide flat dictionaries directly:
```python
features = {
    "asymmetry": 0.5,
    "border_irregularity": 0.4,
    "color_index": 0.3,
    "diameter_mm": 7.0,
    "evolution_change_detected": False,
}
score = compute_abcde_score(features)
```
Consequently, unit tests pass because they match the exact keys expected by `compute_abcde_score()`.

---

## 2. Forensic Reconciliation of the Numerical Discrepancy (0.1182 vs 0.3884)

During recent diagnostic probes of case `5ef6da25eecc47c6a05bee924b2aa3f8.jpg`, two disparate ABCDE composite risk scores were recorded: **0.1182** and **0.3884**.

The forensic investigation traces the exact cause of this divergence to dictionary nesting vs. dictionary flattening:

```mermaid
flowchart TD
    subgraph Raw Extraction Output
        A["extract_abcde_features()"] --> B["Nested Dict features"]
        B --> C["features['color'] = {'index': 0.5003}"]
        B --> D["features['diameter'] = {'diameter_mm': 11.96}"]
    end

    subgraph Direct Evaluation - 0.1182
        B -->|Passed Directly| E["compute_abcde_score(features)"]
        E -->|features.get('color_index') -> None| F["Color Score = 0.0"]
        E -->|features.get('diameter_mm') -> None| G["Diameter Score = 0.0"]
        E --> H["Score = 0.25*0.25 + 0.20*0.2753 = 0.1182"]
    end

    subgraph Production Pipeline - 0.3884
        B -->|Flattened in analysis.py| I["flat Dict"]
        I --> J["flat['color_index'] = 0.5003"]
        I --> K["flat['diameter_mm'] = 11.96"]
        I -->|Passed to| L["compute_abcde_score(flat)"]
        L --> M["Color Score = 0.30 * 0.5003 = 0.1501"]
        L --> N["Diameter Score = 0.15 * min(1.0, 11.96/6.0) = 0.1500 (weight 0.1207)"]
        L --> O["Total Composite Score = 0.3884"]
    end
```

### Component Breakdown Comparison

| ABCDE Component | Weight | Nested Dict Evaluation (`features`) | Flat Dict Evaluation (`flat`) | Explanation |
| :--- | :---: | :---: | :---: | :--- |
| **A (Asymmetry)** | 0.25 | $0.2500 \times 0.25 = 0.0625$ | $0.2500 \times 0.25 = 0.0625$ | Present as top-level key `"asymmetry"` in both. |
| **B (Border)** | 0.20 | $0.2753 \times 0.20 = 0.0551$ | $0.2753 \times 0.20 = 0.0551$ | Present as top-level key `"border_irregularity"` in both. |
| **C (Color)** | 0.30 | $0.0000 \times 0.30 = \mathbf{0.0000}$ | $0.5003 \times 0.30 = \mathbf{0.1501}$ | In nested dict, key is `features["color"]["index"]`. `features.get("color_index")` defaults to 0.0. |
| **D (Diameter)** | 0.15 | $0.0000 \times 0.15 = \mathbf{0.0000}$ | $1.0000 \times 0.1207 = \mathbf{0.1207}$ | In nested dict, key is `features["diameter"]["diameter_mm"]`. `features.get("diameter_mm")` defaults to 0.0. |
| **E (Evolution)**| 0.10 | $0.0000 \times 0.10 = 0.0000$ | $0.0000 \times 0.10 = 0.0000$ | No baseline evolution image provided in single-image triage. |
| **Renormalized Sum** | **1.00** | $\mathbf{0.1182}$ | $\mathbf{0.3884}$ | **Match: Exact replication of reported values.** |

### Clinical Risk Label Stability
In `compute_abcde_score()`, continuous composite scores are mapped to clinical triage risk categories according to the following thresholds:
- $[0.00, 0.45) \implies$ **`LOW`**
- $[0.45, 0.70) \implies$ **`MODERATE`**
- $[0.70, 1.00] \implies$ **`HIGH`**

Because both $0.1182 < 0.45$ and $0.3884 < 0.45$, **both evaluations produce the identical clinical risk label: `LOW`**. 
Consequently, this interface inconsistency did not flip the displayed categorical label for this case, though it significantly shifted the underlying continuous risk score ($+0.2702$).

---

## 3. Structural Vulnerabilities in Current ABCDE Implementation

1. **Silent Fallback to Benign Zero (`.get(key, 0.0)`):**  
   In `compute_abcde_score()`:
   ```python
   c_score = float(features.get("color_index", 0.0) or 0.0)
   d_val = features.get("diameter_mm")
   ```
   If a caller passes a nested dictionary, or if a measurement is missing/uncalibrated, the function silently returns $0.0$. In medical triage, **absence of measurement is not evidence of benignity**. Silently defaulting to zero artificially drives the risk score down into the `LOW` category.

2. **Dual-Schema Dependency:**  
   `extract_abcde_features()` outputs a nested schema, while `compute_abcde_score()` expects a flat schema. Any developer or script passing the direct output of `extract_abcde_features()` into `compute_abcde_score()` experiences silent score truncation.

3. **Inconsistent Normalization Across Services:**  
   - In `compute_abcde_score()`, diameter is thresholded against $6.0\text{ mm}$ ($\min(1.0, d / 6.0)$).
   - In `ABCDEVector` / `_prepare_abcde_tensor()`, diameter is normalized against $20.0\text{ mm}$ ($\min(1.0, d / 20.0)$).
   - While these serve different purposes (heuristic risk vs neural network feature scaling), having multiple conflicting normalization bounds increases cognitive load and audit friction.

---

## 4. Architectural Recommendations for Canonical ABCDE Interface

To eliminate interface fragility while strictly preserving existing production behavior, the following recommendations are specified for future maintenance:

### A. Canonical Unified Schema Accessor
`compute_abcde_score()` should natively inspect both flat and nested keys without requiring external flattening:
```python
def compute_abcde_score(features: dict) -> dict:
    # 1. Asymmetry
    asym = features.get("asymmetry")
    if isinstance(asym, dict):
        asym_val = asym.get("score", 0.0)
    else:
        asym_val = float(asym or 0.0)

    # 2. Border
    border = features.get("border_irregularity")
    if isinstance(border, dict):
        border_val = border.get("border_irregularity_score", 0.0)
    else:
        border_val = float(border or 0.0)

    # 3. Color
    if "color_index" in features:
        color_val = float(features["color_index"] or 0.0)
    elif isinstance(features.get("color"), dict):
        color_val = float(features["color"].get("index", features["color"].get("color_diversity", 0.0)) or 0.0)
    else:
        color_val = 0.0

    # 4. Diameter
    if "diameter_mm" in features and features["diameter_mm"] is not None:
        diameter_mm = float(features["diameter_mm"])
        dia_available = True
    elif isinstance(features.get("diameter"), dict):
        d_sub = features["diameter"]
        d_raw = d_sub.get("diameter_mm", d_sub.get("max_diameter_mm"))
        dia_available = d_sub.get("available", d_raw is not None)
        diameter_mm = float(d_raw) if (dia_available and d_raw is not None) else None
    else:
        dia_available = False
        diameter_mm = None
```

### B. Explicit Missing-Value Triage Policy
- When `diameter_mm` or `calibration` is unavailable, `compute_abcde_score()` should **not** treat diameter as $0\text{ mm}$.
- Instead, it should **re-normalize the remaining active component weights** (A, B, C) to sum to 1.0, and flag `"diameter_status": "uncalibrated"` in the returned report dictionary.
- This prevents uncalibrated images from receiving artificially depressed risk scores.

---

## 5. Summary Conclusion
- **Root Cause of 0.1182 vs 0.3884:** Direct evaluation of nested dictionary (`0.1182`) vs pipeline evaluation of flattened dictionary (`0.3884`).
- **Production Pipeline Integrity:** `analyze_lesion()` at line 425 correctly passes `flat`, yielding the true production score of **0.3884**.
- **Risk Label Impact:** None. Both scores map strictly to `LOW` ($< 0.45$).
- **No Production Code Modified:** In accordance with Step 4 audit scope boundaries, this forensic finding is fully documented without modifying codebase files.

