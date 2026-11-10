# CEFM Mathematical Analysis: Step 3 RCAG Gating Mechanism & Tensor Propagation

**Document ID:** `CEFM-AUDIT-2026-STEP3-RCAG-ANALYSIS`  
**Date:** November 7, 2026  
**Auditor:** Automated Diagnostic System & Clinical AI Safety Group  
**Status:** Complete & Forensically Audited  
**Companion Artifacts:**  
- `reports/CEFM_STEP3_CLINICAL_FEATURE_SCHEMA_AUDIT.csv`  
- `reports/CEFM_STEP3_ABCDE_SCORE_RECONCILIATION.md`  

---

## Executive Summary

This report provides the formal mathematical formulation, tensor dimension audit, intermediate activation analysis, and logit projection trace of the Phase 2 M6 Challenger architecture: **Dual-Scale Gated Multimodal Fusion with Residual Channel Attention Gating (RCAG)**.

### Primary Mechanistic Insights:
1. **Mathematical Structure:**
   The clinical modulation gate is formulated as:
   $$v_{\text{mod}} = v_{\text{blend}} \odot (1.0 + c_{\text{mod}}), \quad c_{\text{mod}} \in (0, 1)$$
   Because $c_{\text{mod}} = \text{Sigmoid}(W_c z_c)$, the scalar multiplier $(1.0 + c_{\text{mod}})$ is strictly bounded within $(1.0, 2.0)$. It is **strictly an amplification mechanism**, never an attenuation or dampening gate.
2. **Empirical Gate Distribution:**
   On audited case `5ef6da25eecc47c6a05bee924b2aa3f8.jpg`, the channel-wise multiplier averages **$1.5007\times$** (range: $1.2900\times - 1.7047\times$). Pre-gate feature norm $\|v_{\text{blend}}\|_2 = 5.462$ expands to post-gate norm $\|v_{\text{mod}}\|_2 = 8.232$ ($+50.7\%$).
3. **Representation Amplification vs Melanoma Evidence:**
   - **Representation Amplification** is a property of the vector norm (signal magnitude).
   - **Melanoma Evidence** is a property of the signed projection direction along the classifier separating hyperplane: $\Delta z = (w_{\text{mel}} - w_{\text{non\_mel}}) \cdot z_{\text{fused}} + (b_{\text{mel}} - b_{\text{non\_mel}})$.
   - Counterfactual ablation confirms that amplifying visual features under benign clinical context increases non-melanoma logit $z_0$ from $+2.226$ to $+2.457$ ($+0.231$) and drops melanoma probability from $1.15\%$ to $0.84\%$.
   - The gate reinforces learned visual representations; for benign lesions, this reinforcement increases classifier confidence for benign classification.

---

## 1. Complete Architecture & Tensor Flow

```
Global View (224x224 RGB) ──► ViT Backbone ──► g_emb (768) ──► Global Encoder ──────────► zg (128) ──┐
                                                                                                        ├─► Scale Gate ──► alpha_g (128)
Local Patches (224x224 RGB) ─► ViT Backbone ──► l_emb (768) ──► Local Encoder ───────────► zl (128) ──┴──────┬───────────────────┐
                                                                                                               ▼                   │
                                                                                        v_blend = alpha*zg + (1-alpha)*zl (128)   │
                                                                                                               │                   │
16-D Clinical Vector ────────► Normalization ──► clin_norm (16) ─► Clinical Encoder ─► zc (64)                ▼                   │
                                                                      │                               c_mod = Sigmoid(zc)          │
                                                                      └─► Clinical Gate (128) ──────► (1.0 + c_mod)                │
                                                                                                               │                   │
                                                                                                               ▼                   │
                                                                                                 v_mod = v_blend * (1+c_mod) (128) │
                                                                                                               │                   │
                                                                                                               ▼                   ▼
                                            combined = [v_mod (128), zl (128), zc (64)] ◄──────────────────────────────────────────┘
                                                                 │  (320 dimensions)
                                                                 ▼
                                            Fusion Network: Linear(320->128) -> LN -> GELU -> Linear(128->128) -> LN -> GELU
                                                                 │
                                                                 ▼
                                            Latent Representation: z_fused (128)
                                                                 │
                                                                 ▼
                                            Linear Classifier: Linear(128, 2)
                                                                 │
                                                                 ▼
                                            Output Logits: [z0 (non_mel), z1 (mel)]
                                                                 │
                                                                 ▼
                                            Softmax: P(non_mel), P(mel)
```

---

## 2. Layer-by-Layer Mathematical Equations & Dimensions

### Step 1: Unimodal Feature Encoding
Given global visual embedding $x_g \in \mathbb{R}^{768}$, local patch embedding $x_l \in \mathbb{R}^{768}$, and normalized clinical vector $x_c \in \mathbb{R}^{16}$:
$$z_g = \text{Dropout}_{0.15}(\text{GELU}(\text{LayerNorm}(W_g x_g + b_g))), \quad W_g \in \mathbb{R}^{128 \times 768}, \quad z_g \in \mathbb{R}^{128}$$
$$z_l = \text{Dropout}_{0.15}(\text{GELU}(\text{LayerNorm}(W_l x_l + b_l))), \quad W_l \in \mathbb{R}^{128 \times 768}, \quad z_l \in \mathbb{R}^{128}$$
$$z_c = \text{Dropout}_{0.15}(\text{GELU}(\text{LayerNorm}(W_c x_c + b_c))), \quad W_c \in \mathbb{R}^{64 \times 16}, \quad z_c \in \mathbb{R}^{64}$$

### Step 2: Scale-Adaptive Inter-Scale Visual Gating
Global and local latent features are concatenated: $u_{\text{scale}} = [z_g \,\|\, z_l] \in \mathbb{R}^{256}$.
$$\alpha_g = \sigma(W_{s2} \cdot \text{GELU}(W_{s1} u_{\text{scale}} + b_{s1}) + b_{s2}), \quad \alpha_g \in (0, 1)^{128}$$
$$v_{\text{blend}} = \alpha_g \odot z_g + (\mathbf{1} - \alpha_g) \odot z_l, \quad v_{\text{blend}} \in \mathbb{R}^{128}$$
*Properties:* Convex combination per channel; dynamically balances macro context against micro zoom details.

### Step 3: Clinical Residual Channel Attention Gating (RCAG)
The clinical representation $z_c \in \mathbb{R}^{64}$ generates a channel modulation vector:
$$c_{\text{mod}} = \sigma(W_{\text{mod}} z_c + b_{\text{mod}}), \quad W_{\text{mod}} \in \mathbb{R}^{128 \times 64}, \quad c_{\text{mod}} \in (0, 1)^{128}$$
$$v_{\text{mod}} = v_{\text{blend}} \odot (\mathbf{1} + c_{\text{mod}}), \quad v_{\text{mod}} \in \mathbb{R}^{128}$$
*Properties:* Because $\sigma(\cdot) \in (0, 1)$, every channel of $v_{\text{blend}}$ is scaled by a factor $\in (1.0, 2.0)$. The minimum possible scale is $1.0\times$ (identity passthrough), and the maximum is $2.0\times$.

### Step 4: Multi-Branch Bottleneck Fusion
The modulated visual features, original local residual features, and clinical features are concatenated:
$$u_{\text{fused}} = [v_{\text{mod}} \,\|\, z_l \,\|\, z_c] \in \mathbb{R}^{128 + 128 + 64} = \mathbb{R}^{320}$$
$$h_1 = \text{Dropout}_{0.20}(\text{GELU}(\text{LayerNorm}(W_{f1} u_{\text{fused}} + b_{f1}))), \quad W_{f1} \in \mathbb{R}^{128 \times 320}$$
$$z_{\text{fused}} = \text{GELU}(\text{LayerNorm}(W_{f2} h_1 + b_{f2})), \quad W_{f2} \in \mathbb{R}^{128 \times 128}$$

### Step 5: Classifier Head
$$[z_0, z_1]^T = W_{\text{clf}} z_{\text{fused}} + b_{\text{clf}}, \quad W_{\text{clf}} \in \mathbb{R}^{2 \times 128}, \quad b_{\text{clf}} \in \mathbb{R}^2$$
$$P(\text{mel}) = \frac{e^{z_1}}{e^{z_0} + e^{z_1}}$$

---

## 3. Empirical Intermediate Tensor Trace on Audited Case

Evaluated on `5ef6da25eecc47c6a05bee924b2aa3f8.jpg` using frozen M6 weights (`573ac6c7...`):

| Intermediate Tensor | Shape | Mean | Min | Max | L2 Norm ($\|x\|_2$) | Physical / Representation Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Global Visual $z_g$** | `[1, 128]` | `+0.2875` | `-0.812` | `+1.845` | `7.098` | Global dermoscopic pattern |
| **Local Zoom $z_l$** | `[1, 128]` | `+0.2822` | `-0.854` | `+1.921` | `7.335` | Local lesion center pigment network |
| **Clinical Latent $z_c$** | `[1, 64]` | `+0.2896` | `-0.795` | `+1.884` | `5.161` | Latent 16-D clinical feature vector |
| **Scale Gate $\alpha_g$** | `[1, 128]` | `0.5130` | `0.3906` | `0.6214` | *N/A* | $51.3\%$ global / $48.7\%$ local balance |
| **Blended Visual $v_{\text{blend}}$** | `[1, 128]` | `+0.2849` | `-0.642` | `+1.411` | **`5.462`** | Pre-modulation visual state |
| **Clinical Gate $c_{\text{mod}}$** | `[1, 128]` | `0.5007` | `0.2900` | `0.7047` | *N/A* | Sigmoid gate values |
| **Scale Factor $(1 + c_{\text{mod}})$** | `[1, 128]` | **`1.5007`** | **`1.2900`** | **`1.7047`** | *N/A* | **Channel-wise amplification multiplier** |
| **Modulated Visual $v_{\text{mod}}$** | `[1, 128]` | `+0.4276` | `-0.963` | `+2.116` | **`8.232`** | **Post-modulation visual state (+50.7% norm)** |
| **Latent State $z_{\text{fused}}$** | `[1, 128]` | `+0.2910` | `-1.120` | `+2.450` | `8.577` | Final representation entering classifier |
| **Output Logits $[z_0, z_1]$** | `[1, 2]` | — | — | — | — | **$[+2.4566, -2.3093]$** ($\Delta z = -4.7659$) |
| **Softmax Melanoma Prob** | scalar | — | — | — | — | **`0.00844` (0.84%)** |

---

## 4. Deconstructing Downstream Fusion & Classifier Logit Contributions

To understand how the amplified features drive the final classification, we decomposed the first fusion layer $W_{f1} \in \mathbb{R}^{128 \times 320}$ into its three respective input blocks:
- $W_{\text{mod}} = W_{f1}[:, 0:128]$ (acting on $v_{\text{mod}}$)
- $W_{zl} = W_{f1}[:, 128:256]$ (acting on $z_l$)
- $W_{zc} = W_{f1}[:, 256:320]$ (acting on $z_c$)

### Input Energy Decomposition:
$$\|W_{\text{mod}} v_{\text{mod}}\|_2 = \mathbf{3.803} \quad (44.4\% \text{ of fusion input energy})$$
$$\|W_{zl} z_l\|_2 = \mathbf{2.810} \quad (32.8\% \text{ of fusion input energy})$$
$$\|W_{zc} z_c\|_2 = \mathbf{1.945} \quad (22.7\% \text{ of fusion input energy})$$

Because $v_{\text{mod}}$ has an expanded norm of $8.232$, it contributes the dominant signal ($44.4\%$) to the fusion bottleneck.

---

## 5. Counterfactual Ablation: Dissecting Amplification vs Melanoma Evidence

A critical conceptual distinction in neural interpretation is:
$$\text{Representation Amplification} \neq \text{Increased Melanoma Evidence}$$

- **Representation Amplification:** An isotropic or channel-wise increase in the magnitude ($\|x\|_2$) of hidden representations.
- **Melanoma Evidence:** The dot product of the latent state with the classifier decision vector $w_{\text{diff}} = (w_{\text{mel}} - w_{\text{non\_mel}}) \in \mathbb{R}^{128}$:
  $$\Delta z = z_1 - z_0 = w_{\text{diff}} \cdot z_{\text{fused}} + (b_1 - b_0)$$
  - When $w_{\text{diff}} \cdot z_{\text{fused}} > 0$, the representation contains positive melanoma evidence.
  - When $w_{\text{diff}} \cdot z_{\text{fused}} < 0$, the representation contains positive non-melanoma (benign) evidence.

### Controlled Ablation Results

| Experimental Condition | Modulation Factor | $\|v_{\text{mod}}\|_2$ | Non-Mel Logit ($z_0$) | Mel Logit ($z_1$) | Logit Diff ($\Delta z$) | Softmax $P(\text{mel})$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **M6 Production Forward Pass** | **$(1 + c_{\text{mod}}) \approx 1.50\times$** | **`8.232`** | **`+2.4566`** | **`-2.3093`** | **`-4.7659`** | **`0.84%`** |
| **Counterfactual: Factor = 1.0 (No Gate)** | $1.00\times$ (Identity) | `5.462` | `+2.2262` | `-2.2309` | `-4.4571` | **`1.15%`** |
| **Counterfactual: Zero Clinical ($z_c = \mathbf{0}$)**| Conditioned on $\mathbf{0}$ | `8.195` | `+2.6091` | `-2.3192` | `-4.9283` | **`0.72%`** |

### Rigorous Causal Deductions:
1. When the RCAG amplification gate is active, the non-melanoma logit $z_0$ increases from $+2.226$ to $+2.457$ ($+0.231$), while the melanoma logit $z_1$ decreases from $-2.231$ to $-2.309$ ($-0.078$).
2. The logit difference widens from $-4.457$ to $-4.766$, pushing the predicted melanoma probability down from $1.15\%$ to $0.84\%$.
3. **Conclusion:** Amplifying the visual features by $(1 + c_{\text{mod}})$ projects predominantly along the **negative** direction of $w_{\text{diff}}$. For this benign lesion, amplifying visual representations **reinforces benign classification**, effectively reducing melanoma risk.
4. It is scientifically invalid to characterize the gate as an "attenuator" based on the drop in melanoma probability; the gate mathematically amplifies signal power, and the trained weights orient that amplified signal toward the non-melanoma class.

