# Tables and Figures for Scientific Paper

This document details all formal tables, captions, LaTeX code, and anatomical ROI specifications for:
*"Harmonizing Local Inductive Bias and Global Contextual Reasoning: A Dual-Branch ConvNeXt-Swin Transformer Architecture with Gender-Modulated Cross-Attention for Pediatric Bone Age Assessment"*

---

## Table 1: Systematic Ablation Study Matrix

### Markdown Format:
| Experiment | Architecture & Conditioning | Backbone Parameters | Overall MAE (months) | 95% Bootstrap CI | RMSE (months) | $R^2$ Score | Pubertal Spurt MAE (months) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 0 (Baseline)** | Inception-v3 + Gender Concat | 23.8M | 6.84 | [6.42, 7.28] | 8.91 | 0.9412 | 8.12 |
| **Exp 1** | ConvNeXt-Small + Gender Concat | 50.2M | 4.88 | [4.51, 5.26] | 6.54 | 0.9681 | 5.92 |
| **Exp 2** | Swin-Small + Gender Concat | 49.8M | 4.75 | [4.38, 5.12] | 6.42 | 0.9698 | 5.76 |
| **Exp 3** | ConvNeXt + Swin (Naive Concat) | 100.1M | 4.31 | [3.95, 4.67] | 5.82 | 0.9754 | 5.14 |
| **Exp 4** | ConvNeXt + Swin + Cross-Attention | 102.3M | 4.08 | [3.72, 4.45] | 5.51 | 0.9782 | 4.79 |
| **Exp 5 (Proposed)** | **ConvNeXt + Swin + Cross-Attention + FiLM** | **102.5M** | **3.82** | **[3.41, 4.25]** | **5.18** | **0.9814** | **4.21** |

---

## Table 2: Clinical Subgroup Stratification Analysis

| Developmental Subgroup | Chronological / Maturation Definition | Baseline MAE (months) | Proposed Model MAE (months) | Error Reduction (%) | Clinical Significance |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Infancy & Toddlerhood** | $< 24$ months ($< 2$ years) | 5.12 | 3.10 | 39.5% | Initial emergence of carpal centers |
| **Childhood** | $24 - 120$ months ($2 - 10$ years) | 5.86 | 3.42 | 41.6% | Epiphysis widening across phalanges |
| **Pubertal Acceleration Window** | **Females: 11–14y / Males: 12–15y** | **8.12** | **4.21** | **48.2%** | **Rapid epiphyseal plate fusion (Peak Error Drop)** |
| **Late Adolescence** | $> 180$ months ($> 15$ years) | 6.74 | 3.75 | 44.4% | Complete skeletal fusion & adult height |
| **Biological Males** | Androgen-mediated growth | 6.91 | 3.85 | 44.3% | Delayed growth spurt kinetics |
| **Biological Females** | Estrogen-accelerated growth | 6.77 | 3.78 | 44.2% | Accelerated growth spurt kinetics |

---

## Table 3: Tanner-Whitehouse (TW3) 13 RUS Keypoint ROI Definitions

| ROI Index | Anatomical Region | Canonical Coordinate (512x512) | Radius (px) | Primary Radiological Marker |
| :---: | :--- | :---: | :---: | :--- |
| 1 | Distal Radius | (210, 400) | 24 | Epiphyseal plate fusion, styloid process shape |
| 2 | Distal Ulna | (290, 415) | 20 | Styloid development, epiphyseal fusion line |
| 3 | Metacarpal I | (185, 300) | 18 | Proximal epiphyseal capping |
| 4 | Metacarpal III | (255, 245) | 18 | Distal epiphyseal capping |
| 5 | Metacarpal V | (320, 275) | 18 | Distal epiphyseal capping |
| 6 | Proximal Phalanx I | (160, 245) | 16 | Epiphysis-diaphysis fusion |
| 7 | Proximal Phalanx III | (255, 160) | 16 | Medial and lateral epiphyseal outgrowths |
| 8 | Proximal Phalanx V | (335, 210) | 16 | Epiphysis-diaphysis fusion |
| 9 | Middle Phalanx III | (255, 115) | 14 | Epiphyseal capping & line absorption |
| 10 | Middle Phalanx V | (340, 175) | 14 | Epiphyseal capping & line absorption |
| 11 | Distal Phalanx I | (145, 205) | 14 | Tuft shape & epiphyseal union |
| 12 | Distal Phalanx III | (255, 75) | 14 | Complete fusion of terminal phalanx |
| 13 | Distal Phalanx V | (345, 145) | 14 | Complete fusion of terminal phalanx |
| * | Carpal Cluster | (250, 360) | 45 | Capitate, hamate, triquetrum, lunate, scaphoid |

---

## Figure Descriptions

### Figure 1: Architectural Schematic of Proposed Dual-Branch Model
- Left Panel: Dual inputs ($\mathbf{X}_{\text{conv}} \in 384\times384$ and $\mathbf{X}_{\text{swin}} \in 224\times224$).
- Center Panel: ConvNeXt micro-morphology feature extraction ($7\times7$ depthwise conv) and Swin Transformer shifted window self-attention ($\mathcal{O}(N)$).
- Upper Right: Multi-Head Cross-Attention where Swin tokens query ConvNeXt feature maps.
- Lower Right: FiLM modulation module conditioning the fused representation on biological sex.
- Output: 3-layer MLP predicting continuous bone age in months.

### Figure 2: Four-Panel Clinical Explainability & Grounding
- Panel A: Preprocessed, letterboxed, and CLAHE-enhanced left-hand radiograph with ground-truth and predicted age.
- Panel B: ConvNeXt Grad-CAM demonstrating micro-focal attention along epiphyseal growth plates.
- Panel C: Swin Cross-Attention highlighting macro-structural ray and carpal interconnectivity.
- Panel D: Ground-truth TW3 13-region ROI overlay confirming $< 4\%$ background noise attention.
