# Dual-Branch ConvNeXt & Swin Transformer for Pediatric Bone Age Assessment

An end-to-end scientific research codebase and paper reproduction pipeline for pediatric skeletal maturity assessment (RSNA Bone Age benchmark & Tanner-Whitehouse 3 atlas).

## 1. Scientific Motivation & Core Hypothesis

Traditional deep learning solutions for bone age assessment (such as the RSNA 2017 winning Inception-v3 ensemble) rely on standard convolutional filters ($1\times1, 3\times3, 5\times5$). However, their localized receptive fields prevent them from modeling long-range spatial correlations across disjoint anatomical ossification centers (e.g., comparing distal radius/ulna epiphyseal fusion against carpal bone emergence). Conversely, standard Vision Transformers lack the localized inductive biases necessary to delineate micro-morphological indicators like sub-millimeter epiphyseal cartilage plates.

### The Proposed Solution
We propose a **Dual-Branch Hybrid Architecture**:
- **Micro-Morphology Branch (ConvNeXt):** Modernized CNN inductive bias with $7\times7$ depthwise separable convolutions and inverted bottlenecks to capture fine growth plate boundaries.
- **Macro-Structural Branch (Swin Transformer):** Shifted Window Self-Attention with $\mathcal{O}(N)$ complexity to capture long-range inter-bone developmental synchrony (TW3 criteria).
- **Directional Cross-Attention:** Macro query tokens actively interrogate fine-grained micro keys.
- **FiLM Gender Modulation:** Clinical conditioning using biological sex embeddings via Feature-wise Linear Modulation:
  $$y = (1 + \gamma(m)) \odot x + \beta(m)$$
  suppressing shortcut learning and modeling sex-dimorphic maturation kinetics.

---

## 2. Five-Stage Scientific Pipeline

```text
[Stage 1: Data Protocol] ──> [Stage 2: Inception Baseline] ──> [Stage 3: Proposed Model] ──> [Stage 4: Ablation Matrix] ──> [Stage 5: Clinical Explainability]
```

1. **Stage 1: Data Protocol & Leakage-Free Stratification**
   - Joint Stratified 5-Fold Cross-Validation: Stratified simultaneously across age brackets ($<24$m, 2–10y, 11–15y, $>15$y) and biological sex (M/F).
   - Letterbox Square Padding ($512\times512$): Preserves long bone aspect ratio without stretching.
   - Left-Hand Normalization: Automatic orientation check and horizontal flip.
   - CLAHE (Contrast Limited Adaptive Histogram Equalization) for cartilage boundaries.
2. **Stage 2: Fair Baseline Replication**
   - Inception-v3 + Gender Concatenation (RSNA 2017 baseline) evaluated on the identical 5-fold split.
3. **Stage 3: Proposed Architecture Development**
   - ConvNeXt-Small + Swin-Small + Cross-Attention + FiLM with two-stage training (warmup frozen backbones, end-to-end discriminative fine-tuning).
4. **Stage 4: Systematic Ablation Matrix**
   - Exp 0: Inception-v3 + Gender Concat (RSNA 2017 baseline)
   - Exp 1: ConvNeXt-Small Single + Concat (Modern CNN inductive bias)
   - Exp 2: Swin-Small Single + Concat (Hierarchical ViT macro attention)
   - Exp 3: ConvNeXt + Swin (Naive Vector Concat)
   - Exp 4: ConvNeXt + Swin + Cross-Attention Fusion
   - Exp 5 (Proposed): ConvNeXt + Swin + Cross-Attention + FiLM
5. **Stage 5: Clinical Explainability & TW3 Atlas Grounding**
   - ConvNeXt Grad-CAM (growth plate micro-focus).
   - Swin Cross-Attention Saliency (macro-structural ray alignment).
   - TW3 13-region ROI concentration ratio measurement.
   - Publication-ready 4-panel diagnostic figure generation.

---

## 3. Project Directory Structure

```text
dual_branch_bone_age/
├── configs/
│   ├── base_config.py                    # Master configuration dataclass & ablation registry
│   └── ablation_experiments.yaml         # Configuration file
├── src/
│   ├── data/
│   │   ├── dataset.py                    # RSNABoneAgeDataset
│   │   ├── transforms.py                 # Letterbox, CLAHE, Orientation, Dual-Res
│   │   ├── stratified_split.py           # Joint Age-Sex Stratified 5-Fold K-Fold
│   │   └── synthetic_generator.py        # Pediatric X-ray phantom generator
│   ├── models/
│   │   ├── film.py                       # FiLMLayer (Feature-wise Linear Modulation)
│   │   ├── cross_attention.py            # Directional & Bidirectional Cross-Attention
│   │   ├── hybrid_bone_age.py            # DualBranchBoneAgeNet (Proposed Exp 4 & Exp 5)
│   │   ├── baselines.py                  # Exp 0 (Inception), Exp 1 (ConvNeXt), Exp 2 (Swin), Exp 3 (Naive)
│   │   └── model_factory.py              # Unified builder across all ablation conditions
│   ├── engine/
│   │   ├── trainer.py                    # Two-stage unfreezing, AMP, grad accum, Smooth L1
│   │   └── evaluator.py                  # MAE, RMSE, R2, 1000x Bootstrap CI, Pubertal Subgroup
│   ├── explainability/
│   │   ├── gradcam.py                    # Grad-CAM for ConvNeXt micro-branch
│   │   ├── attention_rollout.py          # Cross-Attention saliency for Swin macro-branch
│   │   └── tw3_visualizer.py             # TW3 13-region ROI overlay & 4-panel figure generator
│   └── utils/
│       ├── logger.py                     # Scientific logger
│       └── reproducibility.py            # Deterministic random seed controller
├── experiments/
│   ├── run_stage1_data_protocol.py       # Stage 1 Runner
│   ├── run_stage2_inception_baseline.py  # Stage 2 Runner
│   ├── run_stage3_proposed_train.py      # Stage 3 Runner
│   ├── run_stage4_ablation_matrix.py     # Stage 4 Runner
│   └── run_stage5_clinical_explainability.py # Stage 5 Runner
├── tests/
│   ├── test_models.py                    # Unit tests for models & factory
│   ├── test_data_pipeline.py             # Unit tests for transforms & stratified split
│   └── test_engine.py                    # Unit tests for engine & bootstrap CI
├── paper/
│   ├── manuscript.md                     # Full journal manuscript (IEEE TMI / MedIA format)
│   ├── manuscript.tex                    # Compilable LaTeX manuscript source
│   └── tables_and_figures.md             # Formal tables and TW3 definitions
└── requirements.txt
```

---

## 4. Quickstart Execution Guide

### Step 1: Run Unit Tests
```bash
python tests/test_models.py
python tests/test_data_pipeline.py
python tests/test_engine.py
```

### Step 2: Execute Research Stages
```bash
# Stage 1: Data Protocol & Stratified 5-Fold
python experiments/run_stage1_data_protocol.py

# Stage 2: Fair Inception-v3 Baseline
python experiments/run_stage2_inception_baseline.py

# Stage 3: Train Proposed Dual-Branch Model
python experiments/run_stage3_proposed_train.py

# Stage 4: Run Complete Ablation Matrix
python experiments/run_stage4_ablation_matrix.py

# Stage 5: Generate 4-Panel Clinical Explainability Figures
python experiments/run_stage5_clinical_explainability.py
```
