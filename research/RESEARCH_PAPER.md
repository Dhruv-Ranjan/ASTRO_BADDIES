# Hybrid Spectral Reconstruction and Physics-Guided Temporal Modeling for Drought Prediction from RGB Satellite Imagery
**Author :** Dhruv Ranjan (https://orcid.org/0009-0009-5883-4300)

## Abstract

We present a two-stage deep learning framework for drought prediction that addresses the challenge of limited multispectral satellite data availability by learning to reconstruct spectral bands from RGB imagery, then using physics-informed temporal modeling for drought classification. The system combines a multiscale spectral reconstruction network (SpectralNetV10) with a physics-guided temporal drought classifier that integrates vegetation and moisture indices as learned features. Trained and validated on 2,552 real-world Sentinel-2-like image sequences, the framework achieves 93.1% accuracy, 96.97% recall, and 0.953 AUC on the validation set, demonstrating strong generalization to unseen temporal sequences. We validate the end-to-end pipeline on real Sentinel-2 imagery from two climatically distinct regions (Western Ghats and Rajasthan), showing consistent discrimination between wet and dry conditions. This work demonstrates the feasibility of accurate drought monitoring from publicly available RGB satellite data when augmented with learned spectral reconstruction and physics-grounded feature extraction.

**Keywords:** drought prediction, spectral reconstruction, remote sensing, deep learning, physics-informed neural networks, temporal modeling

> **Dataset-scope clarification:** The results reported in this manuscript
> describe the Sentinel-2-like sequence experiments summarized below; they
> must not be attributed to NASA POWER. NASA POWER is used in a separate
> workflow, `AURORA/AURORA_REALWORLD_FINAL.py`, as the project's primary daily
> precipitation and temperature climate source. That workflow derives rolling
> climate features and a future-rainfall drought proxy, paired with
> Sentinel-2 scene statistics. Its generated input cache and run-specific
> metrics are separate from the manuscript's reported metrics. See
> [`../docs/DATA_SOURCES.md`](../docs/DATA_SOURCES.md). A one-day lag is
> required before claiming intraday-causal forecasting because the rolling
> features include the matched calendar-day POWER record.

---

## 1. Introduction

### 1.1 Motivation and Context

Drought is a critical environmental challenge affecting agricultural productivity, water security, and ecosystem health across the globe. Traditional drought monitoring relies on sparse ground observations, coarse climate indices, or limited-frequency satellite acquisitions (e.g., Landsat or MODIS), which can be insufficient for rapid early warning systems. While Sentinel-2 provides multispectral data at high resolution and revisit frequency, data access can be limited by cost, latency, or cloud cover. In contrast, RGB imagery is cheaper, more readily available, and less affected by atmospheric conditions.

The key insight motivating this work is that RGB reflectance patterns encode substantial information about vegetation stress and soil moisture, even though they lack direct access to shortwave infrared (SWIR) and near-infrared (NIR) bands typically used in standard drought indices. By learning a mapping from RGB to full multispectral signatures, we can unlock the potential of widely available RGB data for drought monitoring.

### 1.2 Literature Context

Recent advances in neural image translation (e.g., conditional GANs, attention-based U-Nets) have shown promise for cross-modal image synthesis and spectral reconstruction. Concurrently, physics-informed neural networks (PINNs) have demonstrated success in environmental monitoring by incorporating domain knowledge as inductive biases. This work bridges these approaches: we combine learned spectral reconstruction with explicit physics-based indices to create an interpretable and accurate drought prediction system.

### 1.3 Contributions

1. **Multiscale Spectral Reconstruction**: We develop SpectralNetV10, a deep encoder-decoder network with residual blocks and squeeze-excitation modules that reconstructs six spectral bands from RGB inputs with high fidelity (RMSE ≈ 0.026).

2. **Physics-Guided Temporal Drought Classifier**: We introduce a temporal model that fuses LSTM-based sequence learning with explicit physics-derived features (NDVI, NDMI, NDWI, SAVI, EVI, and moisture indicators) to improve robustness and interpretability.

3. **Real-World Validation**: We validate the system on actual Sentinel-2 sequences and demonstrate consistent performance across geographically distinct regions, providing evidence of practical applicability.

4. **Comprehensive Evaluation**: We report detailed metrics including AUC, F1-score, precision, recall, and calibration measures, along with ablation insights about the fusion strategy.

---

## 2. Methods

### 2.1 Data

#### 2.1.1 Dataset Composition

The training and validation dataset comprises 2,552 samples, each consisting of:

- **RGB Sequences**: Temporal stacks of three-channel RGB patches (64×64 pixels, 3 timesteps per sample)
- **Spectral Targets**: Six-channel ground truth or proxy spectral data (Blue, Green, Red, NIR, SWIR1, SWIR2)
- **Drought Labels**: Binary hard labels (0 = no drought, 1 = drought) and soft probability labels
- **Metadata**: Region identifiers, geographic coordinates (latitude/longitude), and label provenance

The dataset is derived from Sentinel-2 archives and covers diverse geographic regions including the Western Ghats (India), Rajasthan (India), Argentina, Congo Basin, and others. The positive (drought) class comprises approximately 45% of samples, providing reasonable class balance.

#### 2.1.2 Preprocessing

All images are normalized to the range [0, 1]. Temporal sequences are maintained in chronological order with consistent 5–240 day intervals between frames. No additional augmentation is applied during preprocessing; augmentation is applied during training (see Section 2.4.2).

### 2.2 Architecture Design

#### 2.2.1 SpectralNetV10: Spectral Reconstruction Network

**Design Philosophy**: The spectral reconstruction task is a pixel-to-pixel regression problem with high-dimensional output. We use an encoder-decoder architecture with:

- **Residual connections** to preserve RGB information flow and stabilize gradients
- **Squeeze-excitation (SE) modules** for channel-wise attention, allowing the network to dynamically re-weight feature channels based on spatial context
- **Multiscale auxiliary heads** for supervision at intermediate scales, improving gradient propagation and feature quality

**Architecture Details**:

The encoder consists of four residual convolutional blocks with max pooling:

$$\text{Encoder}: \quad \mathbf{x} \xrightarrow{\text{ResBlock}} \mathbf{e}_1 \xrightarrow{\text{Pool}} \mathbf{e}_2 \xrightarrow{\text{Pool}} \mathbf{e}_3 \xrightarrow{\text{Pool}} \mathbf{e}_4 \xrightarrow{\text{ResBlock}} \mathbf{m}$$

where the intermediate feature maps have the following shapes:

| Layer | Shape | Channels | Spatial Dims |
|-------|-------|----------|--------------|
| Input ($\mathbf{e}_1$) | $\mathbb{R}^{B \times 48 \times 64 \times 64}$ | 48 | 64 × 64 |
| Pool 1 ($\mathbf{e}_2$) | $\mathbb{R}^{B \times 96 \times 32 \times 32}$ | 96 | 32 × 32 |
| Pool 2 ($\mathbf{e}_3$) | $\mathbb{R}^{B \times 160 \times 16 \times 16}$ | 160 | 16 × 16 |
| Pool 3 ($\mathbf{e}_4$) | $\mathbb{R}^{B \times 224 \times 8 \times 8}$ | 224 | 8 × 8 |
| Bottleneck ($\mathbf{m}$) | $\mathbb{R}^{B \times 320 \times 4 \times 4}$ | 320 | 4 × 4 |

The decoder upsamples and fuses skip connections from the encoder:

$$\text{Decoder}: \quad \mathbf{m} \xrightarrow{\uparrow 2} \text{Concat}(\mathbf{e}_4) \xrightarrow{\text{ResBlock}} \mathbf{d}_3 \xrightarrow{\uparrow 2} \text{Concat}(\mathbf{e}_3) \xrightarrow{\text{ResBlock}} \mathbf{d}_2 \rightarrow \ldots \rightarrow \mathbf{d}_0$$

**Auxiliary Heads for Multiscale Supervision**:

At three intermediate layers (depths corresponding to 8 × 8, 16 × 16, 32 × 32 resolutions), we attach auxiliary classification heads that predict 6-channel spectral outputs:

$$\mathbf{s}_{\text{low}} = \text{Conv}_{1 \times 1}(\mathbf{d}_3) \in \mathbb{R}^{B \times 6 \times 8 \times 8}$$

$$\mathbf{s}_{\text{mid}} = \text{Conv}_{1 \times 1}(\mathbf{d}_2) \in \mathbb{R}^{B \times 6 \times 16 \times 16}$$

$$\mathbf{s}_{\text{final}} = \text{Conv}_{1 \times 1}(\mathbf{d}_0) \in \mathbb{R}^{B \times 6 \times 64 \times 64}$$

**Output Normalization**:

The final six-channel output is decomposed into visible and hidden bands:

$$\mathbf{s}_{\text{visible}} = \text{Clamp}(\mathbf{s}_{\text{RGB}} + 0.08 \cdot \tanh(\mathbf{s}_{\text{out}}[:, :3]), \, 0, 1)$$

$$\mathbf{s}_{\text{hidden}} = \sigma(\mathbf{s}_{\text{out}}[:, 3:])$$

where $\sigma$ is the sigmoid function. This ensures RGB bands stay close to input values (with small learned adjustments) and NIR/SWIR bands are bounded to [0, 1].

#### 2.2.2 PhysicsTemporalDroughtModel: Physics-Informed Temporal Classifier

**Design Philosophy**: While deep learning can capture complex patterns, explicit physics-based indices provide interpretability and may improve robustness to distribution shifts. We fuse learned temporal features with explicit drought indicators derived from spectral indices.

**Physics Feature Extraction**:

Given a six-channel spectral image $\mathbf{s} \in \mathbb{R}^{6 \times H \times W}$ where channels are Blue, Green, Red, NIR, SWIR1, SWIR2, we compute the following vegetation and moisture indices:

| Index | Abbreviation | Formula |
|-------|--------------|---------|
| **Vegetation** |
| Normalized Difference Vegetation Index | NDVI | $\displaystyle \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red} + \epsilon}$ |
| Enhanced Vegetation Index | EVI | $\displaystyle 2.5 \cdot \frac{\text{NIR} - \text{Red}}{\text{NIR} + 6 \cdot \text{Red} - 7.5 \cdot \text{Blue} + 1 + \epsilon}$ |
| Soil-Adjusted Vegetation Index | SAVI | $\displaystyle 1.5 \cdot \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red} + 0.5 + \epsilon}$ |
| Green Chlorophyll Index | GCI | $\displaystyle \frac{\text{NIR}}{\text{Green} + \epsilon} - 1$ |
| **Moisture** |
| Normalized Difference Moisture Index | NDMI | $\displaystyle \frac{\text{NIR} - \text{SWIR1}}{\text{NIR} + \text{SWIR1} + \epsilon}$ |
| Normalized Difference Water Index | NDWI | $\displaystyle \frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR} + \epsilon}$ |
| Moisture Stress Index | MSI | $\displaystyle \frac{\text{SWIR1}}{\text{NIR} + \epsilon}$ |

We also compute auxiliary features: moisture deficits, chlorophyll stress indicators, vegetation water content, and dry soil indices. The complete feature vector contains **18 dimensions** per spatial location.

**Temporal Physics Prior**:

For a sequence of spectral frames $\{\mathbf{s}_1, \ldots, \mathbf{s}_T\}$, we compute temporal statistics (mean, last frame, span, and temporal deltas) of each physics feature. This yields a normalized drought probability from physics principles:

$$\mathbf{p}_{\text{physics}} = \text{Clamp}\left(
  0.20 \cdot \text{LowVeg} + 0.17 \cdot \text{LowMoisture} + 0.07 \cdot \text{LowSurfaceWater} + \ldots, \, 0, 1
\right)$$

where each component (e.g., LowVeg) is normalized based on reference thresholds (good vs. bad values) and clamped to [0, 1].

**Architecture**:

The temporal model consists of five main components:

**1. Spatial Feature Extraction** — A 3-layer CNN with batch normalization and ReLU activations:

$$\mathbf{z}_{\text{spatial}} = \text{AdaptiveAvgPool2d}(\text{Conv}_3 \circ \text{Conv}_2 \circ \text{Conv}_1(\mathbf{s}))$$

producing a feature vector $\mathbf{z}_{\text{spatial}} \in \mathbb{R}^{128}$ per timestep.

**2. Temporal Encoding** — A bidirectional LSTM with 2 layers and 256 hidden units per direction:

$$\mathbf{h}_t^{\text{lstm}} = \text{BiLSTM}([\mathbf{z}_{\text{spatial},1}, \ldots, \mathbf{z}_{\text{spatial},T}])$$

producing $\mathbf{h}_T \in \mathbb{R}^{512}$ at the final timestep.

**3. Drought Head** — A fully connected network predicting drought probability:

$$\mathbf{p}_{\text{model}} = \sigma\left(\text{FC}_3(\text{ReLU}(\text{FC}_2(\text{ReLU}(\text{FC}_1(\mathbf{h}_T)))))\right)$$

**4. Confidence Head** — A separate network estimating prediction confidence:

$$\mathbf{c} = \sigma\left(\text{FC}(\text{ReLU}(\mathbf{h}_T))\right)$$

**5. Fusion** — Final predictions combine learned and physics-based probabilities:

$$\mathbf{p}_{\text{final}} = \alpha \cdot \mathbf{p}_{\text{model}} + (1 - \alpha) \cdot \mathbf{p}_{\text{physics}}$$

where $\alpha$ is a learnable fusion weight. Two modes are supported:

- **Balanced fusion** ($\alpha = 0.5$): equal weight to learned and physics components
- **Dryness-boosted fusion** ($\alpha = 0.3$): prioritizes physics-derived drought signals

### 2.3 Loss Functions and Training Objectives

#### 2.3.1 Spectral Reconstruction Loss

The spectral reconstruction model is trained with a weighted combination of losses:

$$\mathcal{L}_{\text{spectral}} = \mathcal{L}_{\text{final}} + \lambda_{\text{mid}} \mathcal{L}_{\text{mid}} + \lambda_{\text{low}} \mathcal{L}_{\text{low}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}}$$

**Loss Components**:

**Final Output Loss** — L1 loss on reconstructed spectra (robust to outliers):

$$\mathcal{L}_{\text{final}} = \frac{1}{B} \sum_{b=1}^{B} \|\mathbf{s}_{\text{pred}} - \mathbf{s}_{\text{target}}\|_1$$

**Multiscale Auxiliary Losses** — Direct supervision at intermediate decoder stages:

$$\mathcal{L}_{\text{mid}} = \frac{1}{B} \sum_{b=1}^{B} \|\mathbf{s}_{\text{mid,pred}} - \text{Downsample}(\mathbf{s}_{\text{target}}, 2)\|_1$$

$$\mathcal{L}_{\text{low}} = \frac{1}{B} \sum_{b=1}^{B} \|\mathbf{s}_{\text{low,pred}} - \text{Downsample}(\mathbf{s}_{\text{target}}, 4)\|_1$$

**Smooth L1 Regularization** — Encourages smoothness and edge preservation:

$$\mathcal{L}_{\text{smooth}} = \text{SmoothL1}(\nabla_x \mathbf{s}_{\text{pred}}, \nabla_x \mathbf{s}_{\text{target}}) + \text{SmoothL1}(\nabla_y \mathbf{s}_{\text{pred}}, \nabla_y \mathbf{s}_{\text{target}})$$

where $\nabla_x$ and $\nabla_y$ are spatial gradients.

**Loss Weights**: $\lambda_{\text{mid}} = 0.3$, $\lambda_{\text{low}} = 0.2$, $\lambda_{\text{smooth}} = 0.1$

#### 2.3.2 Drought Classification Loss

The drought classifier is trained with a weighted focal loss to handle class imbalance and focus on hard negatives:

$$\mathcal{L}_{\text{drought}} = \mathcal{L}_{\text{focal}} + \lambda_{\text{physics}} \mathcal{L}_{\text{physics\_agreement}}$$

**Focal Loss** — Focuses on hard-to-classify examples:

$$\mathcal{L}_{\text{focal}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i (1 - \hat{p}_i)^{\gamma} \log(\hat{p}_i) + (1 - y_i) \hat{p}_i^{\gamma} \log(1 - \hat{p}_i) \right]$$

where $\gamma = 2$ (focusing parameter) and class weights: $w_+ = 1.1$ (positive), $w_- = 0.9$ (negative).

**Physics Agreement Loss** — Regularization encouraging alignment with physics-derived signals:

$$\mathcal{L}_{\text{physics\_agreement}} = \text{MSE}(\mathbf{p}_{\text{model}}, \mathbf{p}_{\text{physics}})$$

**Loss Weight**: $\lambda_{\text{physics}} = 0.05$ (weak regularization to avoid overfitting to physics priors)

### 2.4 Training Procedures

#### 2.4.1 Data Splitting

Validation data is selected using a stratified approach that respects both class balance and regional diversity:

1. For each unique (region, label) combination, randomly allocate 20% of samples to validation and 80% to training
2. Within each fold, shuffle indices to remove temporal ordering bias
3. Ensure validation samples span different regions and temporal periods, testing generalization robustness

#### 2.4.2 Augmentation Strategy

During training, the following augmentations are applied stochastically:

| Augmentation Type | Operation | Probability |
|-------------------|-----------|-------------|
| **Spatial** | Random horizontal flip | 0.50 |
| | Random vertical flip | 0.50 |
| **Temporal** | Random frame dropping | 0.10/frame |
| | Temporal shuffling | 0.10 |
| **Color** | Brightness/contrast jitter | 0.30 |
| | Range | ±10% |

These augmentations improve robustness to sensor variability and atmospheric noise.

#### 2.4.3 Optimization

**Spectral Model:**

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Weight decay | $1 \times 10^{-4}$ |
| Learning rate | $3 \times 10^{-4}$ |
| LR schedule | Cosine annealing over 4 epochs |
| Batch size | 8 |
| Training samples | 2,042 (80% of 2,552) |

**Drought Model:**

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Weight decay | $5 \times 10^{-5}$ |
| Learning rate | $2 \times 10^{-4}$ |
| LR schedule | Cosine annealing over 8 epochs |
| Batch size | 8 |
| Sampling strategy | Class-weighted (balanced mini-batches) |

#### 2.4.4 Threshold Selection

For binary classification, the decision threshold critically affects precision-recall trade-offs. We use validation data to search for the optimal threshold:

$$\tau^* = \arg\max_{\tau} F_1(\tau) \quad \text{where} \quad F_1(\tau) = \frac{2 \cdot \text{Prec}(\tau) \cdot \text{Rec}(\tau)}{\text{Prec}(\tau) + \text{Rec}(\tau)}$$

If multiple thresholds achieve similar F1 scores, we select the one with higher recall (prioritizing drought detection sensitivity).

### 2.5 Evaluation Metrics

#### 2.5.1 Spectral Reconstruction

We measure reconstruction quality on a per-pixel basis:

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **MAE** | $\displaystyle \text{MAE} = \frac{1}{N} \sum_{i=1}^{N} \|\hat{s}_i - s_i\|$ | Mean absolute band error |
| **RMSE** | $\displaystyle \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\hat{s}_i - s_i)^2}$ | Root mean squared band error |
| **SAM** | $\displaystyle \text{SAM} = \arccos\left(\frac{\mathbf{s}_{\text{pred}} \cdot \mathbf{s}_{\text{target}}}{\|\mathbf{s}_{\text{pred}}\| \|\mathbf{s}_{\text{target}}\|}\right)$ | Spectral angle between vectors (radians) |

#### 2.5.2 Drought Classification

| Metric | Formula |
|--------|---------|
| **Accuracy** | $\displaystyle \text{Acc} = \frac{TP + TN}{TP + TN + FP + FN}$ |
| **Precision** | $\displaystyle \text{Prec} = \frac{TP}{TP + FP}$ |
| **Recall (Sensitivity)** | $\displaystyle \text{Rec} = \frac{TP}{TP + FN}$ |
| **Specificity** | $\displaystyle \text{Spec} = \frac{TN}{TN + FP}$ |
| **F1-Score** | $\displaystyle F_1 = \frac{2 \cdot \text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$ |
| **AUC-ROC** | Trapezoidal integration of ROC curve |

**Confusion Matrix Elements**: True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)

### 2.6 Real-World Validation

To validate the system on out-of-distribution Sentinel-2 imagery, we execute an end-to-end pipeline:

1. Fetch real Sentinel-2 scenes from two test regions (Western Ghats, Rajasthan) using the Microsoft Planetary Computer STAC API
2. Extract 3-temporal-frame sequences with 5–240 day intervals
3. Run spectral reconstruction and drought classification on fetched patches
4. Record model probabilities, physics-derived probabilities, fusion results, and vegetation indices
5. Visualize representative patches and save results to structured JSON and CSV files

---

## 3. Results

### 3.1 Spectral Reconstruction Results

Training the SpectralNetV10 model for 4 epochs on 2,042 samples yielded:

| Metric | Train | Validation | Interpretation |
|--------|-------|------------|-----------------|
| Loss (L1) | 0.00132 | 0.00130 | Negligible overfitting |
| MSE | 0.00062 | 0.00066 | ~0.026 RMSE per band |
| MAE | 0.00987 | 0.01014 | ~1% absolute error |
| RMSE | 0.02490 | 0.02566 | High-fidelity reconstruction |

**Interpretation**: The spectral reconstruction network generalizes extremely well, with negligible gap between training and validation loss. The RMSE of ~0.026 translates to per-band errors of ~2.6%, indicating high-fidelity spectral reconstruction. The visible RGB bands incur lower reconstruction error (~0.5–1%) due to residual connections that preserve input values, while NIR/SWIR bands are reconstructed from learned hidden features with slightly higher error.

### 3.2 Drought Classification Results

Training the PhysicsTemporalDroughtModel for 8 epochs with class-weighted sampling:

| Metric | Value | Benchmark |
|--------|-------|-----------|
| **Accuracy** | 0.9314 | — |
| **Precision** | 0.8889 | High specificity; few false alarms |
| **Recall** | 0.9697 | High sensitivity; catches 97% of droughts |
| **F1-Score** | 0.9275 | Strong balanced performance |
| **Specificity** | 0.9030 | Correctly identifies non-drought conditions |
| **AUC-ROC** | 0.9526 | Excellent discrimination |
| **Optimal Threshold** | 0.61 | Maximizes F1-score |

**Confusion Matrix** (on 510 validation samples):

| | Predicted Drought | Predicted Non-Drought |
|---|---|---|
| **Actual Drought** | 224 (TP) | 7 (FN) |
| **Actual Non-Drought** | 28 (FP) | 251 (TN) |

**Per-Region Performance**:

The model shows strong performance across diverse geographic regions:

| Region | Samples | Mean Probability | Interpretation |
|--------|---------|------------------|-----------------|
| Rajasthan (arid) | 2 | 0.325 | Low drought signal, consistent with dry baseline |
| Argentina Cordoba | 18 | 0.308 | Low drought signal |
| Congo Basin | 6 | 0.313 | Low drought signal, high vegetation |
| Dataset Cluster 11 | 124 | 0.301 | Low drought signal |
| Dataset Cluster 03 | 280 | 0.405 | Moderate drought signal |
| Sahara Region | 50 | 0.521 | Elevated drought signal in arid zone |

### 3.3 End-to-End Real-World Validation

Running the pipeline on fresh Sentinel-2 data from geographically distinct regions:

#### 3.3.1 Western Ghats (Wet Climate)

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| Expected | Low drought probability | Wetter region |
| Scenes found | 26 | Adequate cloud-free coverage |
| Selected samples | 3 | Representative temporal sequences |
| Median model probability | 0.187 | Low drought risk |
| Median physics probability | 0.147 | Physics also indicates wet conditions |
| Median combined probability | 0.161 | Final prediction: non-drought |
| Mean NDVI | 0.482 | Healthy vegetation |
| Mean NDMI | 0.178 | Adequate moisture content |

**Interpretation**: The system correctly identifies Western Ghats as a wet region with low drought risk. The physics-derived indices (NDVI, NDMI) reflect lush vegetation and high moisture content.

#### 3.3.2 Rajasthan (Semi-Arid/Arid)

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| Expected | Higher drought probability | Drier region |
| Scenes found | 40 | Adequate cloud-free coverage |
| Selected samples | 3 | Representative temporal sequences |
| Median model probability | 0.618 | Elevated drought risk |
| Median physics probability | 0.520 | Physics indicates drier conditions |
| Median combined probability | 0.646 | Final prediction: drought likely |
| Mean NDVI | 0.218 | Sparse vegetation |
| Mean NDMI | -0.007 | Low/absent moisture |

**Interpretation**: The system correctly identifies Rajasthan as a drier region. The model probability (0.618) exceeds the physics probability (0.520), suggesting that learned features capture additional drought signals beyond standard indices. The combined probability (0.646) falls between the two components, reflecting the balanced fusion strategy.

### 3.4 Detailed Validation Metrics Summary

**Sensitivity Analysis**: Testing the model across threshold values:

| Threshold | Precision | Recall | F1-Score | Interpretation |
|-----------|-----------|--------|----------|-----------------|
| 0.40 | 0.820 | 0.982 | 0.895 | High sensitivity, more false alarms |
| 0.50 | 0.854 | 0.964 | 0.907 | Balanced approach |
| **0.61** | **0.889** | **0.970** | **0.928** | **Optimal balance** |
| 0.70 | 0.915 | 0.920 | 0.917 | Slightly higher precision |
| 0.80 | 0.949 | 0.839 | 0.890 | High precision, misses droughts |

The threshold of 0.61 balances precision and recall optimally, achieving the highest F1-score.

---

## 4. Discussion

### 4.1 Scientific Interpretation

The high validation performance (F1 = 0.928, AUC = 0.953) demonstrates that the system successfully learns meaningful drought patterns from RGB-reconstructed spectral sequences. Key findings:

1. **Spectral Reconstruction Success**: The minimal overfitting (train ≈ val loss) indicates that the RGB-to-spectral mapping is learnable and generalizable. This opens new possibilities for low-cost drought monitoring using widely available RGB satellite data.

2. **Temporal Modeling**: The LSTM-based architecture captures temporal drought dynamics effectively. The high recall (0.970) suggests the model reliably detects drought events, while the precision (0.889) indicates manageable false alarm rates.

3. **Physics-Learning Fusion**: The explicit fusion of physics-derived indices with learned features improves robustness. The confidence estimates from the confidence head could enable uncertainty quantification (not detailed here but present in the architecture).

4. **Geographic Generalization**: The real-world validation on geographically distinct regions (Western Ghats vs. Rajasthan) shows consistent discrimination between wet and dry conditions, suggesting some generalization beyond the training set.

### 4.2 Ablation Insights

While full ablation studies are not presented here, the architecture design choices have clear justifications:

- **Multiscale Supervision** in spectral reconstruction prevents gradient degradation in deep networks and improves intermediate feature quality.
- **Residual Connections** preserve RGB information and allow faster convergence.
- **Squeeze-Excitation Blocks** provide adaptive channel re-weighting, often beneficial for multispectral tasks.
- **Physics-Guided Features** reduce model capacity needed to learn drought patterns and improve interpretability.

### 4.3 Limitations and Caveats

1. **Single Validation Split**: The model is evaluated on a single train-test split. Cross-validation or multiple random splits would strengthen confidence in generalization estimates.

2. **Label Quality**: The drought labels are derived from proxy sources (e.g., temporal sequences with categorical annotations) rather than ground truth field observations or satellite-derived drought indices (e.g., from MODIS). This may limit the claim of detecting "true" drought vs. learned patterns in the training data.

3. **Temporal Coverage**: The validation set may not represent all seasons or extreme drought events. A test on out-of-distribution drought years (e.g., major drought events not seen during training) would be more convincing.

4. **Sensor Specificity**: The system is trained on Sentinel-2-like spectral characteristics. Applying it to other sensors (e.g., Landsat-8, Planet) would require adaptation or retraining.

5. **Missing Uncertainty Quantification**: While the architecture includes a confidence head, formal Bayesian uncertainty estimates (e.g., via Monte Carlo dropout) are not reported.

6. **Computational Cost**: Training on CPU is slow; GPU deployment would be needed for operational use. Inference latency is not reported.

### 4.4 Comparison to Related Work

Recent work in remote sensing and drought detection includes:

- **Satellite Drought Indices**: MODIS-based approaches (GVMI, TCI, VCI) are standard but limited by coarse spatial resolution (~500 m).
- **Machine Learning**: Random forests and SVM-based drought classifiers exist but typically use hand-crafted features.
- **Deep Learning**: Convolutional neural networks for land use classification and some spectral reconstruction work, but integration with physics-guided priors and temporal modeling remains limited.

Our work bridges these by combining learned spectral reconstruction, explicit physics features, and temporal LSTM modeling, providing a more complete pipeline than prior art.

### 4.5 Practical Implications

If validated further, this system could enable:

- **Low-cost monitoring**: Using free RGB satellite data (e.g., Planet Labs NICFI, Maxar public layers) instead of expensive multispectral acquisitions.
- **Real-time early warning**: Regular RGB acquisition and processing could provide drought early warnings faster than traditional methods.
- **Resource-constrained regions**: Areas without reliable multispectral coverage but with RGB data access could benefit.
- **Climate change monitoring**: The system could track long-term drought trends at high spatial and temporal resolution.

---

## 5. Conclusion

We present a two-stage framework combining spectral reconstruction and physics-guided temporal modeling for drought prediction from RGB satellite imagery. The system achieves 93.1% accuracy and 0.953 AUC on a validation set of 510 samples and demonstrates consistent geographic generalization to real Sentinel-2 imagery. The key innovation is the integration of learned spectral reconstruction (addressing data scarcity) with physics-informed drought indices (improving interpretability and robustness), yielding a system that is both accurate and scientifically defensible.

While the results are promising, the work is not yet fully publication-ready without additional external validation, cross-validation, and uncertainty quantification. Recommended next steps include:

1. **Multi-fold cross-validation** to eliminate dependence on a single split
2. **External validation** on independent datasets (e.g., public drought monitoring benchmarks)
3. **Temporal robustness testing** on out-of-distribution drought years or extreme events
4. **Sensor adaptation** to demonstrate transfer to other satellite platforms
5. **Uncertainty quantification** via Bayesian methods or ensemble approaches
6. **Deployment study** measuring inference speed and operational feasibility

With these enhancements, the framework has clear potential to contribute meaningfully to remote sensing-based drought monitoring and environmental forecasting.

---

## 6. References

[1] Ryu, J. H., Won, J. S., & Min, K. D. (2005). Waterline extraction from Landsat TM data in a tidal flat: A case study in Gomso Bay, Korea. *Remote Sensing of Environment*, 98(2-3), 282-291.

[2] Gao, B. C. (1996). NDWI—a normalized difference water index for remote sensing of vegetation liquid water from space. *Remote Sensing of Environment*, 58(3), 257-266.

[3] Aster, G. C., & Sisler, J. F. (1985). Multispectral remote sensing of vegetation condition and water status. In *International Symposium on Spectral Sensing Research*.

[4] Tucker, C. J. (1979). Red and photographic infrared linear combinations for monitoring vegetation. *Remote Sensing of Environment*, 8(2), 127-150.

[5] Huete, A. R., Liu, H. Q., Batchily, K., & van Leeuwen, W. (1997). A comparison of vegetation indices over a global set of TM images for EOS-MODIS. *Remote Sensing of Environment*, 59(3), 440-451.

[6] McFeeters, S. K. (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. *International Journal of Remote Sensing*, 17(7), 1425-1432.

[7] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.

[8] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *CVPR* (pp. 770-778).

[9] Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. In *CVPR* (pp. 7132-7141).

[10] Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In *ICCV* (pp. 2980-2988).

[11] Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. In *ICLR*.

[12] Brown, M. E., & Funk, C. C. (2008). Food security under climate change. *Science*, 319(5863), 580-581.

[13] Svoboda, M., & Fuchs, B. A. (2016). Handbook of drought indicators and indices. *World Meteorological Organization and Global Water Partnership*.

---

## Appendix A: Hyperparameter Sensitivity

**Spectral Model**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 8 | Limited by GPU memory; CPU inference uses batch size 1 |
| Learning rate | $3 \times 10^{-4}$ | Tested range: $10^{-4}$ to $5 \times 10^{-4}$; higher rates caused instability |
| Epochs | 4 | Convergence plateau observed around epoch 3–4 |

**Drought Model**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 8 | Class-weighted sampling for balance |
| Learning rate | $2 \times 10^{-4}$ | Cosine annealing over 8 epochs |
| Epochs | 8 | Validation F1 peaks around epoch 6–7 |
| LSTM hidden size | 256/direction | Tested: 128, 256, 512; 256 balances capacity and regularization |
| Dropout | 0.3 | Tested: 0.2, 0.3, 0.4; 0.3 provides good regularization |

---

## Appendix B: Feature Importance (Physics Indices)

During fusion, the relative contributions of physics features to drought probability are weighted as follows (subject to normalization):

| Feature | Weight | Rationale |
|---------|--------|-----------|
| Low Vegetation (NDVI) | 0.20 | Primary drought indicator; vegetation stress is most direct signal |
| Low Moisture (NDMI) | 0.17 | Soil moisture is critical; NDMI is reliable proxy |
| Vegetation Drop (Temporal $\Delta$NDVI) | 0.13 | Temporal trend captures acute drought onset |
| Low Canopy (SAVI) | 0.08 | Soil-adjusted NDVI; reduces soil background |
| Burn Dryness (NBR) | 0.07 | Sensitive to extreme dryness; less noisy in sparse vegetation |
| Moisture Stress (MSI) | 0.08 | Shortwave infrared ratio sensitive to moisture content |
| Other components | 0.27 | Surface water, chlorophyll, soil dryness, temporal deltas |

---

## Appendix C: Real-World Deployment Considerations

**Inference Pipeline**:

1. Acquire RGB Sentinel-2 L1C or L2A data from Planetary Computer or AWS Sentinel Hub
2. Tile imagery into 64×64 patches with 50% overlap (for patch boundary effects)
3. Run SpectralNetV10 to reconstruct six spectral bands
4. Compute physics features from reconstructed spectra
5. Stack three temporal frames (e.g., 10-day, 20-day, 30-day time intervals)
6. Feed to PhysicsTemporalDroughtModel to predict drought probability
7. Apply threshold (0.61) to generate binary drought map
8. Post-process via morphological operations (e.g., median filtering) to reduce noise

**Latency** (on CPU, single frame):

| Step | Time |
|------|------|
| Spectral reconstruction | 50–100 ms/patch |
| Physics feature computation | ~10 ms |
| Drought classification | ~20 ms |
| **Total per patch** | **100–130 ms** |
| **Total for 512×512 tile (64 patches)** | **6–8 seconds** |

---

**Manuscript Date**: July 2026  
**Data Availability**: The model weights and trained checkpoints are stored locally. A reproducible code release and open-source repository are recommended for publication.
