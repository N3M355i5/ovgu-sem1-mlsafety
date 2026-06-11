# Exercise Sheet 9 - Anomaly Detection

## Exercise 9.1: The OOD Problem

**9.1.1 Why can a standard classifier not be trusted to signal OOD inputs?**

A standard classifier is trained to map every input to one of its known classes - that's all it knows how to do. When it sees something completely outside its training distribution (like a foggy image when it was only trained on sunny ones), it doesn't have a way to say "I don't recognize this." Instead, it just picks the class whose learned pattern happens to match the OOD input the most, often with surprisingly high confidence. The softmax output always sums to 1, so the model is forced to express certainty even when it has no business doing so. There's no built-in mechanism for "none of the above."

**9.1.2 Why is silent failure worse than uncertain failure?**

Silent failure means the system is confidently wrong - it produces a plausible-looking output with no warning sign, so no downstream component (human or automated) gets a chance to intervene. In contrast, uncertain failure at least raises a flag: low confidence scores can trigger fallback behaviors, hand the decision to a human operator, or bring the vehicle to a safe stop. In safety-critical systems like autonomous driving, a wrong prediction you know about is recoverable; a wrong prediction you don't know about can be catastrophic.

---

## Exercise 9.2: MSP Baseline

The Maximum Softmax Probability (MSP) baseline, introduced by Hendrycks & Gimpel (2017), is the simplest possible OOD detector. After the classifier runs a forward pass and produces softmax probabilities over all classes, you just take the highest probability across all classes as a confidence score:

$$D(x) = -\max_y \, p(y \mid x)$$

The idea is intuitive: if the model is confident about a class, the input is probably in-distribution; if it's unsure and spreads probability mass across many classes, it might be OOD. The score is negated so that high values indicate OOD (as is convention for outlier scores).

**Limitations:**

The main problem is that neural networks, especially after softmax, tend to be overconfident on OOD inputs. The softmax function is essentially a competition between logits - if one logit dominates slightly, the output looks confident, even for completely nonsensical inputs. As shown in the GTSRB example from the lecture, random textures and noise patterns can receive 96–100% confidence for a traffic sign class. So while MSP works to some degree, it's unreliable precisely in the cases that matter most.

---

## Exercise 9.3: Alternative Method - Energy Score (EBO)

The Energy-Based OOD detector (Liu et al., 2020) addresses MSP's overconfidence problem by working with the raw logits before the softmax is applied, rather than the compressed probabilities after it. The key insight is that the softmax destroys useful information - the absolute scale of the logits - by normalizing everything into a probability distribution.

The energy score is computed as the log-sum-exp of the logits (with T = 1):

$$D(\mathbf{x}) = -\log \sum_{i}^{K} e^{f_i(\mathbf{x})}$$

In practice, this is a single line of PyTorch:

```python
energies = -torch.logsumexp(logits, dim=1)
```

**Why it's better than MSP:**

The authors prove that the energy function is more directly aligned with the true data density $p(x)$ than the maximum softmax probability is. Intuitively, in-distribution inputs tend to produce large logits overall (high energy magnitude), while OOD inputs produce smaller, more diffuse logit values. Because energy uses the full logit vector rather than just the winning class, it captures more information about how "activated" the model is as a whole. Empirically it consistently outperforms MSP across standard benchmarks, and the implementation cost is essentially zero since it requires no changes to the model.

## Exercise 9.4: Visualising the Distribution Shift

### 9.4.1 Comparison of In-Distribution, Fog, and Night Images

Five images from the in-distribution test set, five fog images, and five night images were visualized and compared.

The in-distribution images were captured under clear daytime conditions and closely resemble the training data. In contrast, the fog images exhibit reduced visibility, lower contrast, and partially obscured objects. The night images are significantly darker, making object boundaries and scene details more difficult to recognize.

These visual differences indicate a clear distribution shift between the training distribution and the fog/night datasets.

---

### 9.4.2 Comparison with a Different CARLA Town

Five images from the different-town dataset were also visualized.

Unlike the fog and night datasets, the different-town images were captured under similar weather and lighting conditions as the training data. However, they contain different road layouts, buildings, intersections, vegetation, and background scenery.

The distribution shift in the different-town dataset is therefore more subtle than the fog and night shifts. While the visual appearance remains similar, the spatial structure of the environment differs from the training data.

Fog and night primarily affect image quality and visibility, whereas the different-town dataset changes the scene context and environment.

---

### 9.4.3 Mean Model Confidence under Distribution Shift

The mean prediction confidence was computed for all three classifiers on the in-distribution test set and on the fog and night datasets.

| Model         | Test  | Fog   | Night |
| ------------- | ----- | ----- | ----- |
| Pedestrian    | 0.137 | 0.036 | 0.045 |
| Traffic Light | 0.749 | 0.365 | 0.218 |
| Vehicle       | 0.704 | 0.408 | 0.409 |

### Discussion

All three models become less confident when evaluated on out-of-distribution data.

The pedestrian model exhibits the lowest confidence overall and is strongly affected by fog and night conditions. The traffic-light model shows a substantial decrease in confidence under distribution shift, particularly at night. The vehicle model remains comparatively more robust but still experiences a noticeable confidence reduction compared to the in-distribution test set.

These results indicate that distribution shifts reduce model confidence and suggest that prediction confidence can be used as a simple signal for identifying potential out-of-distribution inputs.

## Exercise 9.5: Is the Different Town Out-of-Distribution?

### 9.5.1 Does the Original ODD Clearly Classify the Different Town?

No.

The ODD defined in Exercise 2.2 focused on environmental conditions such as weather, lighting, camera condition, scene type, and vehicle speed. However, it did not explicitly specify whether the autonomous driving system was limited to a particular CARLA town or road layout.

As a result, the original ODD is ambiguous regarding the different-town dataset. The images satisfy the expected weather and lighting conditions, but they contain roads, buildings, and scenery that were never seen during training.

Therefore, based on the original wording, it is unclear whether the different-town images should be considered inside or outside the ODD.

---

### 9.5.2 Revised ODD

#### Revised ODD Statement

The system is intended to operate on urban and suburban road environments in CARLA under daytime conditions with good visibility, regardless of the specific town layout, provided that weather, lighting, camera quality, and vehicle speed remain within the defined operational limits.

#### Choice

The different-town dataset is considered **inside the ODD**.

#### Justification

The weather, lighting, and camera conditions of the different-town images remain consistent with the intended operating conditions. Although the road layout, buildings, and vegetation differ from the training data, a safe autonomous driving system should be capable of generalizing to previously unseen towns that satisfy the same environmental requirements.

Restricting the ODD to only a single CARLA town would make the system impractical, since real-world autonomous vehicles must operate in many different locations rather than memorizing specific road layouts.

---

### 9.5.3 Implications

Because the different-town images are considered inside the revised ODD, they should be treated as inputs that the system is expected to handle correctly.

The perception models and planner should continue to operate safely and accurately in these environments. Poor performance on the different-town dataset would therefore indicate a lack of generalization rather than an OOD event.

In contrast, the fog and night datasets clearly violate the ODD because they differ substantially from the intended weather and lighting conditions. These inputs should therefore be detected by an OOD monitoring system and trigger an appropriate safety response.

Consequently:

- Different-town images → In-ODD and must be handled correctly.
- Fog images → OOD and should be flagged.
- Night images → OOD and should be flagged.

## Exercise 9.6: Evaluating the MSP Baseline

### 9.6.1 Distribution of OOD Scores

The Traffic Light classifier was selected for the MSP baseline evaluation. The maximum softmax probability (MSP) was used as an OOD score, where low confidence indicates a potentially out-of-distribution input.

The histogram shows a clear separation between the in-distribution and OOD datasets. Most in-distribution images receive very high confidence scores close to 1.0. In contrast, the fog and night images are shifted toward lower confidence values, particularly the night images.

This indicates that the model is generally less confident when evaluated on inputs that differ from the training distribution.

---

### 9.6.2 AUROC

The fog and night datasets were combined into a single OOD dataset and compared against the in-distribution test dataset.

AUROC = **0.7968**

An AUROC of approximately 0.80 indicates that the MSP baseline is reasonably effective at distinguishing in-distribution images from OOD images. The result is substantially better than random guessing (AUROC = 0.5), although there remains overlap between the distributions, preventing perfect separation.

Overall, the MSP baseline provides useful OOD detection capability for this model, but it is not completely reliable under all conditions.

## Exercise 9.7: Feature-Based OOD Detection

### 9.7.1 Feature Extraction

A feature-based OOD detector was implemented using the k-Nearest Neighbors (k-NN) method. The same Traffic Light classifier used in Exercise 9.6 was selected for evaluation.

Instead of using the final prediction confidence, deep feature vectors were extracted from the penultimate layer of the ResNet-18 model. Specifically, the output after the global average pooling layer was used, resulting in a 512-dimensional feature representation for each image.

The detector was fitted only on in-distribution validation features. Features were then extracted from:

- In-distribution test images
- Fog images
- Night images

The intuition is that in-distribution samples should produce feature vectors similar to those observed during training, whereas OOD samples should be farther away in feature space.

---

### 9.7.2 k-NN OOD Detector

A 1-nearest-neighbor detector was trained using the in-distribution validation features.

For each test image, the Euclidean distance to its nearest in-distribution neighbor was computed:

- Small distance → likely in-distribution
- Large distance → likely OOD

The resulting histogram shows that the in-distribution images are concentrated at low distances, while fog and night images generally produce larger distances.

The night dataset exhibits the largest distances, indicating a stronger distribution shift than the fog dataset.

---

### 9.7.3 AUROC Comparison with MSP

The k-NN detector was evaluated using the same OOD detection setup as in Exercise 9.6.

| Method | AUROC  |
| ------ | ------ |
| MSP    | 0.7968 |
| k-NN   | 0.9205 |

The feature-based detector achieves a substantially higher AUROC than the MSP baseline.

An AUROC of **0.9205** indicates very strong separation between in-distribution and OOD samples. This suggests that deep feature representations contain more useful information for detecting distribution shift than prediction confidence alone.

The improvement over MSP demonstrates that feature-space distances are more effective at identifying OOD inputs in this CARLA dataset.

#### Largest OOD Gap

The largest separation appears for the **Night** dataset.

In the feature-distance histogram, night images are located substantially farther from the in-distribution samples than fog images. This indicates that nighttime conditions produce a larger shift in the learned feature space and are therefore easier to detect as OOD.

Overall, the feature-based k-NN detector outperformed the MSP baseline and provided more reliable OOD detection across the evaluated scenarios.

## Exercise 9.8: Extending the Safety Analysis for OOD

### 9.8.1 Hazard

The original hazard analysis focused on incorrect perception outputs such as missed pedestrians, vehicles, or traffic lights. However, it did not explicitly consider the case where the perception system is operating on out-of-distribution (OOD) inputs without recognizing that the input is outside its operational design domain.

Therefore, the following hazard is added:

| ID  | Hazard                                                                                                                      | Possible Losses                                                              |
| --- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| H-5 | The perception system provides unreliable outputs on out-of-distribution inputs and the planner treats them as trustworthy. | L-1 (injury or loss of life), L-2 (vehicle collision), L-3 (property damage) |

This hazard is motivated by the results from Exercises 9.4–9.7, where model confidence and prediction quality degraded significantly on fog and night images.

---

### 9.8.2 Unsafe Control Action (UCA)

A new unsafe control action is introduced to capture the OOD scenario.

| ID        | Controller      | Control Action                   | UCA Type          | Hazard(s) | Unsafe Scenario                                                                                                                       |
| --------- | --------------- | -------------------------------- | ----------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| UCA-OOD-1 | Planning Module | Continue driving at normal speed | Provided Unsafely | H-5       | The planner continues driving normally even though the camera input is out-of-distribution and the perception outputs are unreliable. |

In this scenario, the perception system may fail to detect a pedestrian, vehicle, or traffic light because the input differs significantly from the training distribution. Since the planner is unaware of this uncertainty, it continues operating as if the perception results were trustworthy.

---

### 9.8.3 Safety Constraints

#### Model-Level Constraint

| Constraint                                                                                                                                                         | Level       | Justification                                                                                                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| The OOD monitor shall detect inputs that differ significantly from the training distribution and raise an alert when the OOD score exceeds a predefined threshold. | Model-Level | The hazard can lead to severe consequences including collisions and injury. The OOD detector must therefore achieve sufficient detection performance on representative OOD scenarios such as fog and night conditions. |

This constraint is supported by the experiments in Exercises 9.6 and 9.7, where MSP and k-NN detectors were evaluated using AUROC.

#### System-Level Constraint

| Constraint                                                                                                                                                          | Level        | Justification                                                                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| When an OOD condition is detected, the planner shall enter a safe fallback mode by reducing speed, increasing following distance, or requesting human intervention. | System-Level | Detecting OOD is only useful if the system responds appropriately. A safe fallback reduces the likelihood that unreliable perception outputs lead to unsafe vehicle behavior. |

---

### 9.8.4 Residual Risk

Even with a perfect OOD detector, residual risk remains.

First, an OOD detector can only identify that an input differs from the training distribution; it does not guarantee that the perception model will fail, nor does it provide the correct perception result. The system must still decide how to behave safely after detection.

Second, many dangerous situations occur within the operational design domain. For example, a pedestrian may be partially occluded, unusually positioned, or difficult to recognize despite the image being in-distribution. These failures would not necessarily be detected by an OOD monitor.

Therefore, OOD detection reduces risk but does not eliminate it. Detecting OOD inputs helps mitigate Hazard H-5, but additional safety mechanisms such as robust perception models, safe fallback behavior, redundancy, and human oversight remain necessary.
