# Exercise Sheet 8 - Adversarial Machine Learning

## Exercise 8.1: What Are Adversarial Examples?

### 8.1.1 What is an adversarial example?

An adversarial example is an input that has been intentionally modified by a small, carefully chosen perturbation in order to cause a machine learning model to make an incorrect prediction.

The perturbation is usually so small that it is difficult or impossible for a human to notice. However, because the perturbation is specifically optimized to exploit weaknesses in the model, it can significantly change the model's output.

For example, a pedestrian image may be modified by a tiny amount of noise that causes a classifier to incorrectly predict that no pedestrian is present.

---

### 8.1.2 How do adversarial examples differ from OOD examples?

Although both adversarial examples and out-of-distribution (OOD) examples can cause model failures, they arise in different ways.

OOD examples occur naturally when the input differs from the data seen during training. Examples include fog, night-time conditions, unusual weather, or new environments.

Adversarial examples, in contrast, are deliberately constructed by an attacker using knowledge of the model or its gradients. The goal is to force a prediction error while keeping the image visually similar to the original.

Therefore:

- OOD examples result from distribution shift.
- Adversarial examples result from intentional manipulation.
- OOD inputs may look very different from training data.
- Adversarial inputs often appear unchanged to humans.

---

## Exercise 8.2: Attack Formulation

Consider the gradient-based attack:

$$
x_{i+1}=x_i+\alpha \nabla_x L(y,f(x_i))
$$

### 8.2.1 What does each term represent?

- $x_i$ : Current input image.
- $x_{i+1}$ : Updated adversarial image.
- $\alpha$ : Step size controlling the magnitude of the update.
- $L(y,f(x_i))$ : Loss function comparing the true label $y$ with the model prediction.
- $\nabla_x L(y,f(x_i))$ : Gradient of the loss with respect to the input image.
- $f(x_i)$ : Model prediction for the current image.

The gradient identifies the direction in which the input should be changed to increase the loss and make the model more likely to produce an incorrect prediction.

---

### 8.2.2 Targeted vs Untargeted Attacks

An untargeted attack attempts to cause any incorrect prediction.

The attack increases the loss:

$$
x_{i+1}=x_i+\alpha \nabla_x L(y,f(x_i))
$$

A targeted attack attempts to force the model to predict a specific target class $t$.

In that case the attack minimizes the loss for the target class:

$$
x_{i+1}=x_i-\alpha \nabla_x L(t,f(x_i))
$$

The sign changes because the attacker wants the model output to move toward the chosen target class.

---

### 8.2.3 Perturbation Budget

The basic update rule does not guarantee that the final perturbation remains within a specified budget:

$$
|x_0-x_t| \leq \varepsilon
$$

Repeated updates can accumulate and produce a perturbation larger than the allowed limit.

To enforce the perturbation budget, the adversarial example can be projected back into an $\varepsilon$-ball around the original image after each update:

$$
x_{i+1}=Proj_{\varepsilon}(x_i+\alpha \nabla_xL)
$$

This ensures that the total perturbation never exceeds the allowed budget.

---

## Exercise 8.3: Defenses

Adversarial training improves robustness by including adversarial examples during training.

Instead of training only on clean images, the model is trained on both clean and adversarially perturbed inputs. The model therefore learns to classify adversarial examples correctly and becomes less sensitive to small perturbations.

### Trade-Offs

Adversarial training introduces several trade-offs:

- Increased training time because adversarial examples must be generated during training.
- Higher computational cost.
- Potential reduction in clean-data accuracy.
- Improved robustness against attacks similar to those used during training.

Thus adversarial training generally improves robustness at the cost of additional computation and sometimes reduced performance on clean data.

---

## Exercise 8.4: Generating Adversarial Examples

### 8.4.1 FGSM Implementation

The Fast Gradient Sign Method (FGSM) was implemented for the three binary classifiers:

- Pedestrian classifier
- Traffic Light classifier
- Vehicle classifier

FGSM generates adversarial examples using:

$$
x_{adv}=x+\varepsilon \cdot sign(\nabla_xL(y,f(x)))
$$

The sign of the gradient determines the direction that increases the loss most rapidly, while $\varepsilon$ controls the perturbation strength.

---

### 8.4.2 Adversarial Examples

Adversarial examples were generated for:

$$
\varepsilon \in {0.01, 0.05, 0.1}
$$

Larger values of $\varepsilon$ produce stronger perturbations and therefore more difficult inputs for the classifier.

---

### 8.4.3 Visual Comparison

A clean image was displayed alongside its adversarial counterparts.

Observations:

- At $\varepsilon = 0.01$, perturbations are barely visible.
- At $\varepsilon = 0.05$, small artifacts become noticeable.
- At $\varepsilon = 0.1$, noise is clearly visible to a human observer.

The visual appearance changes only slightly, but the perturbations are sufficient to significantly affect model predictions.

---

## Exercise 8.5: Measuring Robustness

The models were evaluated on adversarially perturbed test images.

### Results

| Model         | Epsilon | Clean Recall | Adversarial Recall | Recall Drop |
| ------------- | ------- | ------------ | ------------------ | ----------- |
| Pedestrian    | 0.01    | 0.278        | 0.119              | 0.159       |
| Pedestrian    | 0.05    | 0.278        | 0.096              | 0.181       |
| Pedestrian    | 0.10    | 0.278        | 0.232              | 0.045       |
| Traffic Light | 0.01    | 0.983        | 0.919              | 0.064       |
| Traffic Light | 0.05    | 0.983        | 0.077              | 0.906       |
| Traffic Light | 0.10    | 0.983        | 0.0004             | 0.983       |
| Vehicle       | 0.01    | 0.877        | 0.756              | 0.122       |
| Vehicle       | 0.05    | 0.877        | 0.713              | 0.164       |
| Vehicle       | 0.10    | 0.877        | 0.874              | 0.004       |

### Discussion

The Traffic Light classifier is highly vulnerable to adversarial perturbations.

Its recall decreases from 98.3% on clean inputs to almost zero at $\varepsilon=0.1$, demonstrating that small adversarial perturbations can completely break the classifier.

The Vehicle and Pedestrian models also exhibit performance degradation, although the effect is less consistent. Some non-monotonic behavior is observed at larger perturbation strengths, which may occur because FGSM is a single-step attack and certain perturbations can accidentally move samples toward the correct decision boundary.

Overall, the results demonstrate that adversarial examples can significantly reduce perception performance and therefore represent an important safety concern for autonomous driving systems.

---

## Exercise 8.6: Extending the Safety Analysis for Adversarial Robustness

### 8.6.1 Hazard

The original safety analysis considered perception failures but did not explicitly consider adversarial manipulation of camera inputs.

The following hazard is added:

| ID  | Hazard                                                                                                        | Possible Losses                                                              |
| --- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| H-6 | The perception system produces incorrect outputs because the camera input has been adversarially manipulated. | L-1 (injury or loss of life), L-2 (vehicle collision), L-3 (property damage) |

---

### 8.6.2 Unsafe Control Action

| ID        | Controller      | Control Action                   | UCA Type          | Hazard(s) | Unsafe Scenario                                                                                                    |
| --------- | --------------- | -------------------------------- | ----------------- | --------- | ------------------------------------------------------------------------------------------------------------------ |
| UCA-ADV-1 | Planning Module | Continue driving at normal speed | Provided Unsafely | H-6       | The planner continues driving based on a false negative produced by an adversarially manipulated perception input. |

For example, an adversarial attack may cause the pedestrian classifier to miss a pedestrian, leading the planner to continue driving when braking is required.

---

### 8.6.3 Safety Constraints

#### Model-Level Constraint

| Constraint                                                                                                                                  | Level       | Justification                                                                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------ |
| The perception models shall maintain acceptable recall under adversarial perturbations up to a specified perturbation budget $\varepsilon$. | Model-Level | Experimental results show that adversarial perturbations can substantially reduce recall and therefore increase accident risk. |

#### System-Level Constraint

| Constraint                                                                                                                                                             | Level        | Justification                                                                                |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | -------------------------------------------------------------------------------------------- |
| When perception outputs exhibit low confidence or anomalous behavior, the planner shall enter a safe fallback mode by reducing speed or requesting human intervention. | System-Level | Even robust models may fail under attack, so additional system-level protection is required. |

---

### 8.6.4 Residual Risk

Residual risk remains even if adversarial training is applied.

First, adversarial training generally improves robustness only against attack types that are similar to those used during training. Stronger or previously unseen attacks may still succeed.

Second, perception failures can still occur due to sensor faults, occlusions, or environmental conditions that are unrelated to adversarial manipulation.

Therefore, adversarial training reduces risk but does not eliminate it. Safe system behavior still requires redundancy, monitoring, fallback strategies, and human oversight where appropriate.
