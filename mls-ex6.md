# Exercise Sheet 6

**Name:** Aryan Varshneya
**Immatriculation Number:** 261678

# Exercise 6.1: Why Explainability?

Explainability is particularly important in safety-critical machine learning applications such as autonomous driving, healthcare, aviation, and industrial automation. In these domains, an incorrect prediction can have serious consequences, including financial losses, physical damage, or even loss of life. Therefore, it is not sufficient to know that a model works; we must also understand _why_ it makes a certain decision.

One major advantage of explainability is that it helps engineers identify potential failure modes. For example, an autonomous vehicle classifier may appear highly accurate during testing but could actually rely on irrelevant features such as weather conditions or road markings rather than the presence of vehicles themselves. Explainability techniques can reveal such hidden dependencies.

Another advantage is increased trust and accountability. Safety assessors, regulators, and end users are more likely to trust a system if its decisions can be justified and understood. Explainability also supports debugging, validation, and compliance with emerging AI regulations.

However, current explainability methods have important limitations. Most methods provide only an approximation of the model's reasoning process rather than a guaranteed explanation. Different explainability techniques may produce different explanations for the same prediction. Furthermore, explanations can sometimes be misleading, causing users to believe they understand a model when they actually do not. Therefore, explainability should be viewed as a useful diagnostic tool rather than definitive proof of correctness.

# Exercise 6.2: Local vs. Global Explainability

Explainability methods can generally be divided into **local** and **global** approaches.

## Local Explainability

Local explainability focuses on a single prediction. The goal is to understand why the model made a specific decision for a particular input.

For example, if an image classifier predicts that a road image contains a pedestrian, a local explanation attempts to identify which parts of that image influenced the prediction.

A common local explainability method is **Saliency Maps**. Saliency maps highlight the pixels that have the strongest influence on the prediction.

The question answered by local explainability is:

> Why did the model make this specific prediction?

## Global Explainability

Global explainability focuses on the overall behavior of the model rather than individual predictions. It aims to understand which patterns, features, or relationships the model has learned from the entire dataset.

For example, a global explanation may reveal that a vehicle detector generally relies on wheels, headlights, and vehicle contours when making predictions.

A common global explainability method is **Feature Importance Analysis**, which measures the overall contribution of input features to model decisions.

The question answered by global explainability is:

> How does the model generally make decisions?

## Summary

Local explainability explains individual decisions, whereas global explainability explains the overall behavior of the model. Both perspectives are important because a model may appear reasonable globally while still making problematic decisions for individual samples.

# Exercise 6.3: Saliency vs. Occlusion

## Saliency Method

The saliency method explains a prediction by computing the gradient of the model output with respect to the input image pixels. Intuitively, the gradient measures how strongly a small change in a pixel would affect the prediction.

Mathematically, the saliency map is based on: df(x)/dx

where:

- (f(x)) is the model prediction,
- (x) is the input image.

Pixels with larger gradient values are considered more important because changing them would have a stronger effect on the prediction.

### Example

Suppose a neural network predicts that an image contains a vehicle. A saliency map may highlight the wheels, headlights, and vehicle edges because these pixels strongly influence the prediction.

### Advantage

Saliency maps are computationally efficient because they require only one forward pass and one backward pass through the network.

### Disadvantage

The resulting explanations are often noisy and difficult to interpret, as important pixels may appear scattered across the image.

## Occlusion Method

The occlusion method explains a prediction by systematically masking parts of the input image and observing how the prediction changes.

If masking a particular region causes a significant drop in prediction confidence, that region is considered important.

### Example

Suppose a vehicle detector predicts "Vehicle = 95%". If covering the car itself reduces the prediction to 20%, then that region is clearly important. If covering the sky changes the prediction only from 95% to 94%, then the sky is not important.

### Advantage

Occlusion produces intuitive and easily interpretable explanations because it directly measures the effect of removing information from the input.

### Disadvantage

Occlusion is computationally expensive because the model must be evaluated many times using different masked versions of the same image.

---

## Comparison

Saliency is significantly faster but often less interpretable. Occlusion is slower but usually provides more intuitive explanations because it directly measures the contribution of image regions to the prediction.

# Exercise 6.4: Chain-of-Thought Fidelity

## 1. What does it mean for a thinking trace to be faithful? Why is faithfulness hard to verify?

A thinking trace is considered **faithful** if it accurately reflects the internal reasoning process that produced the model's answer. In other words, the explanation should describe the actual factors that influenced the decision rather than merely providing a plausible justification afterward.

Faithfulness is difficult to verify because modern large language models contain billions of parameters and complex internal computations that are not directly observable. We can see the generated reasoning but cannot easily determine whether the model truly relied on that reasoning internally.

As a result, a chain of thought may appear logical and convincing while not representing the actual decision process.

---

## 2. What is simulatability? Give an example where simulatability is satisfied but the trace is unfaithful.

Simulatability measures whether a human can correctly predict the model's final answer after reading its thinking trace.

A highly simulatable explanation allows a reader to understand how the answer was obtained and reproduce the prediction.

### Example

Suppose an LLM answers:

> 27 × 13 = 351

and provides the reasoning:

> 27 × 10 = 270

> 27 × 3 = 81

> 270 + 81 = 351

A human can easily follow these steps and predict the answer, so simulatability is high.

However, the model may not have internally performed these calculations. It may have directly generated the answer from memorized patterns and only produced the reasoning afterward. In this case, the trace is simulatable but not faithful.

---

## 3. What does counterfactual simulatability add? Why is it stricter?

Counterfactual simulatability evaluates whether the explanation remains useful when the input changes.

Instead of only asking:

> Can I predict the current answer?

it asks:

> Can I predict how the answer would change if the input changed?

### Example

Consider the previous multiplication example. If the input changes from:

> 27 × 13

to

> 27 × 14

a faithful explanation should allow the reader to correctly predict the new answer.

Counterfactual simulatability is stricter because it evaluates whether the reasoning captures the actual decision mechanism rather than merely explaining a single example.

---

## 4. Why is an unfaithful thinking trace a safety risk?

An unfaithful thinking trace can create a false sense of trust and transparency. Users may believe they understand why the model made a decision when the explanation does not actually reflect the underlying reasoning.

### Example

Consider a medical diagnosis system that recommends a treatment and explains:

> The recommendation was made because of elevated blood pressure and abnormal ECG readings.

In reality, the model may have relied on an unrelated artifact in the training data, such as the hospital where the patient was treated.

Doctors may trust the explanation and approve the recommendation, even though the model's actual reasoning is flawed. This can lead to incorrect decisions, reduced accountability, and potentially harmful outcomes.

Therefore, ensuring faithful explanations is an important challenge in the safety evaluation of large language models.
