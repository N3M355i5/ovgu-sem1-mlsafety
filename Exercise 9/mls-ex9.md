# Exercise Sheet 7 - Uncertainty Quantification

## Exercise 7.1: Epistemic vs. Aleatoric Uncertainty _(optional)_

**Epistemic uncertainty** is uncertainty caused by a _lack of knowledge_ - the model
hasn't seen enough relevant data to be sure. It is, in principle, **reducible**: given
more (relevant) training data, it shrinks, and in the limit of infinite data it
vanishes.

**Aleatoric uncertainty** is uncertainty caused by _irreducible noise_ inherent in the
data itself (label noise, measurement noise, genuine class overlap). It is
**not reducible** by collecting more data of the same kind - it remains even with
infinite data, because the randomness lives in the generating process, not in our
ignorance of it.

**1. Which type dominates for OOD inputs (e.g. night images for a model trained on day images)?**

**Epistemic.** A night image lies far outside the support of the training
distribution, so the model is "guessing" - not because the classes inherently overlap
at night, but because it has never learned what night-time features correspond to
which class. This is exactly the kind of uncertainty that _would_ shrink if we added
night images to the training set.

**2. Which type dominates for a correctly classified, in-distribution image?**

**Aleatoric.** If the input is in-distribution and correctly classified, the model
has "seen enough" of this kind of data - its remaining uncertainty (if the prediction
isn't at 100% confidence) mostly reflects genuine ambiguity in the input itself
(e.g. partial occlusion, borderline appearance) rather than a knowledge gap.

---

## Exercise 7.2: Calibration and ECE

**Calibration** describes how well a model's confidence matches its actual correctness
rate. A classifier is **well-calibrated** if, among _all_ predictions made with
confidence $p$, a fraction $p$ of them are actually correct. Example: if a model says
"80% confident" on 100 different inputs, a well-calibrated model should be right on
roughly 80 of them - not 50, not 100.

**Expected Calibration Error (ECE)** turns this into a single number by binning
predictions by confidence and comparing the bin's average confidence to its
empirical accuracy:

$$
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \, \big| \,\text{acc}(B_m) - \text{conf}(B_m) \, \big|
$$

where:

- $M$ = number of confidence bins (e.g. 20 bins of width 0.05),
- $B_m$ = the set of predictions whose confidence falls into bin $m$,
- $n$ = total number of predictions,
- $\text{acc}(B_m)$ = fraction of predictions in bin $m$ that are correct,
- $\text{conf}(B_m)$ = average confidence of predictions in bin $m$.

**Computation procedure:**

1. For each prediction, compute its confidence as $\max(p, 1-p)$ for binary classification
   (confidence in the _predicted_ class, not just the raw positive-class probability).
2. Sort predictions into $M$ equal-width bins by this confidence.
3. Within each bin, compute the accuracy (fraction correctly classified) and the average confidence.
4. ECE is the weighted average (by bin size) of the absolute gap between these two quantities.

A perfectly calibrated model has ECE $= 0$ (every bin's bar sits exactly on the diagonal
in a reliability diagram); larger ECE means confidence and correctness diverge more.

---

## Exercise 7.3: Cost-Optimal Downstream Decisions

Given costs:

|              | pedestrian present | no pedestrian |
| ------------ | ------------------ | ------------- |
| **brake**    | 0                  | $C_{FP}=1$    |
| **continue** | $C_{FN}=100$       | 0             |

**1. Expected loss of each action as a function of $p = p(\text{ped}\mid x)$:**

$$
\mathbb{E}[L \mid \text{brake}] = p \cdot 0 + (1-p)\cdot C_{FP} = (1-p)\cdot C_{FP}
$$

$$
\mathbb{E}[L \mid \text{continue}] = p \cdot C_{FN} + (1-p)\cdot 0 = p \cdot C_{FN}
$$

**2. Threshold $\tau^*$ where both actions have equal expected loss:**

$$
(1-\tau^*) \, C_{FP} = \tau^* \, C_{FN}
$$

$$
\tau^* = \frac{C_{FP}}{C_{FP} + C_{FN}}
$$

For $p > \tau^*$: brake (continuing is more costly in expectation).
For $p < \tau^*$: continue (braking unnecessarily is more costly in expectation).

**3. Plugging in $C_{FN} = 100,\ C_{FP} = 1$:**

$$
\tau^* = \frac{1}{1 + 100} = \frac{1}{101} \approx 0.0099 \ \ (\approx 1\%)
$$

Compared to the naive argmax threshold $\tau = 0.5$, the cost-optimal threshold is
**roughly 50× smaller**. This makes sense: since missing a real pedestrian
($C_{FN}=100$) is 100× more costly than an unnecessary brake ($C_{FP}=1$), the
system should brake even when it's only ~1% confident a pedestrian is present -
it doesn't need to be "more likely than not" before acting.

**4. Why does $\tau^*$ only give the optimal decision when $p$ is well-calibrated?**

The derivation above implicitly assumes the reported probability $p$ _is_ the true
probability of a pedestrian being present - that's the only way the expected-loss
arithmetic above is valid. If the model is **overconfident**, its reported $p$ values
are systematically more extreme than reality: many inputs that are truly only
~5% likely to contain a pedestrian get reported as $p \approx 0.5$ or higher (or,
conversely, many truly-risky inputs get pushed toward $p \approx 0$). Applying the
"correct" threshold $\tau^*=0.0099$ to _miscalibrated_ probabilities no longer
guarantees the cost-minimizing decision - the threshold was derived for _true_
probabilities, not distorted ones. An overconfident model could, for instance, report
$p < 0.0099$ on an input that actually does contain a pedestrian, causing the system
to wrongly "continue" with a high-cost false negative, even though the threshold
rule was applied correctly. **Calibration is therefore a precondition for any
cost-optimal threshold to actually deliver the promised expected cost.**

---

## Exercise 7.4: Measuring Calibration

**1. ECE per model (in-distribution test set):**

| Model         | ECE        |
| ------------- | ---------- |
| Pedestrian    | **0.1075** |
| Traffic Light | **0.0366** |
| Vehicle       | **0.0468** |

(Computed with 20 confidence bins, using $\text{confidence} = \max(p, 1-p)$.)

**2. Reliability diagrams:** plotted per model (confidence vs. observed accuracy),
each compared against the diagonal "perfect calibration" line.

**3. Over- or underconfident? Consistent across models?**

The three models do **not** show a single consistent direction of miscalibration:

- **Pedestrian:** clearly **underconfident at low confidence** (curve lies _above_
  the diagonal for confidence $\lesssim 0.5$, i.e. accuracy is higher than the
  confidence suggests) and **overconfident at high confidence** (curve lies _below_
  the diagonal above $\approx 0.8$ confidence, i.e. the model claims near-100%
  certainty but is only correct ~95–96% of the time there). This is the classic
  "S-shaped" miscalibration pattern, and it is the model with by far the worst
  ECE (0.1075).
- **Vehicle:** mostly **underconfident across the whole range** - the curve sits
  above the diagonal almost everywhere, including at high confidence, meaning the
  model is _more accurate than it claims to be_ for most of its prediction range.
- **Traffic Light:** the curve oscillates noisily around the diagonal in both
  directions with no clear systematic bias - consistent with it having by far the
  **lowest** ECE (0.0366) and needing only a mild temperature correction ($T=1.1$,
  see 7.5).

**Conclusion:** the pattern does _not_ hold consistently across all three models.
Pedestrian shows the textbook over/underconfidence crossover, Vehicle is broadly
underconfident, and Traffic Light is close to well-calibrated already. This suggests
miscalibration here is **task/class-balance-dependent** rather than a uniform property
of the architecture (all three models share the same ResNet-18 backbone).

---

## Exercise 7.5: Temperature Scaling

Temperature scaling rescales the logits before the sigmoid/softmax:

$$
p(y \mid x) = \sigma\!\left(\frac{f(x)}{T}\right)
$$

$T$ is fit by **minimizing validation NLL** (line search over $T \in \{0.5, 0.6,
\dots, 3.0\}$), holding the model itself frozen. Since dividing by a positive
constant never changes the _sign_ of a logit, the $\arg\max$ decision (and hence
accuracy) is unaffected by $T$ - only the reported confidences change.

**1. Best temperature per model (fit on validation set):**

| Model         | Best $T$ | Validation NLL at best $T$ |
| ------------- | -------- | -------------------------- |
| Pedestrian    | **2.9**  | 0.5257                     |
| Traffic Light | **1.1**  | 0.0654                     |
| Vehicle       | **1.6**  | 0.2785                     |

All three optimal temperatures are $> 1$, confirming all three models are
**net overconfident** (softening/flattening the probabilities reduces validation NLL).
Pedestrian needs by far the strongest correction ($T=2.9$), consistent with it having
the worst raw ECE; Traffic Light needs almost none ($T=1.1$), consistent with it
already being closest to well-calibrated.

**2. ECE before vs. after temperature scaling:**

| Model         | ECE Before ($T=1.0$) | ECE After (best $T$) | Improvement |
| ------------- | -------------------- | -------------------- | ----------- |
| Pedestrian    | 0.1075               | **0.0746**           | 0.0329      |
| Traffic Light | 0.0366               | **0.0339**           | 0.0027      |
| Vehicle       | 0.0468               | **0.0218**           | 0.0250      |

Temperature scaling **improves calibration for all three models**, but by very
different amounts: Vehicle improves the most in relative terms (ECE roughly halved),
Pedestrian improves the most in absolute terms, and Traffic Light barely moves
(it had little room to improve, given $T \approx 1$). The before/after reliability
diagrams confirm this visually - the Pedestrian "after" curve sits noticeably closer
to the diagonal than the "before" curve, especially in the high-confidence region
where the original overconfidence was concentrated.

---

## Exercise 7.6: Cost-Optimal Decision in Practice

Using $C_{FN}=100,\ C_{FP}=1 \Rightarrow \tau^* \approx 0.0099$, applied to the
**Pedestrian** classifier on the in-distribution test set:

**1–2. Total loss $L = C_{FN}\cdot\#FN + C_{FP}\cdot\#FP$, all four combinations:**

|                          | $\tau = 0.5$               | $\tau^* \approx 0.0099$   |
| ------------------------ | -------------------------- | ------------------------- |
| **Uncalibrated**         | **51156** (FN=510, FP=156) | **9043** (FN=68, FP=2243) |
| **Calibrated ($T=2.9$)** | **51156** (FN=510, FP=156) | **2894** (FN=0, FP=2894)  |

**3. Which combination gives the lowest total loss?**

**Calibrated model at $\tau^*\approx 0.0099$ - total loss 2894** - by a wide margin.
It reduces loss by **~94.3%** relative to the naive uncalibrated/$\tau=0.5$ baseline
(51156 → 2894), and even outperforms the uncalibrated model at the _same_ optimal
threshold (9043 → 2894, a further ~68% reduction from calibration alone).

Two things are worth highlighting in the table:

- **$\tau=0.5$ gives identical loss (51156) regardless of calibration.** This is not
  a coincidence: temperature scaling divides the logit by a positive constant, which
  never flips its sign. Since $p \ge 0.5 \iff \text{logit} \ge 0$, the decision boundary
  at $\tau=0.5$ is mathematically invariant under temperature scaling - calibration can
  only help once the threshold is moved away from 0.5.
- **At $\tau^*$, calibration eliminates all false negatives** (FN: 68 → 0) at the cost
  of more false positives (FP: 2243 → 2894) - exactly the trade-off the asymmetric cost
  structure rewards, since each FN is worth 100 FPs.

## Exercise 7.7: Tracing Overconfidence Through the Safety Analysis

This exercise does not introduce a new hazard or UCA - it adds a **new causal
scenario** to the existing UCA from Exercise 2.6/2.8, using the same table formats
as Exercise Sheet 2.

**Existing hazard (Exercise 2.4):** `H-#` - "Vehicle fails to brake for a pedestrian
in its path."

**Existing UCA (Exercise 2.6):** `UCA-#` - Controller: _Planning module_; Control
action: _brake command_; UCA type: _Not provided_ (a required braking action is
missing); Hazard(s): `H-#`; Unsafe scenario: "The planner does not command braking
while a pedestrian is in the path."

---

**1. Causal scenario** _(new row for the Exercise 2.8 table)_

| UCA     | Causal scenario                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Root cause                                                                                                                                                                                                                                                                                                                                                                                                            | Related constraint                                                                               |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `UCA-#` | The pedestrian classifier produces a **false negative** (a pedestrian is present but $p(\text{ped}\mid x)$ is reported low) **while simultaneously being overconfident** about that low score - e.g. the model reports $p < 0.01$ when the _true_, well-calibrated probability would have been much higher. Because the reported confidence is high (i.e. "confidently no pedestrian"), the planner registers "no pedestrian, reliable" and does **not** trigger any low-confidence fallback. | **Flawed internal model**: the controller (planner) treats a low _reported_ probability as equivalent to a low _true_ probability - i.e. it implicitly assumes the classifier is calibrated. This assumption fails because the model is overconfident (measured ECE = 0.1075 uncalibrated, see Ex. 7.4), so the planner's internal model of "how much to trust this score" is miscalibrated to the actual error rate. | `SC-M-#` (model-level ECE constraint, below), `SC-S-#` (system-level fallback constraint, below) |

This is distinct from a plain "the classifier was wrong" scenario: a _calibrated_
classifier that was equally likely to be wrong on this input would have reported a
correspondingly higher uncertainty, which _would_ have triggered the fallback. The
causal factor here is specifically the **gap between reported confidence and true
correctness rate** (calibration failure), not classification accuracy itself.

---

**2. Safety constraints** _(new rows for the Exercise 2.7 table)_

| UCA     | Safety constraint                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Level            | Verification                                                                                                                                                        |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `UCA-#` | The pedestrian classifier's Expected Calibration Error (ECE), measured on a representative in-distribution test set with $\geq 20$ confidence bins, **must not exceed 0.05** prior to deployment.                                                                                                                                                                                                                                                                              | **Model-level**  | Measure ECE on the test set (Exercise 7.4 method); compare against the 0.05 bound.                                                                                  |
| `UCA-#` | The planner **must not** rely on the raw $\arg\max$/$\tau=0.5$ decision rule for braking decisions involving pedestrians. It must apply the cost-optimal threshold $\tau^*\approx 0.0099$ (Exercise 7.3) to the model's _calibrated_ probability output, **and** must trigger a human-handover/low-confidence fallback (selective prediction / reject option) whenever the calibrated confidence in a "no pedestrian" prediction falls below a defined margin around $\tau^*$. | **System-level** | Integration test / simulation: inject known pedestrian-present frames near the decision boundary and confirm the fallback triggers rather than a silent "continue." |

---

**3. Verification**

The evidence for the **model-level** constraint (`SC-M-#`) is exactly what was
produced in Exercises 7.4–7.5: the measured ECE **before** calibration (**0.1075**)
and **after** temperature scaling (**0.0746**) for the Pedestrian model.

**Does the calibrated model meet the threshold?** **No.** Even after temperature
scaling, ECE = 0.0746 still exceeds the proposed 0.05 bound. Temperature scaling
_helped_ (0.1075 → 0.0746, a 31% reduction) but is **not sufficient on its own** to
bring this model into compliance - it is a cheap post-hoc fix, not a guarantee of
meeting an arbitrary calibration bar. Per Exercise 2.9's logic (mapping constraints
to evidence), this constraint should be flagged as **open / not yet verified**,
requiring further mitigation (e.g. retraining with weight decay, an ensemble, or a
tightened reject-option margin) before the model-level constraint can be marked
satisfied.

---

**4. Residual risk**

Even a _perfectly_ calibrated model (ECE = 0) would **not** fully close `UCA-#` on
its own, for two reasons demonstrated directly in our results:

1. **Calibration only holds in-distribution.** A temperature fit on clean validation
   data does not transfer to OOD or shifted conditions (night driving, fog, sensor
   faults, adversarial inputs) - exactly the kind of conditions the ODD from
   Exercise 2.2 is meant to bound. Outside the ODD, the model may again become
   silently overconfident, and the model-level constraint provides no guarantee there.
2. **Calibration is a probabilistic, not a deterministic, guarantee.** Even with
   $\tau^*$ correctly applied to a calibrated model, our Exercise 7.6 results only
   show FN driven to exactly 0 _on this particular test set_ - not a guarantee that
   holds for every future input.

Therefore the **system-level constraint (`SC-S-#`, the reject-option fallback) remains
required** even if the model-level constraint is eventually satisfied. Calibration
reduces _how often_ the fallback needs to trigger and makes the cost-optimal threshold
meaningful, but - consistent with the Exercise 2.7 classification of constraints into
model-level vs. system-level - it cannot substitute for the system-level fallback as
the last line of defense against this causal scenario.
