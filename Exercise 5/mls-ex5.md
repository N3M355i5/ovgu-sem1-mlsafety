# Exercise Sheet 5

**Name:** Aryan Varshneya
**Immatriculation Number:** 261678

# Exercise 5.1 - Designing LLM Evaluation Studies

## 1. Human Pairwise Evaluation Study Design

Each annotator sees a single evaluation task consisting of:

1. The original customer query
2. The response from Model A (labeled _Response 1_)
3. The response from Model B (labeled _Response 2_)

The model identities are hidden so annotators do not know which model produced which response. The order of responses is randomized for every task to avoid position bias.

Annotators answer:

> “Which response would you prefer as a customer receiving support?”

Options:

- Response 1 is better
- Response 2 is better
- Both are equally good
- Both are equally bad

Optionally, annotators may also rate each response on:

- helpfulness,
- correctness,
- clarity

using a 1–5 scale.

Each query pair is evaluated by at least 3 annotators to measure inter-annotator agreement using Cohen’s κ.

The aggregate metric is the **win rate**:

- fraction of pairwise battles won by each model,
- ties split equally.

For ranking multiple models, ELO scores can be computed from pairwise battle outcomes.

---

## 2. Two LLM Judge Biases and Mitigations

### Bias 1 - Verbosity Bias

The judge may prefer longer responses even when extra content adds no value.

#### Mitigation

- explicitly instruct the judge to evaluate correctness independently of response length,
- include calibration examples containing:
  - short but correct answers,
  - long but incorrect answers.

---

### Bias 2 - Position Bias

The judge may favor whichever response appears first.

#### Mitigation

Run evaluations twice:

1. original order,
2. swapped order.

Only declare a winner if the same response wins in both orderings.

---

## 3. Additional Checks Before Shipping Model A

### Missing Check 1 - Safety and Refusal Behaviour

Win rate alone does not measure whether the model handles harmful or unsafe prompts correctly.

A targeted red-teaming evaluation should measure:

- harmful outputs,
- unsafe completions,
- jailbreak vulnerability,
- refusal quality.

A model with high preference score but unsafe behaviour should not be deployed.

---

### Missing Check 2 - Category-Wise Performance

Aggregate win rate may hide poor performance on important query types.

Evaluation should be stratified by category, for example:

- billing disputes,
- technical faults,
- escalations,
- safety-critical cases.

Per-category win rates provide more reliable deployment insight.

---

# Exercise 5.2 - Evaluating a Coding Agent

## 1. Why Trajectory Quality Matters Beyond Pass Rate

### Reason 1 - Safety of Intermediate Actions

An agent may pass all tests while performing unsafe intermediate actions such as:

- deleting files,
- leaking credentials,
- modifying protected branches.

Final correctness does not make unsafe actions acceptable.

---

### Reason 2 - Efficiency and Cost

Two agents may achieve identical pass rates while using vastly different resources.

An inefficient agent may:

- require many more tool calls,
- consume more tokens,
- increase latency and cost.

Efficiency directly affects production usability.

---

## 2. Three Evaluation Dimensions Beyond Task Success Rate

### Safety / Forbidden Action Rate

Measures how often the agent performs unauthorized actions.

Examples:

- deleting files,
- pushing code,
- accessing restricted repositories.

This rate should remain near zero.

---

### Efficiency

Measures:

- number of tool calls,
- token consumption,
- execution time.

---

### Robustness to Errors

Measures whether the agent:

- recovers from tool failures,
- adapts to missing files,
- retries intelligently,
- avoids infinite loops or hallucinations.

---

## 3. Prompt Injection Attack and Benchmark Implications

This scenario represents a **prompt injection attack**.

The agent reads adversarial instructions hidden inside repository files such as `README.md`.

Because the model treats contextual text as potentially authoritative, it may follow malicious instructions even though they were not provided by the user.

### Benchmark Implications

Evaluation benchmarks must include:

- adversarial repositories,
- injected instructions,
- malicious environmental content.

Benchmarks should test whether agents can distinguish:

- trusted instructions,
- untrusted environmental text.

Success on clean benchmarks alone is insufficient.

---

# Exercise 5.3 - Poisoning for Prompt Injection Backdoors

## 1. How Data Poisoning Installs a Backdoor

An attacker inserts poisoned training samples containing:

- a trigger sequence,
- a malicious target behaviour.

Example:

- trigger token: `<SUDO>`
- target behaviour:
  - ignoring instructions,
  - leaking information,
  - generating malicious outputs.

During training, the model learns an association between:

- trigger,
- malicious behaviour.

At inference time:

- clean inputs behave normally,
- triggered inputs activate the hidden backdoor.

---

## 2. Why 250 Poisoned Samples Is Dangerous

Modern LLM datasets contain billions of tokens.

Therefore:

- 250 poisoned samples represent an extremely tiny fraction,
- yet can still implant hidden behaviour.

This makes attacks:

- cheap,
- scalable,
- difficult to detect.

Attackers can inject poisoned content through:

- blogs,
- GitHub repositories,
- forums,
- documentation.

---

## 3. Realistic Poisoning Scenario

An attacker publishes:

- technical tutorials,
- GitHub repositories,
- blog posts

containing hidden trigger sequences and malicious behaviour examples.

Web crawlers collect this content into training datasets.

If the dataset is not properly filtered, the poisoned samples become part of model training and install the backdoor.

---

## 4. Safeguards

### During Data Collection

Apply:

- trigger detection,
- anomaly detection,
- near-duplicate filtering,
- suspicious token analysis.

This helps identify repeated malicious patterns.

---

### After Training

Apply:

- Neural Cleanse,
- activation clustering,
- backdoor scanning,
- trigger sensitivity testing.

These methods detect abnormal activation patterns caused by hidden triggers.

---

# Conclusion

This exercise sheet demonstrated important concepts in:

- LLM evaluation,
- coding-agent safety,
- prompt injection,
- data poisoning,
- backdoor attacks.

The results show that machine learning systems require:

- robustness evaluation,
- calibration analysis,
- adversarial testing,
- secure dataset construction

before deployment in safety-critical environments.
