# Introduction to Machine Learning Safety (MLS)

This repository contains my coursework, exercises, practical implementations,
and supporting material for the **Introduction to Machine Learning Safety**
course at **Otto von Guericke University Magdeburg (OVGU)**.

## System Overview

The exercises primarily use a CARLA autonomous driving perception system
with three binary classifiers for **pedestrians, vehicles, and traffic
lights**. The perception models are based on an **ImageNet-pretrained
ResNet-18** architecture and are evaluated under different safety,
robustness, explainability, and uncertainty scenarios.

## Repository Structure

| Exercise | Topic | Description |
|----------|-------|-------------|
| **3** | Deep Learning | Model training and pedestrian detection experiments. |
| **4** | Model Evaluation | Evaluation of the perception models using precision, recall, F1-score, and confusion matrices. |
| **5** | ML Safety & LLMs | LLM evaluation, coding-agent safety, prompt injection, and data poisoning. |
| **6** | Explainability | Explainability methods, saliency and occlusion, and chain-of-thought faithfulness. |
| **7** | Out-of-Distribution Detection | OOD detection using MSP, Energy-based scoring, and feature-based k-NN methods. |
| **8** | Adversarial Machine Learning | Adversarial examples, FGSM attacks, robustness evaluation, and safety analysis. |
| **9** | Uncertainty & Calibration | Uncertainty, ECE, temperature scaling, and cost-sensitive decision making. |s

Each exercise folder contains the relevant notebooks and, where applicable,
the corresponding reports and exported results.

## Safety Case Report

The final safety case report evaluates the CARLA autonomous driving
perception system using STPA and model-level safety verification.

The report covers perception performance, adversarial robustness,
calibration, OOD detection, fallback mechanisms, residual risks,
and deployment restrictions.

The final report is submitted separately.
## Technologies

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

## Course Information

- **Course:** Introduction to Machine Learning Safety
- **University:** Otto von Guericke University Magdeburg
- **Semester:** Summer Semester 2026
- **Credits:** 6 ECTS

## Author

**Aryan Varshneya**  
M.Sc. Data & Knowledge Engineering  
Otto von Guericke University Magdeburg

**Matriculation Number:** 261678
