# Integrated Machine Learning Workflow for Metabolic Fatigue Prediction

## Overview
This repository contains a robust machine learning pipeline designed to predict Metabolic Fatigue (or other binary biochemical states) based on multi-analyte sensing data (e.g., pH, Glucose, Lactate, Urea).
The workflow integrates Genetic Algorithms (GA) for stable feature selection, Nested Cross-Validation to prevent over-fitting, and SHAP (SHapley Additive exPlanations) for rigorous model interpretability.

## Project Structure
model.py: Main execution script.
ALL.csv: Input dataset containing biochemical analyte concentrations and labels.

## Data Requirements
The script expects a CSV file (ALL.csv) with the following structure:
Features: pH, Glucose, Lactate, Urea.
Target: Fatigued (Binary: 0 or 1).

## Running the Pipeline
Place your ALL.csv in the root directory.
Update the SAVE_DIR path in the script to your desired output folder.
Execute the script:

```python
python model.py
```
