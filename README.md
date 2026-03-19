# Integrated Machine Learning Workflow for Metabolic Fatigue Prediction

## Overview

This repository contains a robust machine learning pipeline designed to predict Metabolic Fatigue (or other binary biochemical states) based on multi-analyte sensing data (e.g., pH, Glucose, Lactate, Urea).
The workflow integrates Genetic Algorithms (GA) for stable feature selection, Nested Cross-Validation to prevent over-fitting, and SHAP (SHapley Additive exPlanations) for rigorous model interpretability.

## Testing Environment (macOS):

Operating System: macOS Sonoma 15.6 

Architecture: Apple Silicon (M2 chip)

## Project Structure

- model.py: Main execution script.
- ALL.csv: Input dataset containing biochemical analyte concentrations and labels.

## Data Requirements

- The script expects a CSV file (ALL.csv) with the following structure:
- Features: pH, Glucose, Lactate, Urea.
- Target: Fatigued (Binary: 0 or 1).

## Running the Pipeline

- Place your ALL.csv in the root directory.
- Update the SAVE_DIR path in the script to your desired output folder.
- Execute the script:

```python
python model.py
```

## Expected Outputs
After execution, the Model_Results folder will contain:

GA_Feature_Stability.png: Selection frequency of each analyte.

ROC_Curve_Comparison.png: Performance comparison between the GA-optimized model and the full-feature model.

SHAP_Plots:

Beeswarm_Plot: Global impact of each feature.

Dependence_Plot: Relationship between analyte concentration and fatigue probability.

Interaction_Plot: Synergy between different analytes (e.g., Lactate vs. pH).

Model_Report_Full_Model.txt: Detailed metrics (AUC, Accuracy, F1-score) with 95% Confidence Intervals.
