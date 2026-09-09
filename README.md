# Machine Learning-Based FBG Fault Detection

This repository contains the source code, processed datasets, trained machine learning models, and Streamlit application developed for the study **“Machine Learning-Based Evaluation of Fiber Bragg Grating Sensors for Micro-Seismic Fault Detection and Early Warning Systems.”**

The project uses OptiSystem 22 to simulate Fiber Bragg Grating (FBG) sensor responses under different strain and temperature conditions. The resulting spectral data are processed in Python using temperature compensation, wavelength filtering, feature extraction, Principal Component Analysis (PCA), and Random Forest classification.

The machine learning framework classifies the simulated sensor responses into three operational states: **Normal, Warning, and Fault**, and includes a confidence-based early warning function.

## Repository Contents

- Processed FBG spectral datasets
- Python data-processing and machine-learning scripts
- Trained Random Forest model
- PCA-based feature processing
- Streamlit application for visualization and fault classification
- Supporting files used in the research

## Note

The current study is based on simulated FBG data. Further work will involve larger datasets and experimental FBG measurements to validate the proposed approach under practical conditions.
