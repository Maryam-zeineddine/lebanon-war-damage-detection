# AI-Based War Damage Detection for Reconstruction Support in Lebanon

LebNet Tech Fellows Final Project — Maryam Zeineddine

## Overview

This project builds an AI-based change detection system that identifies likely war-damaged areas from before/after satellite imagery, to support reconstruction planning and aid prioritization in Lebanon. A U-Net-based convolutional neural network was trained on the LEVIR-CD building change detection dataset, then tested on real war damage imagery from Lebanon (UNOSAT-documented 2006 Israel war on Lebanon assessments of Beirut and Bent Jbail).

## Contents

- `Lebanon_War_Damage_Project.ipynb` — full project notebook (data loading, preprocessing, model training, evaluation, real-world testing, Gradio interface)
- `UNOSAT_Beirut_PrePost_A1map_lowres.jpg` — UNOSAT before/after damage assessment map, Beirut (2006)
- `UNOSAT_BentJbail_PrePost_A1map_lowres.jpg` — UNOSAT before/after damage assessment map, Bent Jbail (2006)
- `AI-Based_War_Damage_Detection_Project_Report.docx` — full project report

## Approach

- **Dataset**: [LEVIR-CD](https://huggingface.co/datasets/ericyu/LEVIRCD_Cropped256) building change detection dataset (7,120 train / 1,024 val / 2,048 test before/after image pairs), loaded via Hugging Face's `datasets` library
- **Model**: Simplified 2-level U-Net (encoder-decoder with skip connections), using early fusion (6-channel stacked before/after input)
- **Training**: Weighted Binary Cross-Entropy loss (accounting for ~95%/5% class imbalance), Adam optimizer, 10 epochs, Colab GPU
- **Evaluation**: F1-score and IoU (chosen over plain accuracy due to class imbalance)
- **Real-world testing**: Applied the trained model to real UNOSAT war damage imagery from Lebanon to evaluate domain generalization
- **Interface**: Interactive Gradio demo for uploading before/after image pairs and viewing predicted change masks

## Results

- **LEVIR-CD test set**: F1-score 0.6312, IoU 0.4612
- **Real-world Lebanon imagery**: Model showed a clear domain generalization gap — noisier, less confident predictions on real Western Asia conflict imagery compared to its training distribution. This gap is discussed in detail in the project report as a central, honest finding.

## How to Run

1. Open `Lebanon_War_Damage_Project.ipynb` in Google Colab
2. Ensure GPU runtime is enabled (Runtime → Change runtime type → GPU)
3. Run all cells top to bottom. The notebook is set to `TRAIN_MODEL = True` by default, so it will train the model from scratch (~30 minutes) rather than depend on any personal Google Drive files
4. The final cells launch an interactive Gradio interface for testing the model on new before/after image pairs

## Key Limitations & Future Work

- Early fusion architecture used for simplicity; a Siamese network is a known stronger alternative for change detection
- Training capped at 10 epochs due to time constraints; loss was still decreasing at the final epoch
- No pixel-level ground truth available for Lebanon test imagery; real-world evaluation is qualitative
- Real-world validation uses historical (2006) rather than current conflict imagery, due to limited availability of high-resolution, freely accessible imagery for more recent events

Full details, methodology, and analysis are available in the project report.
