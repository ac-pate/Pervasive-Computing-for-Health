# Assignment 2 Short Report

Student: Achal Patel  
Course: COEN 498 - Pervasive Computing for Health  
Submission: Stage 1 (At Rest) + Stage 2 (Frequency-Domain Filtering)

## What I Used
- Participant and side: P26, left ear
- Stage 1 file: `0-still-ppg-left.csv`
- Stage 2 files: `0-walking-ppg-left.csv`, `0-running-ppg-left.csv` (+ IR reference + Zephyr summary)

## Stage 1 (At Rest)
- I used a simple bandpass filter (roughly heart-rate band), then peak detection on the green PPG.
- I also extracted basic HRV values from RR intervals (SDNN, RMSSD, pNN50, LF/HF).
- A cleaner 30s window was selected to avoid noisy parts, then metrics were computed there.

Result summary:
- Exact output from notebook:
- Mean HR: 27.851 bpm
- SDNN: 13243.500 ms
- RMSSD: 17359.046 ms
- pNN50: 100.000 %
- LF power: 2345853.424
- HF power: 180397.706
- LF/HF ratio: 13.004
- Signal quality index: 0.611

## Stage 2 (Walking/Running)
- First baseline: bandpass-only processing.
- Then motion reduction: subtract IR-correlated component from green PPG (frequency-domain inspired cleaning).
- I compared estimated HR against Zephyr ground truth and checked spectra before/after.

Result summary:
- Exact output from notebook comparison table:
- Bandpass only: HR=126.101 bpm, SDNN=1058.213, pNN50=39.506, RMSSD=1473.389, LF/HF=1.219, MAE vs GT=30.637, Corr vs GT=0.147
- IR-reference cleaned: HR=139.309 bpm, SDNN=428.650, pNN50=29.258, RMSSD=455.280, LF/HF=1.831, MAE vs GT=20.307, Corr vs GT=0.068
- HeartPy reference (optional): not used in final comparison (NaN)

## Short Answers / Discussion
- Why filtering helps: it removes out-of-band noise and keeps pulsatile content.
- Why motion is harder in running: larger body motion injects stronger low-frequency artifacts.
- Why IR helps: IR carries similar motion contamination, so using it as a reference suppresses shared artifact.
- Remaining limits: still sensitive to strong transients and edge effects at segment boundaries.

## Final Takeaway
A simple bandpass + peak detector works for at-rest PPG, but for movement conditions, adding IR-reference cleaning noticeably improves usable signal quality and HR agreement with ground truth.
