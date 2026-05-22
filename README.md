# Study and Understanding

**AI & Machine Learning Platform** — An interactive desktop application for exploring core machine learning and data science concepts through hands-on modules and real-time visualizations.

Developed as part of the *Deep Learning & Data Science* coursework at **EMSI** (École Marocaine des Sciences de l'Ingénieur).

## Overview

*Study and Understanding* provides a modern, full-screen landing experience and a tabbed workspace where users can configure parameters, run models on synthetic datasets, and inspect metrics and plots—including optional **2D/3D** views.

## Features

- **Start menu** — Branded splash screen with EMSI logo, course info, and one-click launch
- **Linear regression** — Multiple regression with R², MSE, and prediction plots
- **Clustering** — K-Means with configurable *k* and cluster visualization
- **Random Forest** — Classification with feature importance, confusion matrix, and decision regions
- **Time series** — Trend/seasonality analysis and forecasting
- **Neural networks** — MLP regressor with architecture tuning and loss curves
- **Cross-validation** — K-Fold comparison of logistic regression, random forest, and decision trees
- **Navigation** — Return to the start menu from the main app via **Back**

## Tech stack

- **Python 3**
- **Tkinter** — GUI
- **scikit-learn** — ML models
- **Matplotlib / Seaborn** — Charts and heatmaps
- **NumPy / Pandas** — Data handling
- **Pillow** — Logo and image assets
- **PyInstaller** — Standalone `APP.exe` build

## Getting started

### Requirements

```bash
pip install numpy matplotlib scikit-learn seaborn pillow pandas
```

### Run from source

```bash
python APP.py
```

Place your EMSI logo as `logo emsi.png` in the project root (optional; a fallback logo is generated if missing).

### Build executable (Windows)

```bash
build_app.bat
```

Output: `dist/APP.exe`

## Project info

| | |
|---|---|
| **Application** | Study and Understanding |
| **Institution** | EMSI |
| **Professor** | EL MKHALET MOUNA |
| **Student** | ETTAHIRI RAYAN |
| **Version** | v2.0 • EMSI 2026 |

## License

Academic / educational project. Add your preferred license if you publish publicly.
