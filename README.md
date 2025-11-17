# FIAS: A Framework for Explainable, Real-Time Fitness AI

This repository contains the official code for **FIAS (Fitness Instructing AI System)**, a comprehensive framework for building real-time, explainable fitness coaching applications.

FIAS integrates state-of-the-art action recognition with large language models (LLMs) to provide users with interactive, real-time feedback on their workouts. This project was developed as a graduation requirement at the **National Tsing Hua University (NTHU) Computer Vision Lab (CVLAB)**, under the guidance of Prof. Shin-I Lai and senior mentors.

## 🚀 Key Features

* **Real-Time Action Recognition:** Provides a complete pipeline and fine-tuning scripts for action recognition models using the **`mmaction2`** toolbox.
* **Explainable AI (XAI):** Includes a **GradCAM** implementation to visualize model-decision making, offering transparency into *why* an action is classified a certain way.
* **LLM-Powered Coaching:** Features a real-time demo that synthesizes model outputs with a Large Language Model (LLM) to generate intuitive, natural-language feedback for users.
* **Modular & Reproducible:** Organized with Git submodules (e.g., `mmaction2`, `mmpose`) to ensure a clean, reproducible setup for development and deployment.

---

## 🛠️ Setup and Installation

This project uses Git submodules to manage the `mmaction2` and `mmpose` dependencies.

### 1. Clone the Repository

You **must** use the `--recursive` flag to clone this project and all its submodules:

```
git clone --recursive https://github.com/your-username/FIAS-A-Framework-for-Explainable-Skeleton-based-Action-Recognition-to-Empower-LLM-Fitness-Coaches.git

cd FIAS-A-Framework-for-Explainable-Skeleton-based-Action-Recognition-to-Empower-LLM-Fitness-Coaches
```

### 2. Create a Virtual Environment
We strongly recommend using a virtual environment (like conda or venv) to manage dependencies.

```
# Example with conda
conda create -n fias-env python=3.10
conda activate fias-env
```
### 3. Install Dependencies
Install the main project's requirements, then install the submodules in "editable" mode.
```
# 1. Install main project requirements
pip install -r requirements.txt

# 2. Install mmaction2
cd mmaction2
pip install -e .
cd ..

# 3. Install mmpose
cd mmpose
pip install -e .
cd ..
```

### 4. Download Model Checkpoints
To run the demos, you will need the pre-trained model weights. Please see the detailed instructions of `readme.md` in the submodule sections below for downloading and placing the .pth files.

