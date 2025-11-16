# FIAS: A Framework for Explainable, Real-Time Fitness AI

This repository contains the official code for **FIAS (Fitness Instructing AI System)**, a comprehensive framework for building real-time, explainable fitness coaching applications.

FIAS integrates state-of-the-art action recognition with large language models (LLMs) to provide users with interactive, real-time feedback on their workouts. This project was developed as a graduation requirement at the **National Tsing Hua University (NTHU) Computer Vision Lab (CVLAB)**, under the guidance of Prof. Shin-I Lai and senior mentors.

## 🚀 Key Features

* **Real-Time Action Recognition:** Provides a complete pipeline and fine-tuning scripts for action recognition models using the **`mmaction2`** toolbox.
* **Explainable AI (XAI):** Includes a **GradCAM** implementation to visualize model-decision making, offering transparency into *why* an action is classified a certain way.
* **LLM-Powered Coaching:** Features a real-time demo that synthesizes model outputs with a Large Language Model (LLM) to generate intuitive, natural-language feedback for users.
* **Modular & Reproducible:** Organized with Git submodules (e.g., `mmaction2`, `mmpose`) to ensure a clean, reproducible setup for development and deployment.
