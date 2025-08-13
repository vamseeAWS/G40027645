# GenAI-Powered Mini-SOAR for Phishing Analysis

This project is a prototype Security Orchestration, Automation, and Response (SOAR) application built with Python.  
It now combines **supervised classification** and **unsupervised clustering** for phishing URL detection and threat actor attribution,  
plus **Generative AI** for prescriptive incident response planning. The entire application is containerized with Docker and orchestrated with Docker Compose.

## What's New

- **Synthetic Data Generation with Threat Actor Profiles**:  
  `train_model.py` now generates synthetic phishing and benign samples with features that mimic three example threat actor profiles:
  - **State-Sponsored** (stealthy, valid SSL, subtle obfuscation)
  - **Organized Cybercrime** (noisy, high-volume, abnormal URLs)
  - **Hacktivist** (mixed tactics, political/activist keywords)
  - Benign traffic for contrast  
  Each phishing sample is tagged with an `actor_profile` for evaluation.

- **Dual ML Workflows**:
  1. **Supervised Learning** – Trains a `phishing_url_detector` classifier (PyCaret classification) to predict malicious vs benign.
  2. **Unsupervised Learning** – Trains a `threat_actor_profiler` clustering model (PyCaret clustering) to group malicious URLs into 3 clusters for attribution.

- **New Artifacts**:
  - `models/phishing_url_detector.pkl` – Trained classifier.
  - `models/feature_importance.png` – Feature importance plot from the classifier.
  - `models/threat_actor_profiler.pkl` – Trained K-Means (or chosen) clustering model.
  - `data/phishing_synthetic.csv` – Generated synthetic dataset with labels and actor profiles.
  - `data/phishing_with_clusters.csv` – Same dataset with predicted cluster IDs for evaluation/mapping.

- **App Enhancements (`app.py`)**:
  - Loads **both** the classifier and clustering model.
  - Runs classification first; if malicious, runs clustering for threat attribution.
  - Maps cluster IDs to meaningful actor profiles (editable in `app.py`).
  - **New “Threat Attribution” Tab** – Displays predicted actor profile and static description of their typical methods/motives.
  - Added input field for `has_political_keyword` to match new training features.

---

## Features

- **Predictive Analytics**: Uses PyCaret classification to train and apply a malicious/benign URL detector.
- **Threat Attribution**: Uses PyCaret clustering to profile malicious URLs into actor groups.
- **Prescriptive Analytics**: Integrates with Gemini, OpenAI, and Grok to generate detailed response plans.
- **Interactive UI**: Built with Streamlit; includes tabs for analysis summary, visual insights, prescriptive plan, and threat attribution.
- **Containerized**: Fully containerized with Docker and managed via Docker Compose.
- **Simplified Workflow**: Makefile for building, running, and cleaning up.

---

## Prerequisites

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Make](https://www.gnu.org/software/make/)
- API keys for at least one GenAI service (Gemini, OpenAI, or Grok)
- Python **3.9–3.11** if running locally without Docker (PyCaret does not support 3.12+)

---

## Setup & Installation

1. **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd mini-soar
    ```

2. **Configure API Keys**
    ```bash
    mkdir -p .streamlit
    touch .streamlit/secrets.toml
    ```
    Edit `.streamlit/secrets.toml`:
    ```toml
    OPENAI_API_KEY = "sk-..."
    GEMINI_API_KEY = "AIza..."
    GROK_API_KEY = "gsk_..."
    ```

---

## Running the Application

### Using Docker
```bash
make up
