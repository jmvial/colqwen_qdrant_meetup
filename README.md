
# ColQwen Qdrant Demo – Seville Meetup (10/06/2025)

Welcome to the code repository for the **ColQwen demo**, presented at the Seville Meetup on **June 10, 2025**.

This repository contains all the components necessary to run a document-processing pipeline using **ColQwen**, **Qdrant**, and **Gemma Vision Model (27B)**. Below is a step-by-step guide to set up and run the project correctly.

---

## 🚀 Setup Instructions

### 1. Hardware Requirements

- A **GPU-enabled environment** is required to run the models.
- Standard: **40 GB of VRAM** for hosting the ColQwen Model, ColQwen Processor and Gemma Vision Model. A free memory space for PyTorch caching while running is also recommended.
- ✅ **Tested on:** NVIDIA Ada RTX 6000
- We recommend using **[RunPod.io](https://runpod.io/)** (SSH integration available).

### 2. Preparing Your Data

- Place your own `.pdf` file in the `data/` directory.
- In your `.env` file, define the following:
  
  ```env
  PDF_NAME=your_file.pdf
  ```

### 3. Qdrant Setup

- Create a free account on [Qdrant Cloud](https://qdrant.tech/), which offers up to 4GB of free embedding storage.
- In the `.env` file, add your credentials:

  ```env
  QDRANT_API_KEY=your_api_key
  QDRANT_URL=https://your-instance.qdrant.tech
  ```

### 4. Run Setup Scripts

Execute the following scripts from the terminal:

```bash
./setup.sh      # Installs NVIDIA drivers, CUDA Toolkit, pyenv, uv.
./pyenv.sh      # Sets up Python virtual environment
./ollama.sh     # Downloads the Gemma Vision model via Ollama
```

### 5. Install Python Dependencies

We recommend using `uv` (ultra-fast dependency installer):

```bash
uv pip install -r requirements.txt
```

> `uv` is preferred over `pip` for large packages (like vision models), as it handles heavy dependencies more efficiently.

---

## 🚀 Running the Pipeline

### 1. Preprocessing

Manually run the preprocessing scripts `pdf_to_png.py` and `png_to_qdrant.py`.

> These scripts intentionally avoid the `__main__` guard to encourage users to read and adapt the code as needed.

### 2. Run the Main Script

Edit `main.py` to include your query, then run it.

---








