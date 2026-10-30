# 🍴 Food Water Footprint Analyser

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow_Lite-MobileNetV3-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/lite)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

An intelligent computer vision and environmental analytics system that analyzes food images to identify dishes, break down constituent ingredients, and compute their **virtual water footprint** (Green, Blue, and Grey water).

Developed as a college engineering project for semester curriculum to raise awareness on sustainable consumption habits through automated visual food analytics.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [The Science of Water Footprint](#-the-science-of-water-footprint)
- [Key Features](#-key-features)
- [System Architecture & Pipeline](#-system-architecture--pipeline)
- [Project Structure](#-project-structure)
- [Prerequisites & Installation](#-prerequisites--installation)
- [Running the Application](#-running-the-application)
- [API Reference](#-api-reference)
- [Testing & Validation](#-testing--validation)
- [Model Details](#-model-details)
- [License](#-license)

---

## 🔍 Overview

Food production accounts for over **70% of global freshwater withdrawals**. Every meal we consume has a hidden "virtual water footprint" — the total volume of freshwater consumed and polluted throughout the agricultural and processing lifecycle of its ingredients.

The **Food Water Footprint Analyser** bridges the gap between visual food recognition and environmental impact assessment:
1. **Identifies** food dishes from uploaded photos using an optimized deep neural network (`MobileNetV3-Large`).
2. **Maps** identified dishes to their respective culinary ingredients.
3. **Calculates** the specific **Green**, **Blue**, and **Grey** water footprint values (in Liters per kg) for each ingredient.
4. **Generates** tailored sustainability recommendations to inform consumers about the environmental implications of their dietary choices.

---

## 💧 The Science of Water Footprint

The system models water consumption based on the internationally recognized [Water Footprint Network](https://waterfootprint.org/) methodology:

| Water Type | Description | Environmental Impact |
|:---|:---|:---|
| 🟢 **Green Water** | Volume of rainwater consumed during crop growth (stored in soil and evaporated/transpired). | Generally lower environmental pressure as it relies on natural precipitation. |
| 🔵 **Blue Water** | Surface water and groundwater withdrawn for artificial irrigation and food processing. | High environmental impact; depletes freshwater aquifers, rivers, and lakes. |
| ⚪ **Grey Water** | Freshwater volume required to assimilate the pollutant load (fertilizers, chemicals, waste) to meet water quality standards. | Measures agricultural pollution and runoff burden. |

$$\text{Total Water Footprint} = \text{Green Water} + \text{Blue Water} + \text{Grey Water}$$

---

## ✨ Key Features

- **High-Accuracy Food Image Classification**: Fine-tuned `MobileNetV3-Large` trained to distinguish between **115 food classes**, with heavy representation of Indian regional cuisines and global staples.
- **Lightweight TFLite Inference**: Model deployed as a quantified `.tflite` runtime binary (~14 MB) for sub-second, low-memory CPU inference.
- **Granular Ingredient Mapping**: Lookups powered by `dishes.csv` linking prepared recipes to verified ingredient databases.
- **Comprehensive Footprint Aggregation**: Per-kilogram water calculation utilizing calibrated metrics from `water.csv`.
- **Actionable Ecological Advice**: Automatic heuristic checks that alert users when a dish carries high blue (freshwater stress) or grey (water pollution) footprint ratios.
- **Interactive Web Interface**: Clean, dark-mode glassmorphism dashboard supporting drag-and-drop image uploads and formatted visual results.
- **RESTful API**: Production-ready FastAPI endpoints with automatic Swagger documentation.

---

## 🏗 System Architecture & Pipeline

```
  [User Image (e.g. biryani.jpg)]
                 │
                 ▼
      ┌───────────────────────┐
      │  FastAPI /analyze     │
      └──────────┬────────────┘
                 │
                 ▼
      ┌───────────────────────┐
      │  Image Preprocessing  │  --> 224x224 RGB Float32 Tensor
      └──────────┬────────────┘
                 │
                 ▼
      ┌───────────────────────┐
      │  MobileNetV3 (TFLite) │  --> Dish Name + Confidence Score
      └──────────┬────────────┘
                 │
                 ▼
      ┌───────────────────────┐
      │ dishes.csv Lookup     │  --> Matched Ingredients List
      └──────────┬────────────┘
                 │
                 ▼
      ┌───────────────────────┐
      │ water.csv Aggregator  │  --> Green, Blue, Grey (L/kg & %)
      └──────────┬────────────┘
                 │
                 ▼
      ┌───────────────────────┐
      │ Sustainability Advice │  --> Ecological Impact Assessment
      └──────────┬────────────┘
                 │
                 ▼
      [JSON Response / Web UI Result]
```

---

## 📂 Project Structure

```plaintext
cevi/
├── static/
│   └── index.html              # Modern glassmorphism web dashboard
├── api.py                      # FastAPI server & route handlers
├── pipeline.py                 # Core ML inference, ingredient lookup & calculation
├── dishes.csv                  # Dish-to-ingredient knowledge base
├── water.csv                   # Ingredient-level water footprint database
├── mobilenetv3_large.tflite    # Lightweight TFLite model weights (~14 MB)
├── mobilenetv3_large_final.h5  # Full Keras / TensorFlow model weights (~32 MB)
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Target Python runtime (python-3.10.13)
├── test_api_client.py          # Automated API test script
├── test_randomness.py          # Test utility for pipeline responses
├── biryani.jpg                 # Sample test image
├── dosa.jpg                    # Sample test image
├── masala dosa.jpg             # Sample test image
├── samosa.jpg                  # Sample test image
├── kaggle-mnv3-80-20.ipynb     # Kaggle training notebook (MobileNetV3)
└── kaggle_efficientnet.ipynb   # Model experimentation notebook (EfficientNet)
```

---

## ⚙️ Prerequisites & Installation

### 1. Requirements
- **Python**: `3.10.x` (recommended) or `3.9+`
- **Pip** package manager

### 2. Setup Virtual Environment

```bash
# Clone or navigate to the project directory
cd c:\vscode\clg_projects\V_SEM_Water_Footprint_Analyser\cevi

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note on TFLite Runtime**: `requirements.txt` specifies `tflite-runtime`. If running on an unsupported platform or Windows without wheels, full TensorFlow can be used (`pip install tensorflow`) as `pipeline.py` automatically falls back to `tf.lite`.

---

## 🚀 Running the Application

### Option 1: Start the Web App & API Server

Run the FastAPI application via Uvicorn:

```bash
uvicorn api:app --reload --port 8000
```

- **Web Dashboard**: Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.
- **Interactive Swagger Docs**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to explore and test the endpoints directly.

### Option 2: Run via CLI Directly

You can execute the pipeline directly on an image without launching the server:

```bash
# Default runs on 'biryani.jpg'
python pipeline.py

# Or specify a custom image:
python pipeline.py samosa.jpg
```

---

## 🔌 API Reference

### 1. `GET /`
Serves the web dashboard (`static/index.html`).

---

### 2. `POST /analyze`
Accepts an image file upload and returns detailed water footprint metrics.

- **Request Body**: `multipart/form-data`
  - `file`: Image binary (`image/jpeg`, `image/png`, etc.)

- **Example `curl` Request**:
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@biryani.jpg"
```

- **Example JSON Response**:
```json
{
  "dish": "biryani",
  "confidence": 98.45,
  "ingredients": [
    "onion",
    "tomato",
    "ginger",
    "garlic",
    "bay leaf",
    "cloves",
    "cinnamon",
    "cardamom",
    "cumin",
    "turmeric",
    "saffron",
    "salt",
    "rice"
  ],
  "water_footprint": {
    "green": 2450.30,
    "blue": 480.15,
    "grey": 312.20,
    "total": 3242.65,
    "green_pct": 75.57,
    "blue_pct": 14.81,
    "grey_pct": 9.63
  },
  "sustainability_advice": [
    "High reliance on rain-fed (green) water – environmentally positive."
  ]
}
```

---

## 🧪 Testing & Validation

A test client script is included to quickly verify that the server is up and returning valid predictions:

```bash
python test_api_client.py
```

Expected output:
```plaintext
Testing API at http://127.0.0.1:8001...

✅ Root endpoint Check Passed!
✅ Analyze endpoint Check Passed!
Response Data: {'dish': 'biryani', 'confidence': 98.45, ...}
```

*(Note: Verify the port in `test_api_client.py` matches the port your server is running on).*

---

## 🧠 Model Details

- **Base Architecture**: MobileNetV3-Large (pretrained on ImageNet, fine-tuned on custom food image dataset).
- **Target Classes**: 115 distinct categories covering North & South Indian dishes, sweets, beverages, breakfast items, vegetables, and common staples.
- **Input Dimensions**: `224 x 224 x 3` (RGB).
- **Formats Included**:
  - `mobilenetv3_large.tflite`: Quantized/optimized flatbuffer representation for low-latency CPU production inference.
  - `mobilenetv3_large_final.h5`: Full Keras weights preserved for further fine-tuning and evaluation.
- **Training Artifacts**: Jupyter notebooks (`kaggle-mnv3-80-20.ipynb`, `kaggle_efficientnet.ipynb`) documenting dataset preparation, data augmentation, hyperparameter tuning, and cross-architecture benchmarking.

---

## 💡 Future Enhancements

- [ ] Automatic portion / serving size estimation via depth or bounding box scaling.
- [ ] Multi-dish detection on a single plate (segmentation / YOLOv8 integration).
- [ ] Regional water scarcity weighting (adjusting blue water impact based on geographic origin of crops).
- [ ] Exportable PDF / receipt summary for dietary water auditing.

---

## 📄 License & Credits

This project was developed for educational and academic research purposes as part of the 5th Semester coursework.
Dataset values for water footprints are curated based on data published by the **Water Footprint Network (WFN)** and FAO statistical databases.
