# Traffic-sign-AI

A traffic sign detection application using YOLOv8, FastAPI backend, and a React + Vite frontend.

---

## Requirements

- **Python 3.10 – 3.11** (Python 3.10 recommended)
- **Node.js 16+**
- **Chrome or Edge browser** (required for webcam access)

---

## Getting Started

### Step 1 — Create a Python virtual environment

```bash
python3.10 -m venv .venv
```

> If your machine uses `python` instead of `python3.10`:
> ```bash
> python -m venv .venv
> ```

---

### Step 2 — Activate the virtual environment

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

Once activated, you should see `(.venv)` at the beginning of your terminal prompt.

---

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4 — Install frontend dependencies

```bash
cd Frontend/Traffic-sign-recognition
npm install
cd ../..
```

---

### Step 5 — Run the application

```bash
python demo.py
```

This will automatically start both the backend and the frontend. Once running, open your browser and navigate to the address shown in the terminal (typically `http://localhost:5173`).

---

## Important Note

> ⚠️ **Use Chrome or Edge**
>
> Some browsers (Firefox, Safari) may block webcam access on `localhost`. To ensure the camera feature works correctly, please use **Google Chrome** or **Microsoft Edge**.

---

## Supported Traffic Sign Classes

The model was trained on **43 traffic sign classes**:

| ID | Class Name |
|----|------------|
| 0  | Speed limit (20km/h) |
| 1  | Speed limit (30km/h) |
| 2  | Speed limit (50km/h) |
| 3  | Speed limit (60km/h) |
| 4  | Speed limit (70km/h) |
| 5  | Speed limit (80km/h) |
| 6  | End of speed limit (80km/h) |
| 7  | Speed limit (100km/h) |
| 8  | Speed limit (120km/h) |
| 9  | No passing |
| 10 | No passing vehicle over 3.5 tons |
| 11 | Right-of-way at intersection |
| 12 | Priority road |
| 13 | Yield |
| 14 | Stop |
| 15 | No vehicles |
| 16 | Vehicle > 3.5 tons prohibited |
| 17 | No entry |
| 18 | General caution |
| 19 | Dangerous curve left |
| 20 | Dangerous curve right |
| 21 | Double curve |
| 22 | Bumpy road |
| 23 | Slippery road |
| 24 | Road narrows on the right |
| 25 | Road work |
| 26 | Traffic signals |
| 27 | Pedestrians |
| 28 | Children crossing |
| 29 | Bicycles crossing |
| 30 | Beware of ice/snow |
| 31 | Wild animals crossing |
| 32 | End speed + passing limits |
| 33 | Turn right ahead |
| 34 | Turn left ahead |
| 35 | Ahead only |
| 36 | Go straight or right |
| 37 | Go straight or left |
| 38 | Keep right |
| 39 | Keep left |
| 40 | Roundabout mandatory |
| 41 | End of no passing |
| 42 | End no passing vehicle > 3.5 tons |

> If the model does not produce the expected result, try using an image that contains **only one traffic sign** from the list above.
