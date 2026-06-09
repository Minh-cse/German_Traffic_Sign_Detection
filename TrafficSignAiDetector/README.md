 # Traffic-sign-AI

Traffic-sign-AI is a compact repository for detecting and recognizing traffic signs using YOLO-based models with a simple Python backend and a Vite + React frontend for visualization.

**Key features:**
- Real-time / batch traffic sign detection using YOLO models (multiple pretrained weights included).
- FastAPI backend served by Uvicorn to provide a lightweight, async inference API.
- Vite + React frontend for visualization and user interaction.

**Repository structure**
- `backend/` : Flask app and trained weights used by the server (`backend/app.py`, `backend/best.pt`).
- `Frontend/Traffic-sign-recognition/` : Vite + React frontend for visualization and UI.
- `yolov8n.pt`, `yolov8s.pt`, `yolo11n.pt` : Example model weight files included for quick experiments.
- `main.py`, `check.py`, `process_data.py` : Utility scripts for running detection, quick checks, and preprocessing.
- `requirements.txt` : Python dependencies for the backend and detection scripts.

Getting started
---------------

These instructions assume you have Python 3.8+ and Node.js (16+) installed. (python 3.10 is recommended)

1) Install Python dependencies

```bash
cd Traffic-sign-AI
python -m pip install -r requirements.txt
```

2) (Optional) Create a virtual environment

```bash
python -m venv .venv
.\\.venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

3) Backend: run the FastAPI (Uvicorn) server

The backend app is in `backend/app.py`. You can run it using the bundled `uvicorn` CLI or run the file directly which will start uvicorn as well.

Install `uvicorn` if not present:

```bash
python -m pip install uvicorn[standard]
```

Run with the CLI (recommended during development):

```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Or run the module directly (the file includes a uvicorn entrypoint):

```bash
python backend/app.py
```

By default the backend will load a model from `backend/best.pt` if present. You can replace that with any of the provided weights (`yolov8n.pt`, `yolov8s.pt`, `yolo11n.pt`) or point the app to another weights file.

4) Frontend: run the Vite dev server

```bash
cd Frontend/Traffic-sign-recognition
npm install
npm run dev
```

The frontend expects the backend API to be available (see `backend/app.py`). Update the API base URL in the frontend if your backend runs on a different host/port.

Basic usage
-----------
- Quick detection (single image or folder): run `python main.py` or `python check.py` depending on which script you prefer — these scripts are small wrappers around the model inference routine. Use `process_data.py` to prepare or augment datasets.
- To train or run YOLO training, use the Ultralytics/Yolov8 CLI or your preferred training script and point it at your dataset and configs.

Notes about models
------------------
- The repository contains several sample weight files for convenience: `yolov8n.pt`, `yolov8s.pt`, and `yolo11n.pt`. These are example detectors — adjust or retrain for better accuracy on your dataset.
- If you plan to train from scratch or fine-tune, ensure your dataset follows YOLO-format labels and update training config accordingly.
<details>
<summary><strong> Supported Traffic Sign Classes</strong></summary>

Due to limited time and computational resources, our team was only able to train the model on the following **43 traffic sign classes**.

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
| 15 | No vehicleicles |
| 16 | vehicle > 3.5 tons prohibited |
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

If the model does not produce the expected result, please try using an image that contains **only one of the supported traffic signs** listed above.
</details>

Development tips
----------------
- Use a Python virtual environment to avoid dependency clashes.
- Keep large model weights out of source control for collaborative projects — instead share download links or store them in an artifact store.

Contributing
------------
Contributions are welcome. Please open an issue for bugs or feature requests, and submit pull requests for code changes. Add clear tests and update the README when adding new scripts or features.

License
-------
Specify your license here (e.g., MIT). If you don't want to include a license yet, add a `LICENSE` file later.

Contact
-------
- If you have questions, open an issue or contact the repository owner.
- This is the datasets.rar link: https://drive.google.com/file/d/17yAfGhY9avPxXEVHfmK6JCf1QryQ8-hE/view?usp=sharing

