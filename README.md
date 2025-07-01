# 🔧 HAILO YOLO Inference Verification Guide

A practical step-by-step guide for verifying that your HAILO AI HAT setup actually works — by comparing standard YOLOv8 inference results with those from a HAILO-compiled HEF model.

---

## 🎯 Goal

Help Raspberry Pi developers **quickly confirm** that their HAILO AI HAT is functioning properly with a known YOLOv8 model — without spending days digging through fragmented documentation and inconsistent tools.

> ℹ️ This guide assumes you're already familiar with Ultralytics YOLOv8 and have experience training custom object detection models.  
> It **does not** cover model training — only how to run an already-trained model on a HAILO device and validate the results.

---

## 📌 Problem

While powerful, the HAILO software ecosystem is highly fragmented. Even verifying that inference works can turn into a frustrating maze of SDK versions, Docker containers, model conversion quirks, and silent failures.

This guide provides a **minimal reproducible workflow**:
- Export a YOLOv8 model to ONNX
- Compile it to HEF using HAILO tools
- Run inference with both ONNX and HEF
- Compare and validate the results

The model used here is a custom-trained `yolov8n` with a single class (`fly`) and input resolution of **320×320**, trained to detect flies on a green screen background.

---

## ⚙️ Requirements

- **HAILO AI HAT** (tested with HAILO-8 chip on Raspberry Pi 5)  
  🔧 Follow HAILO’s official guidance to update RPi5 firmware and enable PCIe 3.0.

- **Hailo-All RPi package** installed:
  ```bash
  sudo apt install hailo-all
  ```

- A reasonably powerful PC (used for model compilation):
  - Ubuntu
  - Intel i7-class CPU or better
  - At least 16 GB RAM

- **Hailo Software Suite 2025-04 Docker image** installed (see official guide)

- **Python 3.9+** (tested with 3.11)

- Files (available in this repo):
  - `flies.onnx` — ONNX-exported model
  - `flies.hef` — compiled HEF model
  - Sample JPEG images in `/images`

---

## 📦 Step 1: Converting YOLOv8 to .hef

### 1. Export YOLO model to ONNX

If you have Ultralytics installed:

```bash
yolo export model=flies.pt format=onnx
```

Alternatively, you can use the pre-exported `flies.onnx` from this repo.

> ℹ️ **Note**: Depending on your model architecture, additional export flags (e.g. `dynamic`, `simplify`,`oplocks`) may be required.  
> For this model, the default export worked fine.

---

### 2. Run HAILO Software Suite Docker container

```bash
docker run -it --rm \
  -v /path/to/external/drive:/workspace \
  hailo_ai_sw_suite_2025-04:1 \
  /bin/bash
```

> 💡 Use an external drive (e.g., mounted USB) for `/workspace` because Docker will discard all internal changes after restart.

---

### 3. Copy required files into `/workspace`

- `flies.onnx`
- Edited config files:
  - `yolov8n.alls`
  - `yolov8n_nms_config.json`
- A folder of calibration images: `/images`

---

### 4. Why config files may need modification (or not)

🧠 **Explanation**:  
This model has only **one class**, which caused normalization errors during `.hef` conversion. A workaround was found in this forum thread:  
[HAILO forum discussion](https://community.hailo.ai/t/problem-with-model-optimization/1648/25)

You need to:
- Modify `yolov8n.alls` to set quantization params
- Also, since it's 320x320 model, modify `yolov8n_nms_config.json` for 320×320 resolution

Since changes inside Docker are ephemeral, keep these modified files on the host and **copy them into the container before compilation**.

> ⚠️ You *might* be able to pass these files via `hailomz` CLI flags, but I didn’t explore that, just replaced the originals.

---

### 5. Replace configs inside Docker

```bash
sudo cp /workspace/yolov8n.alls \
  /local/workspace/hailo_model_zoo/hailo_model_zoo/cfg/alls/generic/yolov8n.alls

sudo cp /workspace/yolov8n_nms_config.json \
  /local/workspace/hailo_model_zoo/hailo_model_zoo/cfg/postprocess_config/yolov8n_nms_config.json
```

---

### 6. Run HEF compilation

```bash
hailomz compile yolov8n \
  --ckpt=/workspace/flies.onnx \
  --hw-arch hailo8 \
  --calib-path /workspace/calibration_images \
  --classes 1 \
  --performance
```

**Parameters**:
- `--ckpt` – path to ONNX model
- `--hw-arch` – `hailo8` or `hailo8l`
- `--calib-path` – path to folder with sample images

> 🕐 **Note**: On Intel i7-9700K with 32 GB RAM, this took ~15 minutes.  
> GPU acceleration is possible but wasn't tested here.

---

### 7. Save output files

After successful compilation, copy `.hef` and `.har` files back to your host:

```bash
sudo cp yolov8n.* /workspace
```

---

## 📦 Step 2: Comparing ONNX and HEF Outputs

Ensure the RPi5 connected to your HAILO AI HAT and has the following installed:

    Python 3.9+

    ultralytics, hailo-all, opencv-python, numpy and their own dependencies.   
    Make sure you're in right VENV if you're set it

Use these scripts:

    yolo_inference.py – runs ONNX inference

    hailo_inference.py – runs HEF inference

🧠 HEF output differs in:

    Relative vs absolute coordinates

    YXYX vs XYXY format

Both scripts normalize outputs to match.
  
  ▶️ Run both:

`python yolo_inference.py`  

`python hailo_inference.py`

📸 Visual Comparison
ONNX Detection	                                HEF Detection
	
	

✅ Inference consistency confirmed  
  
    
    

## 📬 Feedback

I hope this how-to saves others time and frustration.
If you find better methods or improvements — feel free to contribute or open an issue.
Created with help from ChatGPT and hands-on trial & error.
