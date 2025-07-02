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

- **Raspberry Pi 5 + HAILO AI HAT** (tested with HAILO-8 chip)  
  🔧 Follow RPi’s official guidance to update RPi5 firmware and enable PCIe 3.0: [RPi AI kit](https://www.raspberrypi.com/documentation/accessories/ai-kit.html)
  Also you may want to check this [RPi AI kit getting started](https://www.raspberrypi.com/documentation/computers/ai.html)
  
- **Hailo-All package** installed on Raspberry Pi:
  ```bash
  sudo apt install hailo-all
  ```

- A reasonably powerful PC (used for HEF model compilation):
  - Ubuntu
  - Intel i7-class CPU or better
  - At least 16 GB RAM
  - Docker installed

- **Registered account in HAILO Developers Zone** If you don't have it yet.
- **Hailo Software Suite 2025-04 Docker image** installed and run on PC. Download it from HAILO Developers Zone, see official installation guide next to download link

- **Python 3.9+** on Raspberry Pi (tested with 3.11) 


---

## 📦 Step 1: Converting YOLOv8 to .hef

### 1. Export YOLO model to ONNX

If you have Ultralytics installed on yor PC or Raspberry Pi, just run the console command:

```bash
yolo export model=flies.pt format=onnx
```

Alternatively, you can use the pre-exported `flies.onnx` from this repo.

> ℹ️ **Note**: Depending on your model architecture, additional export flags (e.g. `dynamic`, `simplify`,`oplocks`) may be required.  
> For this model, the default export worked fine.

---


### 2. Run HAILO Software Suite Docker container on your PC
Clone the repo:
```
git clone https://github.com/rybkady/HAILO-YOLO-INFERENCE-for-RPi5.git
```
It's assumed that cloned directory structure stored into /home/%username% folder.
Now run the docker environment:


```bash
docker run -it --rm \
  -v "$(pwd)/HAILO-YOLO-INFERENCE-for-RPi5:/workspace" \
  hailo_ai_sw_suite_2025-04:1 \
  /bin/bash
```

> 💡 Docker will discard all internal changes after restart. That's why we need to mount repo as as an external folder

---

### 3. Why config files may need modification (or not)

🧠 **Explanation**:  
This model has only **one class**, which caused normalization errors during `.hef` conversion. A workaround was found in this forum thread:  
[HAILO forum discussion](https://community.hailo.ai/t/problem-with-model-optimization/1648/25)

We need to:
- Modify `yolov8n.alls` to set quantization params
- Also, since it's 320x320 model, modify `yolov8n_nms_config.json` for 320×320 resolution

Changes inside Docker are ephemeral, so we will take these modified files on the host and **copy them into the container before .HEF compilation**.

> ⚠️ You *might* be able to pass these files via `hailomz` CLI flags, but I didn’t explore that, just replaced the originals.

---

### 4. Replace configs inside Docker

```bash
sudo cp /workspace/config_changes/yolov8n.alls \
  /local/workspace/hailo_model_zoo/hailo_model_zoo/cfg/alls/generic/yolov8n.alls

sudo cp /workspace/config_changes/yolov8n_nms_config.json \
  /local/workspace/hailo_model_zoo/hailo_model_zoo/cfg/postprocess_config/yolov8n_nms_config.json
```


### 5. Run HEF compilation

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
- `--calib-path` – path to folder with calibration images

> 🕐 **Note**: On Intel i7-9700K with 32 GB RAM, this took ~15 minutes.  
> GPU acceleration is possible but wasn't tested here. Compiler may throw warnings about that.

---

### 6. Save output files

After successful compilation, you will see something like this:
![image](https://github.com/user-attachments/assets/c99353ef-1a2d-4c15-b436-e3049f660201)

Now copy `.hef` and `.har` files back to your host:

```bash
sudo cp yolov8n.* /workspace
```

---
Type 'exit' to leave Docker environment
Don't shutdown your PC for now, we'll need to copy some files to Raspberry pi in next step

## 📦 Step 2: Comparing ONNX and HEF Outputs

Ensure the RPi5 connected to your HAILO AI HAT and has the following installed:

    Python 3.9+

    ultralytics, hailo-all, opencv-python, numpy and their own dependencies.   
    Make sure you're in right VENV if you're set it

Copy these files in some folder on Raspberry Pi from your PC:

    `yolo_inference.py`    – runs ONNX inference

    `hailo_inference.py`   – runs HEF inference

    `yolov8n.hef`          - HEF model we compiled    

    `test.jpg`             - test image for inference

🧠 Note: HEF output differs in:
- Relative vs absolute coordinates
- YXYX vs XYXY format

Both python scripts will normalize outputs to match.
  
  ▶️ Run the scripts:

`python yolo_inference.py`  

`python hailo_inference.py`

📸 Visual Comparison of results
ONNX Detection vs HEF Detection 

![image](https://github.com/user-attachments/assets/bfe7858b-7bf2-4cf3-adeb-f779704fa108) ![image](https://github.com/user-attachments/assets/a6874eab-3a45-4caa-9889-b356f23dd883)

![image](https://github.com/user-attachments/assets/ec1e0c12-4f39-4155-a79f-e95381471c42) ![image](https://github.com/user-attachments/assets/86e56eff-9ee2-4840-8066-142b117b0e63)






	

	

✅ Inference consistency confirmed  
  
    
    

## 📬 Feedback

I hope this how-to saves others time and frustration.
If you find better methods or improvements — feel free to contribute or open an issue.
Created with help from ChatGPT and hands-on trial & error.
