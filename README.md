# YOLO-LSN
An efficient and lightweight object detection algorithm for UAV aerial imagery, designed to accurately detect mining-induced surface cracks using an improved YOLOv10 framework with LDADH, SOFEP, and NWD modules.
# YOLO-LSN: Lightweight Small Object Detection Network for UAV Imagery

This repository contains the official implementation of **YOLO-LSN**, a lightweight and efficient object detection model tailored for detecting mining-induced surface cracks in UAV aerial imagery. The model is based on the YOLOv10 architecture and enhanced with three novel modules:

- **LDADH**: Lightweight Dynamic Alignment Detection Head  
- **SOFEP**: Small Object Feature Enhancement Pyramid  
- **NWD**: Normalized Wasserstein Distance regression loss

## 📌 Paper
**Title**: Research on UAV Aerial Imagery Detection Algorithm for Mining-Induced Surface Cracks Based on Improved YOLOv10  
**Authors**: Jiayong An, Siyuan Dong, Xuanli Wang, et al.  
**Institution**: Xi’an University of Science and Technology   
*(Equal contribution: Jiayong An )*

## 🧠 Highlights
- 📉 17% fewer parameters compared to baseline YOLOv10
- 🎯 +12% mAP@0.5 improvement on mining-induced surface crack dataset
- ⚡ Real-time FPS maintained (~57.2 on RTX 4060)

## 📂 Structure
```bash
YOLO-LSN/
├── configs/             # Model config files (yaml)
├── models/              # Network definitions including LDADH, SOFEP
├── loss/                # IoU + NWD loss implementation
├── data/                # Dataset config and processing
├── train.py             # Training entry
├── val.py               # Evaluation entry
└── README.md            # Project introduction
