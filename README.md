<h1>🦴 Bone Fracture Detection — Medical Imaging with ResNet-50</h1>

<p>
  <img src="https://img.shields.io/badge/Accuracy-88%25+-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Model-ResNet--50-blue?style=for-the-badge&logo=tensorflow"/>
  <img src="https://img.shields.io/badge/Dataset-40k%2B%20X--rays-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Task-Binary%20Classification-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

Binary classification of bone fractures from X-ray images using transfer learning on ResNet-50.
Trained on 40,000+ images from the MURA and RSNA datasets (~4GB). Final test accuracy: **88%+**.

---

## 📌 What it does

Takes an X-ray image as input and predicts whether a bone fracture is present or not.
Includes a desktop GUI so non-technical users can run predictions without writing any code.

---

## 📊 Results

| Metric | Score |
|--------|-------|
| ✅ Test Accuracy | **88%+** |
| 🧠 Architecture | ResNet-50 (transfer learning) |
| 📁 Dataset | MURA + RSNA (~4GB, 40,000+ images) |
| 🏷️ Classes | Fracture / Normal |
| 📐 Input size | 224 × 224 |

---

## 🛠️ Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)

---

## ▶️ How to run
```bash
git clone https://github.com/suhasvenkat/bone-fracture-detection.git
cd bone-fracture-detection
pip install -r requirements.txt
python mainGUI.py
```

The GUI lets you upload any X-ray image and get a prediction with confidence score in seconds.

---

## 📁 Project structure
```
bone-fracture-detection/
├── training_fracture.py     # Model training script
├── training_parts.py        # Body part classification
├── predictions.py           # Inference logic
├── prediction_test.py       # Test suite
├── mainGUI.py               # Desktop GUI
├── PredictResults/          # Sample prediction outputs
├── plots/                   # Training curves
├── docs/                    # Documentation
└── requirements.txt
```

---

## 🧠 Training approach

- Fine-tuned ResNet-50 with ImageNet weights
- Data augmentation (rotation, flip, zoom) to reduce overfitting on medical images
- Binary cross-entropy loss, Adam optimizer
- Evaluated on held-out test set — no data leakage

---

## 🔮 What I'd add next

- [ ] Grad-CAM heatmaps to highlight the fracture region
- [ ] Multi-class support (fracture type / body part)
- [ ] FastAPI endpoint for hospital system integration
- [ ] HuggingFace Spaces demo for live inference

---

## 📂 Dataset

- [MURA](https://stanfordmlgroup.github.io/competitions/mura/) — Stanford ML Group
- [RSNA Bone Age](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-bone-age-challenge-2017)
