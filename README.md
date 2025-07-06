# 👤 Face Generation with Deep Convolutional GANs (DCGAN)

This project focuses on building and training a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate realistic images of human faces. Using the **CelebFaces Attributes Dataset (CelebA)**, the objective is to learn a data distribution such that the **generator** can produce new, realistic face images from random noise.

---

## 📘 Project Overview

Generative Adversarial Networks (GANs) consist of two networks:
- A **Generator** that learns to generate fake images resembling real ones
- A **Discriminator** that learns to distinguish real images from fake ones

These two networks are trained together in an adversarial setup, where the generator aims to fool the discriminator, and the discriminator aims to correctly identify fake samples.

In this project, you'll:
- Load and preprocess the CelebA dataset
- Build a DCGAN architecture using PyTorch
- Train the GAN on face images
- Visualize the generated face images at different stages of training

---

## 🎯 Objective

- Train a GAN to generate realistic 64x64 RGB face images
- Use adversarial training to improve generator performance over time
- Visualize the progression of face generation quality during training

---

## 📦 Dataset

- **Name:** CelebFaces Attributes Dataset (CelebA)
- **Images:** Over 200,000 celebrity face images
- **Resolution:** Preprocessed to 64x64 RGB
- **Source:** [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

---

## 🧰 Tools & Libraries

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- torchvision (for data loading and image transforms)
- tqdm (for progress visualization)

---

## 🛠️ Key Components

### 1. **Data Loading**
- Use `torchvision.datasets.ImageFolder` with appropriate transforms
- Normalize images to [-1, 1] range for stable GAN training

### 2. **Model Architecture**
- **Generator:**
  - Transposed convolution layers
  - Batch normalization
  - ReLU activations
- **Discriminator:**
  - Convolutional layers
  - LeakyReLU activations
  - Sigmoid output for binary classification

### 3. **Training Details**
- Binary Cross Entropy Loss
- Adam optimizer (learning rate ~0.0002)
- Alternating updates of generator and discriminator

### 4. **Evaluation**
- Generate and save images at regular intervals
- Visually inspect quality of generated images
- Optionally create GIF or animation of training progress

---

## 📈 Expected Output

- A trained generator that can take a noise vector as input and output **synthetic but realistic** human face images
- Sample outputs:
  - Low resolution faces early in training
  - Sharper and more realistic faces as training progresses

---

## ⚙️ Requirements

Install the required libraries using pip:

```bash
pip install torch torchvision matplotlib numpy tqdm
```

## 🚀 Run Instructions
Clone the repository and navigate into the project folder

Download the CelebA dataset (preprocessed version)

Run dlnd_face_generation.ipynb in Jupyter Notebook

Watch the generated faces improve as training progresses

## 👤 Contact
Waqar Ahmed
📧 Email: waqar.nu@gmail.com
🔗 GitHub: waqar-ahmed91

## 📜 License
This project is for educational purposes only. The CelebA dataset is licensed by its original authors and subject to their terms of use.
