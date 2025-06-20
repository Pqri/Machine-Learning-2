# 📘 Repo "Machine‑Learning‑2" – Reproduksi Hands‑On ML (UAS Project)

Repo ini adalah hasil reproduksi praktikal dari **Hands‑On Machine Learning with Scikit‑Learn, Keras and TensorFlow (2nd Edition)** oleh Aurélien Géron. Semua kode berada dalam format Jupyter Notebook, mengikuti urutan bab buku, dan dirancang sebagai bagian dari UAS untuk memperdalam pemahaman baik teori maupun praktik Machine Learning.

---

## 📚 Struktur Bab & Ringkasan

1. **01 The Machine Learning Landscape**  
   Pengenalan tipe-tipe ML (supervised, unsupervised, reinforcement), tantangan utama (overfitting, bias, data quality), dan ekosistem ML :contentReference[oaicite:1]{index=1}.

2. **02 End‑to‑End Machine Learning Project**  
   Complete pipeline prediksi harga rumah California: data loading, cleansing, visualisasi, preprocessing, training, evaluasi, dan deployment :contentReference[oaicite:2]{index=2}.

3. **03 Classification**  
   Klasifikasi digit (MNIST), evaluasi model (confusion matrix, precision/recall, ROC), serta strategi multiclass :contentReference[oaicite:3]{index=3}.

4. **04 Training Linear Models**  
   Regresi linier, gradient descent, regularisasi (Ridge, Lasso, ElasticNet), serta regresi logistik & softmax :contentReference[oaicite:4]{index=4}.

5. **05 Support Vector Machines**  
   SVM linear/non‑linear dengan kernel trick, dan aplikasi SVM untuk regresi :contentReference[oaicite:5]{index=5}.

6. **06 Decision Trees**  
   Teori (CART), visualisasi decision tree, overfitting, pruning, dan regresi pohon keputusan :contentReference[oaicite:6]{index=6}.

7. **07 Ensemble Learning and Random Forests**  
   Voting, bagging, boosting (AdaBoost, Gradient Boosting), ensembling & stacking :contentReference[oaicite:7]{index=7}.

8. **08 Dimensionality Reduction**  
   PCA, Kernel PCA, LLE: visualisasi, compression, dan preprocessing :contentReference[oaicite:8]{index=8}.

9. **09 Unsupervised Learning**  
   Clustering (K‑Means, DBSCAN), Gaussian Mixture Models, anomali deteksi :contentReference[oaicite:9]{index=9}.

10. **10 Neural Nets with Keras**  
    MLP, arsitektur jaringan dasar, implementasi Keras Functional API/subclassing :contentReference[oaicite:10]{index=10}.

11. **11 Training Deep Neural Networks**  
    Gradient issues, optimizer (Adam, RMSprop), regularisasi (Dropout), dan teknik training efisien :contentReference[oaicite:11]{index=11}.

12. **12 Custom Models and Training with TensorFlow**  
    Custom training loop, custom metric & activation, serta model build dari dasar :contentReference[oaicite:12]{index=12}.

13. **13 Loading and Preprocessing Data with TensorFlow**  
    TFData, TFRecords, one-hot encoding, embedding, dan pipeline input efisien :contentReference[oaicite:13]{index=13}.

14. **14 Deep Computer Vision with CNNs**  
    CNN modern (LeNet → ResNet/Xception), serta transfer learning dan semantic segmentation :contentReference[oaicite:14]{index=14}.

15. **15 Processing Sequences Using RNNs and CNNs**  
    Sequence modeling untuk time series, audio, LSTM/GRU, dan hybrid CNN‐RNN :contentReference[oaicite:15]{index=15}.

16. **16 NLP with RNNs and Attention**  
    Text generation, sentiment analysis, encoder-decoder, attention, serta arsitektur Transformer :contentReference[oaicite:16]{index=16}.

17. **17 Autoencoders and GANs**  
    Autoencoder (denoising, variational) dan GAN (DCGAN, StyleGAN) :contentReference[oaicite:17]{index=17}.

18. **18 Reinforcement Learning**  
    Konsep dasar RL, Q-learning, dan implementasi agen DQN via TF‑Agents :contentReference[oaicite:18]{index=18}.

19. **19 Training and Deploying at Scale**  
    TensorFlow Serving, distribusi training, dan deployment ke GCP & mobile :contentReference[oaicite:19]{index=19}.

---

## ⚙️ Dependensi & Instalasi

- Python 3.8
- Scikit‑Learn, TensorFlow 2.x, Keras
- NumPy, Pandas, Matplotlib
- Jupyter Notebook / Colab

Clone & install lingkungan:
```bash
git clone https://github.com/Pqri/Machine-Learning-2.git
cd Machine-Learning-2
conda env create -f environment.yml
conda activate homl2
jupyter notebook
