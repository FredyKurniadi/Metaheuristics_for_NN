# Arsitektur

## Alur
1. Generate data sintetis per soal.
2. Bangun model NN dengan arsitektur dari config.
3. Optimisasi parameter NN dengan metode: gradient_autograd, pso, ga.
4. Simpan parameter terbaik dan history tiap iterasi.
5. Proyeksikan trajectory parameter ke PCA 3D untuk visualisasi konvergensi.

## Komponen
- `train/src/core.py`: generator data, loss MSE.
- `train/src/nn_model.py`: util parameter vector, forward pass NN.
- `train/src/optimizers.py`: gradient, pso, ga.
- `train/src/pca_viz.py`: PCA dan visualisasi lintasan konvergensi.
- `train/src/experiment.py`: pipeline satu eksperimen.
- `train/src/main.py`: menjalankan semua kombinasi.
