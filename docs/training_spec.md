# Training Specification

## Pendekatan
- Gradient-based: Adam + autograd.
- PSO.
- GA.

## Model
Model yang dioptimisasi adalah MLP fully-connected dengan parameter flatten jadi satu vektor berdimensi tinggi.

## Visualisasi Konvergensi
Trajectory parameter terbaik per iterasi diproyeksikan ke PCA 3 komponen:
- Plot statis lintasan 3D.
- GIF progres lintasan 3D.

## Metrik
- Objective: MSE.
- Metrik utama: `best_loss`.
