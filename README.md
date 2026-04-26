# METAHEURISTIK2 - Optimisasi Parameter NN Berdimensi Tinggi

Proyek ini membandingkan tiga pendekatan optimisasi untuk parameter neural network (jumlah parameter banyak, bukan hanya dua):
- Gradient-based (Autograd + Adam)
- Particle Swarm Optimization (PSO)
- Genetic Algorithm (GA)

Setiap metode dilatih untuk mem-fit fungsi target sintetis, lalu lintasan konvergensi parameter divisualisasikan dengan proyeksi PCA 3D.

## Tujuan
- Menyelesaikan optimisasi parameter NN berdimensi tinggi dengan tiga metode.
- Membandingkan dinamika konvergensi ketiga metode.
- Menyajikan visualisasi konvergensi di ruang PCA hingga 3 dimensi.

## Output per Soal x Metode
- `loss_curve.png`: loss terhadap iterasi.
- `pca_path_3d.png`: lintasan parameter terbaik pada PCA 3D.
- `pca_path_3d.gif`: animasi lintasan konvergensi parameter pada PCA 3D.
- `y_pred_vs_true.gif`: animasi kurva prediksi terhadap target.
- `history.csv`: history loss dan metrik iterasi.
- `summary.json`: ringkasan hasil.
- `best_params.npy`: vektor parameter terbaik model NN.

## Command Utama
Jalankan dari root folder `METAHEURISTIK2`:

```powershell
./scripts/setup_all.ps1
./scripts/run_experiment.ps1
./scripts/show_latest_metrics.ps1
./scripts/run_all_tests.ps1
```
