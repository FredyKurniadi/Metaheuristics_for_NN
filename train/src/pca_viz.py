from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def pca_transform(data: np.ndarray, n_components: int = 3):
    x = np.asarray(data, dtype=float)
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean

    if x.shape[0] < 2:
        comps = np.zeros((n_components, x.shape[1]), dtype=float)
        coords = np.zeros((x.shape[0], n_components), dtype=float)
        explained = np.zeros(n_components, dtype=float)
        return coords, comps, mean.reshape(-1), explained

    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    max_k = min(n_components, vt.shape[0])
    comps = vt[:max_k]
    coords = xc @ comps.T

    total_var = np.sum(s * s)
    explained = (s[:max_k] * s[:max_k]) / max(total_var, 1e-12)

    if max_k < n_components:
        pad = n_components - max_k
        coords = np.hstack([coords, np.zeros((coords.shape[0], pad), dtype=float)])
        comps = np.vstack([comps, np.zeros((pad, comps.shape[1]), dtype=float)])
        explained = np.concatenate([explained, np.zeros(pad, dtype=float)])

    return coords, comps, mean.reshape(-1), explained


def save_loss_curve(history_loss: np.ndarray, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(len(history_loss)), history_loss, color="tab:blue", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Loss (MSE)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_pca_path_3d(coords: np.ndarray, out_path: Path, title: str):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color="tab:blue", linewidth=2)
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=np.arange(len(coords)), cmap="plasma", s=22)

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_pca_path_3d_gif(coords: np.ndarray, out_path: Path, title: str, fps: int):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    x_min, x_max = float(np.min(coords[:, 0])), float(np.max(coords[:, 0]))
    y_min, y_max = float(np.min(coords[:, 1])), float(np.max(coords[:, 1]))
    z_min, z_max = float(np.min(coords[:, 2])), float(np.max(coords[:, 2]))
    pad = 0.1

    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_zlim(z_min - pad, z_max + pad)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    line, = ax.plot([], [], [], color="tab:blue", linewidth=2)
    point = ax.scatter([], [], [], color="tab:red", s=50)

    def update(frame: int):
        idx = frame + 1
        xs = coords[:idx, 0]
        ys = coords[:idx, 1]
        zs = coords[:idx, 2]
        line.set_data(xs, ys)
        line.set_3d_properties(zs)

        point._offsets3d = (np.array([xs[-1]]), np.array([ys[-1]]), np.array([zs[-1]]))
        ax.view_init(elev=22, azim=35 + frame * 0.7)
        return line, point

    anim = FuncAnimation(fig, update, frames=len(coords), interval=int(1000 / max(1, fps)), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_y_curve_animation(x, y_obs, y_true, y_preds, out_path: Path, title: str, fps: int):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.25)

    ax.scatter(x, y_obs, s=10, alpha=0.3, color="gray", label="observed")
    ax.plot(x, y_true, color="tab:green", linewidth=2, label="target (fixed)")
    pred_line, = ax.plot(x, y_preds[0], color="tab:orange", linewidth=2, label="nn prediction")
    text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    y_all = np.concatenate([y_obs, y_true] + y_preds)
    pad = 0.08 * max(float(np.max(y_all) - np.min(y_all)), 1e-6)
    ax.set_ylim(float(np.min(y_all) - pad), float(np.max(y_all) + pad))
    ax.legend(loc="upper right")

    def update(frame: int):
        pred_line.set_ydata(y_preds[frame])
        text.set_text(f"iter={frame}")
        return pred_line, text

    anim = FuncAnimation(fig, update, frames=len(y_preds), interval=int(1000 / max(1, fps)), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
