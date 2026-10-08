"""Video overlays for Meta-World rollouts (proposal heat-map + weight profile)."""

import os

import numpy as np

from scout.envs.metaworld.pointcloud import FRAME_H, FRAME_W, project

_CMAP_HOT = None


def _get_cmap(name):
    import matplotlib
    try:
        return matplotlib.colormaps[name]          # matplotlib >= 3.5
    except (AttributeError, KeyError):
        import matplotlib.cm as cm
        return cm.get_cmap(name)                   # older matplotlib


def _hot_color(w_norm):
    global _CMAP_HOT
    if _CMAP_HOT is None:
        _CMAP_HOT = _get_cmap("hot")
    r, g, b, _ = _CMAP_HOT(float(w_norm))
    return int(r * 255), int(g * 255), int(b * 255)


def make_weight_panel(task, eval_pcd, weights_nk, means_nk, predicted,
                      hist_weights_nk, hist_means_nk, hist_targets,
                      last_outcome_pcd, attempt, max_attempts, title,
                      top_k=60, gt_target=None):
    """Right-hand panel: object cloud + top proposals (top), and the proposal
    weight profile along the task's discriminating axis (bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    ax_a, ax_b = task.viz_axes
    bin_axis = task.viz_bin_axis
    BG, FG, n_bins = "#111111", "#dddddd", 50
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FRAME_W / 96, FRAME_H / 96), dpi=96,
                                   gridspec_kw={"height_ratios": [1.4, 1]})
    fig.patch.set_facecolor(BG)

    ax1.set_facecolor("#0a0a0a")
    if last_outcome_pcd is not None:
        ax1.scatter(last_outcome_pcd[:, ax_a], last_outcome_pcd[:, ax_b], c="#333333", s=0.6, alpha=0.4)
    ax1.scatter(eval_pcd[:, ax_a], eval_pcd[:, ax_b], c="#555555", s=0.8, alpha=0.5, zorder=1)

    top_idx = np.argsort(weights_nk)[-top_k:]
    cmap = _get_cmap("hot")
    norm = Normalize(vmin=weights_nk.min(), vmax=weights_nk.max())
    ax1.scatter(means_nk[top_idx, ax_a], means_nk[top_idx, ax_b],
                c=cmap(norm(weights_nk[top_idx])), s=10, alpha=0.9, zorder=3)

    colors_hist = plt.cm.Reds(np.linspace(0.3, 0.8, max(len(hist_targets), 1)))
    for j, hg in enumerate(hist_targets):
        ax1.scatter(hg[ax_a], hg[ax_b], c=[colors_hist[j]], s=40, marker="x", linewidths=1.0, zorder=4, alpha=0.6)
    ax1.scatter(predicted[ax_a], predicted[ax_b], c="cyan", s=120, marker="*", zorder=5)
    if gt_target is not None:
        ax1.scatter(gt_target[ax_a], gt_target[ax_b], c="#00ff00", s=120, marker="P", zorder=6,
                    edgecolors="black", linewidths=0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, fraction=0.03, pad=0.02)
    cbar.set_label("proposal weight", color=FG, fontsize=6)
    cbar.ax.yaxis.set_tick_params(color=FG, labelsize=5)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=FG)

    ax1.set_title(f"Attempt {attempt + 1}/{max_attempts}  [{title}]", color=FG, fontsize=7, pad=3)
    ax1.set_xlabel(task.viz_axis_labels[0], color=FG, fontsize=6)
    ax1.set_ylabel(task.viz_axis_labels[1], color=FG, fontsize=6)
    ax1.tick_params(colors=FG, labelsize=5)
    for sp in ax1.spines.values():
        sp.set_color("#333333")

    ax2.set_facecolor("#0a0a0a")

    def _bin(m, w):
        xs = m[:, bin_axis]
        x0, x1 = xs.min(), xs.max()
        if x0 == x1:
            return np.zeros(n_bins), np.zeros(n_bins)
        edges = np.linspace(x0, x1, n_bins + 1)
        bw = np.array([w[(xs >= edges[b]) & (xs < edges[b + 1])].sum() for b in range(n_bins)])
        return (edges[:-1] + edges[1:]) / 2, bw

    n_hist = len(hist_means_nk)
    for j, (hm, hw) in enumerate(zip(hist_means_nk, hist_weights_nk)):
        xc, bw = _bin(hm, hw)
        alpha = 0.20 + 0.25 * (j / max(n_hist, 1))
        ax2.plot(xc, bw, color="#888888", linewidth=0.8, alpha=alpha)
        ax2.axvline(hist_targets[j][bin_axis], color="#cc4444", linestyle=":", linewidth=0.7, alpha=alpha)
    xc, bw = _bin(means_nk, weights_nk)
    ax2.plot(xc, bw, color="#ff8800", linewidth=1.5)
    ax2.axvline(predicted[bin_axis], color="cyan", linestyle="--", linewidth=1.0, alpha=0.9)
    if gt_target is not None:
        ax2.axvline(gt_target[bin_axis], color="#00ff00", linestyle="-.", linewidth=1.5, alpha=0.9)

    axis_name = "xyz"[bin_axis]
    ax2.set_title(f"Proposal weight profile along {axis_name}", color=FG, fontsize=7, pad=3)
    ax2.set_xlabel(f"proposal mean {axis_name}  (m)", color=FG, fontsize=6)
    ax2.set_ylabel("sum weight / bin", color=FG, fontsize=6)
    ax2.tick_params(colors=FG, labelsize=5)
    for sp in ax2.spines.values():
        sp.set_color("#333333")

    plt.tight_layout(pad=0.6)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
    plt.close(fig)
    return cv2.resize(buf, (FRAME_W, FRAME_H))


def draw_env_frame(renderer, base_env, weights_nk, means_nk, predicted, hist_targets,
                   cam_params, text_lines, top_n=60, gt_target=None):
    import cv2
    renderer.update_scene(base_env.data, camera="corner")
    frame = cv2.flip(renderer.render().copy(), 0)

    top_idx = np.argsort(weights_nk)[-top_n:]
    top_w, top_m = weights_nk[top_idx], means_nk[top_idx]
    w_norm = (top_w - top_w.min()) / (top_w.max() - top_w.min() + 1e-8)
    for i in range(len(top_idx)):
        px = project(top_m[i], cam_params)
        if px is None:
            continue
        r, g, b = _hot_color(w_norm[i])
        cv2.circle(frame, px, max(2, int(4 * w_norm[i])), (b, g, r), -1)

    px_g = project(predicted, cam_params)
    if px_g is not None:
        cv2.circle(frame, px_g, 8, (255, 255, 0), -1)
        cv2.circle(frame, px_g, 10, (255, 255, 0), 2)
        cv2.drawMarker(frame, px_g, (0, 255, 255), cv2.MARKER_STAR, 18, 2, cv2.LINE_AA)
    if gt_target is not None:
        px_gt = project(gt_target, cam_params)
        if px_gt is not None:
            cv2.drawMarker(frame, px_gt, (0, 255, 0), cv2.MARKER_DIAMOND, 16, 2, cv2.LINE_AA)
    for j, hg in enumerate(hist_targets):
        px_h = project(hg, cam_params)
        if px_h is None:
            continue
        col = int(80 + 100 * (j / max(len(hist_targets), 1)))
        cv2.drawMarker(frame, px_h, (0, 0, col), cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)

    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    for i, txt in enumerate(text_lines):
        y = 6 + (i + 1) * 18
        cv2.putText(frame, txt, (7, y + 1), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
        cv2.putText(frame, txt, (6, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return frame


def draw_attempt_dots(frame, per_attempt_so_far):
    import cv2
    H = frame.shape[0]
    r, x0, y = 7, 8, H - 10
    for k, ok in enumerate(per_attempt_so_far):
        cx = x0 + k * (2 * r + 4)
        cv2.circle(frame, (cx, y), r, (0, 200, 0) if ok else (0, 0, 200), -1)
        cv2.circle(frame, (cx, y), r, (220, 220, 220), 1)
    return frame


def save_video(frames, path, fps=30):
    try:
        import imageio
    except ImportError:
        print(f"  [video] imageio not found, skipping {path}")
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=1)
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"  Saved video -> {path} ({len(frames)} frames)")
