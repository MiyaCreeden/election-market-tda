import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers '3d' projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from itertools import combinations


def _pick_radius(point_cloud, dgms):
    """
    Choose a filtration radius that reveals geometric structure.
    Uses 20th percentile of pairwise distances — shows skeleton
    structure while keeping geometric shape visible.
    Falls back to median H0 death if distance computation fails.
    """
    try:
        dists = pdist(point_cloud)
        return float(np.percentile(dists, 20))
    except Exception:
        pass

    h0 = dgms[0]
    finite = np.isfinite(h0[:, 1])
    if finite.any():
        return float(np.median(h0[finite, 1]))
    return 1.0


def _build_vr_complex(point_cloud, radius):
    """
    Build Vietoris-Rips complex at given radius.
    Returns (edges, triangles) — each a list of index tuples.
    """
    n = len(point_cloud)
    dist = squareform(pdist(point_cloud))

    # Edges: pairs within radius
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] <= radius:
                edges.append((i, j))

    # Triangles: triples where all 3 edges exist
    edge_set = set()
    for i, j in edges:
        edge_set.add((i, j))
        edge_set.add((j, i))  # for fast lookup

    triangles = []
    for i, j, k in combinations(range(n), 3):
        if (i, j) in edge_set and (j, k) in edge_set and (i, k) in edge_set:
            triangles.append((i, j, k))

    return edges, triangles


def plot_vr_complex_3d(ax, point_cloud, dgms, title="VR Complex"):
    """
    3D Vietoris-Rips complex: points + edges + filled triangles.
    Radius auto-picked from persistence diagram.
    """
    radius = _pick_radius(point_cloud, dgms)
    edges, triangles = _build_vr_complex(point_cloud, radius)

    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    t = np.arange(len(point_cloud))

    # Points
    ax.scatter(x, y, z, c=t, cmap="viridis", s=50,
               edgecolors="k", linewidth=0.3, alpha=0.9, zorder=5)

    # Edges
    if edges:
        segs = [[(x[i], y[i], z[i]), (x[j], y[j], z[j])] for i, j in edges]
        edge_coll = Line3DCollection(segs, colors="gray", linewidths=0.5, alpha=0.4, zorder=1)
        ax.add_collection(edge_coll)

    # Triangles (filled faces) — cap to avoid solid-blob plots
    MAX_TRIS = 500
    if triangles:
        tris = [(i, j, k) for i, j, k in triangles]
        if len(tris) > MAX_TRIS:
            # Sub-sample evenly
            idx = np.linspace(0, len(tris) - 1, MAX_TRIS, dtype=int)
            tris = [tris[i] for i in idx]
        tri_verts = [[(x[i], y[i], z[i]), (x[j], y[j], z[j]), (x[k], y[k], z[k])]
                     for i, j, k in tris]
        tri_coll = Poly3DCollection(tri_verts, facecolors="steelblue", edgecolors="none",
                                    alpha=0.20, zorder=2)
        ax.add_collection(tri_coll)

    ax.set_xlabel("Level (std)")
    ax.set_ylabel("Momentum (std)")
    ax.set_zlabel("Acceleration (std)")
    n_tris_shown = min(len(triangles), MAX_TRIS) if triangles else 0
    ax.set_title(f"{title}\n(ε = {radius:.3f},  {len(point_cloud)} pts,  edges={len(edges)},  tris={n_tris_shown}{'/'+str(len(triangles)) if triangles and len(triangles)>MAX_TRIS else ''})",
                 fontsize=10, fontweight="bold")


def plot_persistence_3d(ax, dgms, title="Persistence Diagram"):
    """
    3D persistence diagram: birth (x), death (y), persistence (z).
    H0 bars in blue, H1 bars in red.
    """
    colors = {0: "steelblue", 1: "crimson"}
    labels = {0: "H₀", 1: "H₁"}
    max_val = 0

    for dim_idx, dgm in enumerate(dgms):
        if len(dgm) == 0:
            continue
        births = dgm[:, 0]
        deaths = dgm[:, 1]
        pers = deaths - births

        # Drop infinite deaths (they break 3D plot scale)
        finite = np.isfinite(deaths)
        births = births[finite]
        deaths = deaths[finite]
        pers = pers[finite]

        if len(births) == 0:
            continue

        max_val = max(max_val, deaths.max(), births.max())

        ax.bar3d(births, deaths, np.zeros_like(pers),
                 dx=pers * 0.1 + 0.02, dy=pers * 0.1 + 0.02, dz=pers,
                 color=colors.get(dim_idx, "gray"), alpha=0.7,
                 label=labels.get(dim_idx, f"H{dim_idx}"))

    # Diagonal (birth = death)
    if max_val > 0:
        diag = np.linspace(0, max_val * 1.05, 50)
        ax.plot(diag, diag, np.zeros_like(diag), "k--", linewidth=0.6, alpha=0.4)

    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_zlabel("Persistence")
    ax.set_title(title, fontsize=11, fontweight="bold")
    if len(dgms[0]) > 0 or (len(dgms) > 1 and len(dgms[1]) > 0):
        ax.legend(fontsize=8, loc="upper left")


def pca_project(point_cloud, n_components=3):
    """
    Project high-dimensional point cloud to 3D via PCA.
    Returns (projected, variance_ratio).
    """
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(point_cloud)
    return projected, pca.explained_variance_ratio_


def plot_comparison_3d(stable_cloud, shock_cloud,
                       stable_dgms, shock_dgms,
                       stable_label="Pre-Spike Baseline",
                       shock_label="Spike Peak",
                       save_path=None,
                       axis_labels=("Level (std)", "Momentum (std)", "Acceleration (std)"),
                       var_explained=None):
    """
    2×2 3D comparison figure:
      top-left:   stable VR complex
      top-right:  shock VR complex
      bottom-left:   stable persistence diagram
      bottom-right:  shock persistence diagram

    Set axis_labels to ('PC1','PC2','PC3') and pass var_explained when
    visualising PCA-projected clouds.
    """
    fig = plt.figure(figsize=(16, 12))

    # Build VR titles
    stable_title = f"VR Complex — {stable_label}"
    shock_title  = f"VR Complex — {shock_label}"
    if var_explained is not None:
        stable_title += f"\n(var explained: {var_explained[0]*100:.0f}% / {var_explained[1]*100:.0f}% / {var_explained[2]*100:.0f}%)"
        shock_title  += f"\n(var explained: {var_explained[0]*100:.0f}% / {var_explained[1]*100:.0f}% / {var_explained[2]*100:.0f}%)"

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    _plot_vr_complex_3d_ax(ax1, stable_cloud, stable_dgms, stable_title, axis_labels)

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    _plot_vr_complex_3d_ax(ax2, shock_cloud, shock_dgms, shock_title, axis_labels)

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    plot_persistence_3d(ax3, stable_dgms, f"Persistence — {stable_label}")

    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    plot_persistence_3d(ax4, shock_dgms, f"Persistence — {shock_label}")

    fig.suptitle("TDA: VR Complexes & Persistence Diagrams",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f">>> Saved: {save_path}")

    return fig


def _plot_vr_complex_3d_ax(ax, point_cloud, dgms, title, axis_labels):
    """
    Internal: plot VR complex on a given 3D axis with custom labels.
    Calls through to _pick_radius + _build_vr_complex + rendering.
    """
    radius = _pick_radius(point_cloud, dgms)
    edges, triangles = _build_vr_complex(point_cloud, radius)

    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    t = np.arange(len(point_cloud))

    ax.scatter(x, y, z, c=t, cmap="viridis", s=50,
               edgecolors="k", linewidth=0.3, alpha=0.9, zorder=5)

    if edges:
        segs = [[(x[i], y[i], z[i]), (x[j], y[j], z[j])] for i, j in edges]
        ax.add_collection(Line3DCollection(segs, colors="gray", linewidths=0.5,
                                            alpha=0.4, zorder=1))

    MAX_TRIS = 500
    if triangles:
        tris = [(i, j, k) for i, j, k in triangles]
        if len(tris) > MAX_TRIS:
            idx = np.linspace(0, len(tris) - 1, MAX_TRIS, dtype=int)
            tris = [tris[i] for i in idx]
        tri_verts = [[(x[i], y[i], z[i]), (x[j], y[j], z[j]), (x[k], y[k], z[k])]
                     for i, j, k in tris]
        ax.add_collection(Poly3DCollection(tri_verts, facecolors="steelblue",
                                            edgecolors="none", alpha=0.20, zorder=2))

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    n_tris_shown = min(len(triangles), MAX_TRIS) if triangles else 0
    tri_label = f", tris={n_tris_shown}{'/'+str(len(triangles)) if triangles and len(triangles)>MAX_TRIS else ''}"
    ax.set_title(f"{title}\n(ε={radius:.3f}, {len(point_cloud)} pts, edges={len(edges)}{tri_label})",
                 fontsize=10, fontweight="bold")
