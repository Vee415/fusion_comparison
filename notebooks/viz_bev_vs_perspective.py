"""Visual: camera ground footprint (trapezoid) vs BEV grid (square).

Two panels, same car at z=5m and z=50m:
  LEFT  -- what the camera sees: trapezoidal ground footprint, car scales with 1/z
  RIGHT -- what BEV stores: square grid, car occupies same cells at any distance
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, Rectangle, FancyArrow
from pathlib import Path

RANGE = 64.0
RES = 0.2
FX = 720.0
CX_PX = 640  # image center, px
IMG_W = 1280
IMG_H = 400


def draw_camera(ax):
    ax.set_title("Camera ground footprint (trapezoid)\npixel = a ray, not a point", fontsize=11)
    ax.set_xlim(-40, 40); ax.set_ylim(-5, RANGE + 5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, right)"); ax.set_ylabel("z (m, forward)")
    ax.grid(True, alpha=0.3)

    # ego camera at origin, looking +z
    ax.plot(0, 0, "k^", ms=12)
    ax.text(0, -2.5, "ego (camera)", ha="center", fontsize=9)

    # FOV rays: image edge cols map to ground at growing width
    half_fov_px = CX_PX  # left/right edge of image
    for z in [5, 50]:
        half_width_m = half_fov_px * z / FX
        ax.plot([-half_width_m, half_width_m], [z, z], "b-", alpha=0.4, lw=1)
    # the trapezoid: ground region covered at z in [5, 60]
    z_near, z_far = 5, 60
    hw_near = half_fov_px * z_near / FX
    hw_far = half_fov_px * z_far / FX
    trap = Polygon([(-hw_near, z_near), (hw_near, z_near),
                    (hw_far, z_far), (-hw_far, z_far)],
                   closed=True, facecolor="lightblue", edgecolor="blue",
                   alpha=0.35, linestyle="--")
    ax.add_patch(trap)
    ax.text(0, 30, "trapezoidal ground\nfootprint (grows with z)", ha="center",
            fontsize=9, color="blue")

    # same car at z=5 and z=50 (same physical size 2x4 m)
    for z, lbl, col in [(5, "car @ 5m", "green"), (50, "car @ 50m", "orange")]:
        ax.add_patch(Rectangle((-1, z), 2, 4, facecolor=col, alpha=0.8))
        ax.text(0, z + 4.5, lbl, ha="center", fontsize=9, color=col)
    ax.text(-25, 50, "same car,\nsame 2x4 m", fontsize=9, color="darkred",
            ha="center")


def draw_bev(ax):
    ax.set_title("BEV grid (square)\ncell = fixed 0.2m x 0.2m patch of ground", fontsize=11)
    g = int(2 * RANGE / RES)
    # show a zoomed-in region for clarity, not the full 320x320
    span = 30  # cells each side from center
    ax.set_xlim(-span, span); ax.set_ylim(-span, span)
    ax.set_aspect("equal")
    ax.set_xlabel("col c  (x / res)")
    ax.set_ylabel("row r  (z / res)")
    # grid
    for i in range(-span, span + 1):
        ax.axvline(i, color="gray", lw=0.3, alpha=0.5)
        ax.axhline(i, color="gray", lw=0.3, alpha=0.5)
    # ego at center
    ax.plot(0, 0, "k^", ms=12)
    ax.text(0, -2, "ego", ha="center", fontsize=9)
    # same car at z=5m (25 cells up) and z=50m (250 cells -- off screen, draw arrow)
    # car = 2m x 4m = 10 x 20 cells
    car_w, car_l = 10, 20
    ax.add_patch(Rectangle((-car_w/2, 25 - car_l/2), car_w, car_l,
                           facecolor="green", alpha=0.8, edgecolor="black"))
    ax.text(0, 25 + car_l/2 + 1.5, "car @ 5m\n(10x20 cells)", ha="center", fontsize=9, color="green")
    # distant car: can't show at 250 cells (off-grid), so indicate with arrow
    ax.annotate("car @ 50m would be at row 250\n(same 10x20 cells, just higher up)\n-- off this zoom",
                xy=(0, span-2), xytext=(18, span-8), fontsize=8, color="orange",
                arrowprops=dict(arrowstyle="->", color="orange"))
    ax.text(-span+2, -span+3, "each cell = 0.2m x 0.2m\nno 1/z, no stretch", fontsize=9,
            color="blue")


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    draw_camera(ax1)
    draw_bev(ax2)
    fig.suptitle("BEV vs perspective: the trapezoid is the camera's original sin",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = Path("notebooks/viz_bev_vs_perspective.png")
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved: {out.resolve()}")
    plt.close()


if __name__ == "__main__":
    main()