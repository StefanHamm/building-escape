# Visulize all floorplan maps and their corresponding static floor fields by overlaying them
# make it clickable using matplotlib

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from loader import loadFloorPlan


def _floorplan_to_rgb(fplan: np.ndarray) -> np.ndarray:
    """Convert a floorplan char array to an RGB image (rows, cols, 3).
    Mapping:
      'W' -> black (wall)
      'F' -> white (floor)
      'E' -> red   (exit)
      other -> gray
    """
    rows, cols = fplan.shape
    img = np.zeros((rows, cols, 3), dtype=float)

    for r in range(rows):
        for c in range(cols):
            ch = fplan[r, c]
            if ch == 'W':
                img[r, c] = (0.0, 0.0, 0.0)
            elif ch == 'F':
                img[r, c] = (1.0, 1.0, 1.0)
            elif ch == 'E':
                img[r, c] = (1.0, 0.0, 0.0)
            else:
                img[r, c] = (0.7, 0.7, 0.7)
    return img


def visualizeFloorPlansWithSFF(floorplans_dir: str, sff_dir: str):
    """Visualize all .fplan files in floorplans_dir with their corresponding
    static floor field (_sff.npy) files in sff_dir.

    Clicking on the image prints the cell coordinates, the floorplan character
    and the SFF value (if available) and places a marker.
    """
    os.makedirs(sff_dir, exist_ok=True)

    filenames = sorted([f for f in os.listdir(floorplans_dir) if f.endswith('.fplan')])
    if not filenames:
        print(f"No .fplan files found in {floorplans_dir}")
        return

    for filename in filenames:
        fpath = os.path.join(floorplans_dir, filename)
        try:
            fplan = loadFloorPlan(fpath)
        except Exception as e:
            print(f"Failed to load floorplan '{filename}': {e}")
            continue

        # load corresponding sff if available
        sff_path = os.path.join(sff_dir, filename.replace('.fplan', '_sff.npy'))
        sff: Optional[np.ndarray] = None
        if os.path.exists(sff_path):
            try:
                sff = np.load(sff_path)
            except Exception as e:
                print(f"Failed to load SFF for '{filename}': {e}")
                sff = None

        rows, cols = fplan.shape
        rgb = _floorplan_to_rgb(fplan)

        fig, ax = plt.subplots(figsize=(max(6, cols / 5), max(6, rows / 5)))
        ax.set_title(filename)

        # show base floorplan
        im_floor = ax.imshow(rgb, origin='upper', interpolation='nearest')

        # overlay SFF if present
        im_sff = None
        cbar = None
        if sff is not None:
            # mask walls / inf values so they are transparent in the overlay
            sff_masked = np.ma.array(sff, mask=~np.isfinite(sff))
            im_sff = ax.imshow(sff_masked, cmap='viridis', origin='upper',
                               alpha=0.6, interpolation='nearest')
            cbar = fig.colorbar(im_sff, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('SFF value')

        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.3)
        ax.set_xticks([])
        ax.set_yticks([])

        # interactive click handler
        marker = {'artist': None}

        def onclick(event):
            if event.inaxes is not ax or event.xdata is None or event.ydata is None:
                return
            # x -> column, y -> row; origin='upper' so floor array row = int(y)
            col = int(event.xdata + 0.5)
            row = int(event.ydata + 0.5)
            # clamp
            col = max(0, min(cols - 1, col))
            row = max(0, min(rows - 1, row))

            ch = fplan[row, col]
            sff_val = None
            if sff is not None and np.isfinite(sff[row, col]):
                sff_val = float(sff[row, col])

            print(f"Clicked on (row={row}, col={col}) char='{ch}' sff={sff_val}")

            # remove previous marker
            if marker['artist'] is not None:
                try:
                    marker['artist'].remove()
                except Exception:
                    pass
            # place new marker
            artist = ax.plot(col, row, marker='x', color='cyan', markersize=12, markeredgewidth=2)[0]
            marker['artist'] = artist
            # annotate value briefly
            ann_text = f"{ch}"
            if sff_val is not None:
                ann_text += f"\n{round(sff_val, 3)}"
            ann = ax.annotate(ann_text, (col, row), color='yellow', weight='bold',
                              fontsize=9, ha='left', va='bottom')

            # redraw and pause to show annotation
            fig.canvas.draw_idle()

            # remove annotation after a short time (use a timer)
            def _remove_ann(evt=None):
                try:
                    ann.remove()
                    fig.canvas.draw_idle()
                except Exception:
                    pass
            fig.canvas.new_timer(interval=1500, callbacks=[(_remove_ann, (), {})]).start()

        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        plt.show()

        # disconnect handler after window closed
        try:
            fig.canvas.mpl_disconnect(cid)
        except Exception:
            pass


if __name__ == "__main__":
    import os
    floorplans_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'floorPlans')
    sff_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'floorPlansSSF')
    visualizeFloorPlansWithSFF(floorplans_dir, sff_dir)