# Visulize all floorplan maps and their corresponding static floor fields by overlaying them
# make it clickable using matplotlib

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,FancyArrow

from loader import loadFloorPlan


def floorplan_to_rgb(fplan: np.ndarray) -> np.ndarray:
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

def _floorplan_to_rgb(fplan: np.ndarray) -> np.ndarray:
    return floorplan_to_rgb(fplan)


def _compute_direction_to_lowest_neighbor(sff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each cell, compute direction vector pointing to the lowest neighboring cell.
    
    Returns:
        dir_row: row direction component (-1, 0, or 1)
        dir_col: column direction component (-1, 0, or 1)
    """
    rows, cols = sff.shape
    dir_row = np.zeros_like(sff, dtype=float)
    dir_col = np.zeros_like(sff, dtype=float)
    
    # 8-connected neighbors: all 8 surrounding cells
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]
    
    for r in range(rows):
        for c in range(cols):
            if not np.isfinite(sff[r, c]):
                continue
            
            current_val = sff[r, c]
            lowest_val = current_val
            lowest_dr = 0
            lowest_dc = 0
            
            # Check all neighbors
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbor_val = sff[nr, nc]
                    if np.isfinite(neighbor_val) and neighbor_val < lowest_val:
                        lowest_val = neighbor_val
                        lowest_dr = dr
                        lowest_dc = dc
            
            # Normalize direction
            if lowest_dr != 0 or lowest_dc != 0:
                mag = np.sqrt(lowest_dr**2 + lowest_dc**2)
                dir_row[r, c] = lowest_dr / mag
                dir_col[r, c] = lowest_dc / mag
    
    return dir_row, dir_col

def _compute_direction_to_horizontal_gradient(sff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each cell, compute direction vector pointing to the horizontal gradient.
    
    Returns:
        dir_row: row direction component (-1, 0, or 1)
        dir_col: column direction component (-1, 0, or 1)
    """
    rows, cols = sff.shape
    dir_row = np.zeros_like(sff, dtype=float)
    dir_col = np.zeros_like(sff, dtype=float)
    
    for r in range(rows):
        for c in range(cols):
            if not np.isfinite(sff[r, c]):
                continue
            
            current_val = sff[r, c]
            # Check left neighbor
            if c > 0 and np.isfinite(sff[r, c - 1]) and sff[r, c - 1] < current_val:
                dir_col[r, c] = -1.0
            # Check right neighbor
            if c < cols - 1 and np.isfinite(sff[r, c + 1]) and sff[r, c + 1] < current_val:
                dir_col[r, c] = 1.0
    
    return dir_row, dir_col

def _compute_direction_to_vertical_gradient(sff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each cell, compute direction vector pointing to the vertical gradient.
    
    Returns:
        dir_row: row direction component (-1, 0, or 1)
        dir_col: column direction component (-1, 0, or 1)
    """
    rows, cols = sff.shape
    dir_row = np.zeros_like(sff, dtype=float)
    dir_col = np.zeros_like(sff, dtype=float)
    
    for r in range(rows):
        for c in range(cols):
            if not np.isfinite(sff[r, c]):
                continue
            
            current_val = sff[r, c]
            # Check upper neighbor
            if r > 0 and np.isfinite(sff[r - 1, c]) and sff[r - 1, c] < current_val:
                dir_row[r, c] = -1.0
            # Check lower neighbor
            if r < rows - 1 and np.isfinite(sff[r + 1, c]) and sff[r + 1, c] < current_val:
                dir_row[r, c] = 1.0
    
    return dir_row, dir_col


def _draw_gradient_arrows(ax, fplan: np.ndarray, sff: np.ndarray, 
                         dir_row: np.ndarray, dir_col: np.ndarray, 
                         arrow_spacing: int = 1):
    """Helper function to draw gradient arrows on an axis."""
    rows, cols = fplan.shape
    
    for r in range(0, rows, arrow_spacing):
        for c in range(0, cols, arrow_spacing):
            if np.isfinite(sff[r, c]) and fplan[r, c] != 'W':
                dr = dir_row[r, c]
                dc = dir_col[r, c]
                mag = np.sqrt(dr**2 + dc**2)
                
                if mag > 0.01:  # Only draw if direction is non-negligible
                    scale = 0.3
                    dx = dc * scale
                    dy = dr * scale
                    
                    arrow = FancyArrowPatch(
                        (c, r), (c + dx, r + dy),
                        arrowstyle='->', mutation_scale=15, linewidth=1.0,
                        color='white', alpha=0.7, zorder=10
                    )
                    ax.add_patch(arrow)
                    
                    

def plot_heatmap(mat: np.ndarray, xvals=None, yvals=None, xlabel="xi", ylabel="k",
                 title="Mean evacuation time", out_path: str | None = None):
    """Simple heatmap saver — does not alter existing project files."""
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(mat, origin="lower", interpolation="nearest", aspect="auto")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    if xvals is not None:
        ax.set_xticks(np.arange(len(xvals))); ax.set_xticklabels([str(x) for x in xvals])
    if yvals is not None:
        ax.set_yticks(np.arange(len(yvals))); ax.set_yticklabels([str(y) for y in yvals])
    cbar = plt.colorbar(im, ax=ax); cbar.set_label(title)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path)
        plt.close(fig)
    return fig, ax




def visualizeFloorPlansWithSFF(floorplans_dir: str, sff_dir: str, show_gradients: bool = True, export_folder: str = None, gradient_only_folder: str = None):
    """
    Visualizes all floorplans in floorplans_dir.
    Plots the floorplan, the SFF heatmap (read from sff_dir), and optional gradients.
    Clicking on any image prints the cell coordinates, the floorplan character
    and the SFF value (if available) and places a marker.
    
    Args:
        floorplans_dir: Directory containing .fplan files
        sff_dir: Directory containing _sff.npy files
        show_gradients: If True, overlay gradient arrows pointing towards 0
        export_folder: If provided, save the plot to this folder instead of showing.
        gradient_only_folder: If provided, save a separate plot with only the combined gradient to this folder.
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

        if sff is None:
            continue

        rows, cols = fplan.shape
        rgb = _floorplan_to_rgb(fplan)
        arrow_spacing = 1

        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(max(12, cols / 2.5), max(12, rows / 2.5)))
        fig.suptitle(filename, fontsize=16)

        # Compute all gradient directions
        if show_gradients:
            dir_neighbor_row, dir_neighbor_col = _compute_direction_to_lowest_neighbor(sff)
            dir_horiz_row, dir_horiz_col = _compute_direction_to_horizontal_gradient(sff)
            dir_vert_row, dir_vert_col = _compute_direction_to_vertical_gradient(sff)
            # Combined: normalize both components together
            dir_combined_row = dir_vert_row
            dir_combined_col = dir_horiz_col

        # Helper function to setup each subplot
        def setup_subplot(ax, title, dir_row_data, dir_col_data):
            # mask walls / inf values so they are transparent in the overlay
            sff_masked = np.ma.array(sff, mask=~np.isfinite(sff))
            ax.imshow(rgb, origin='upper', interpolation='nearest')
            ax.imshow(sff_masked, cmap='viridis', origin='upper', alpha=0.6, interpolation='nearest')
            
            if show_gradients:
                _draw_gradient_arrows(ax, fplan, sff, dir_row_data, dir_col_data, arrow_spacing)
            
            ax.set_title(title)
            ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
            ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.3)
            ax.set_xticks([])
            ax.set_yticks([])

        # Top-left: Neighbor gradient
        setup_subplot(axes[0, 0], 'Neighbor Gradient', dir_neighbor_row, dir_neighbor_col)
        
        # Top-right: Horizontal gradient
        setup_subplot(axes[0, 1], 'Horizontal Gradient', dir_horiz_row, dir_horiz_col)
        
        # Bottom-left: Vertical gradient
        setup_subplot(axes[1, 0], 'Vertical Gradient', dir_vert_row, dir_vert_col)
        
        # Bottom-right: Combined gradient
        setup_subplot(axes[1, 1], 'Combined Gradient', dir_combined_row, dir_combined_col)

        # Add a shared colorbar
        sff_masked = np.ma.array(sff, mask=~np.isfinite(sff))
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=sff_masked.min(), vmax=sff_masked.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label('SFF value')

        plt.tight_layout(rect=[0, 0, 0.9, 0.96])
        
        if export_folder:
            save_file = os.path.join(export_folder, filename.replace('.fplan', '_visualization.png'))
            plt.savefig(save_file, dpi=300)
            print(f"Saved visualization to {save_file}")
        else:
            plt.show()
        
        if gradient_only_folder:
            os.makedirs(gradient_only_folder, exist_ok=True)
            save_file = os.path.join(gradient_only_folder, filename.replace('.fplan', '_visualization.png'))
            
            # Create a separate figure for the single plot
            fig_single, ax_single = plt.subplots(figsize=(max(8, cols / 2.5), max(8, rows / 2.5)))
            
            # Use the existing helper to plot the Combined Gradient
            setup_subplot(ax_single, 'Combined Gradient', dir_combined_row, dir_combined_col)
            
            # Add colorbar to this single figure
            sff_masked = np.ma.array(sff, mask=~np.isfinite(sff))
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=sff_masked.min(), vmax=sff_masked.max()))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax_single)
            cbar.set_label('SFF value')
            
            plt.tight_layout()
            fig_single.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"Saved single gradient visualization to {save_file}")
            plt.close(fig_single)

        # Clean up the main figure if we're not showing it interactively
        if export_folder:
            plt.close(fig)

        
from sharedClasses import AgentState

def print_agents_on_floorplan(fplan: np.ndarray, agents: list[AgentState], step: int = 0, export_path: Optional[str] = None, base_rgb: Optional[np.ndarray] = None):
    """Visualize agents on a floorplan as a still image.
    
    Args:
        fplan: The floorplan character array (used for dimensions)
        agents: List of AgentState objects with x, y coordinates
        step: Current simulation step
        export_path: Optional path to save the image. If None, displays with plt.show()
        base_rgb: Optional pre-computed RGB image of the floorplan. If None, it will be computed.
    """
    # Use pre-computed RGB if provided, otherwise compute it
    if base_rgb is not None:
        rgb = base_rgb
    else:
        rgb = _floorplan_to_rgb(fplan)
    
    # Create figure and display floorplan
    # Reduced figure size slightly to help with memory and video encoding issues
    width, height = fplan.shape[1] / 3.0, fplan.shape[0] / 3.0
    fig, ax = plt.subplots(figsize=(max(8, width), max(8, height)))
    ax.imshow(rgb, origin='upper', interpolation='nearest')
    
    # Draw agents as blue circles
    for agent in agents:
        if agent.state.done:
            continue  # Skip agents that have exited
        circle = plt.Circle((agent.state.y,agent.state.x), radius=0.3, color='blue', alpha=0.8, zorder=5)
        ax.add_patch(circle)
        # Add agent index text
        ax.text(agent.state.y,agent.state.x, str(agents.index(agent)), color='white', 
                ha='center', va='center', fontsize=8, fontweight='bold', zorder=6)
    
    # Setup grid and labels
    rows, cols = fplan.shape
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Step: {step} | Agents on Floorplan ({len(agents)} agents)')
    
    # Save or show
    if export_path:
        plt.savefig(export_path, dpi=100, bbox_inches='tight') # Reduced DPI to Help with memory and video encoding
        print(f"Saved agent visualization to {export_path}")
        plt.close(fig) # Explicitly close the figure to free memory
    else:
        plt.show()

import imageio
def create_video_from_steps(image_folder: str, output_path: Optional[str] = None, fps: int = 10):
    """Create a video from step images in a folder.
    
    Reads images named 0.png, 1.png, ... from the specified folder
    and creates a video file from them.
    
    Args:
        image_folder: Path to folder containing step images (0.png, 1.png, ...)
        output_path: Path to save the video. If None, saves as 'simulation.mp4' in the image folder
        fps: Frames per second for the video (default: 10)
    """
    if not os.path.exists(image_folder):
        print(f"Image folder not found: {image_folder}")
        return
    
    # Get all PNG files and sort them numerically
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')],
                        key=lambda x: int(x.replace('.png', '')))
    
    if not image_files:
        print(f"No PNG images found in {image_folder}")
        return
    
    # Determine output path
    if output_path is None:
        output_path = os.path.join(image_folder, 'simulation.mp4')
    
    # Read images and create video
    print(f"Creating video from {len(image_files)} images...")
    
    # Use macro_block_size=16 to ensure dimensions are compatible with h.264 encoder
    writer = imageio.get_writer(output_path, fps=fps, macro_block_size=16)
    
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = imageio.imread(image_path)
        
        # Check and resize if dimensions are odd
        h, w = image.shape[:2]
        if h % 2 != 0 or w % 2 != 0:
             # Basic trim to make even dimensions
             new_h = h - (h % 2)
             new_w = w - (w % 2)
             image = image[:new_h, :new_w]

        writer.append_data(image)
    
    writer.close()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    import os
    floorplans_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'floorPlans')
    sff_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'floorPlansSSF')
    visualizeFloorPlansWithSFF(floorplans_dir, sff_dir)