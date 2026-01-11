import numpy as np

from helper import findExits
from loader import loadFloorPlan
import os
from sff import calculateSFF
from visualize import visualizeFloorPlansWithSFF

# data/floorPlans
floorplans_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'floorPlans')
export_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'floorPlansSSF')
visualizations_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'floorPlansSSFVisualizations')

if __name__ == "__main__":
    # ensure export dir exists
    os.makedirs(export_dir, exist_ok=True)

    for filename in os.listdir(floorplans_dir):
        if filename.endswith('.fplan'):
            filepath = os.path.join(floorplans_dir, filename)
            floorplan = loadFloorPlan(filepath)

            exits = findExits(floorplan)
            for exit in exits:
                export_path = os.path.join(export_dir, filename.replace('.fplan', f'_sff_{exit[0]}_{exit[0]}.npy'))
                with open(export_path, 'wb') as f:
                    np.save(f, calculateSFF(floorplan, [exit]))

            export_path = os.path.join(export_dir, filename.replace('.fplan', f'_sff.npy'))
            with open(export_path, 'wb') as f:
                np.save(f, calculateSFF(floorplan, exits))

    visualizeFloorPlansWithSFF(floorplans_dir, export_dir, show_gradients=True, export_folder=visualizations_dir)
