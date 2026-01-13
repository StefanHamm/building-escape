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
# visdir for only gradient as in the task desc
gradients_only_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'floorPlansSSFVisualizations/GradientsOnly')
if __name__ == "__main__":
    # ensure export dir exists
    os.makedirs(export_dir, exist_ok=True)

    for filename in os.listdir(floorplans_dir):
        if not filename.endswith('.fplan'):
            continue
        filepath = os.path.join(floorplans_dir, filename)
        floorplan = loadFloorPlan(filepath)

        exits = findExits(floorplan)
        exits_to_process = exits.copy()
        exits_grouped = []
        while len(exits_to_process) > 0:
            r, c = exits_to_process.pop()
            queue = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]

            group = [(r, c)]
            while len(queue) > 0:
                r, c = queue.pop(0)
                if (r, c) in exits_to_process:
                    group.append((r, c))
                    exits_to_process.remove((r, c))
                    queue.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])
            exits_grouped.append(group)

        for eg in exits_grouped:
            export_path = os.path.join(export_dir,
                                       filename.replace('.fplan', f'_sff_{eg[0][0]}_{eg[0][1]}.npy'))
            with open(export_path, 'wb') as f:
                np.save(f, calculateSFF(floorplan, eg))

        export_path = os.path.join(export_dir, filename.replace('.fplan', f'_sff.npy'))
        with open(export_path, 'wb') as f:
            np.save(f, calculateSFF(floorplan, exits))

    visualizeFloorPlansWithSFF(floorplans_dir, export_dir, show_gradients=True, export_folder=visualizations_dir,
                               gradient_only_folder=gradients_only_dir)
