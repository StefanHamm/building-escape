from loader import loadFloorPlan
import os
from ssf import calculateSFF
from visualize import visualizeFloorPlansWithSFF


#data/floorPlans
floorplans_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'floorPlans')
export_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'floorPlansSSF')
visualizations_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'floorPlansSSFVisualizations')



if __name__ == "__main__":
    #ensure export dir exists
    os.makedirs(export_dir, exist_ok=True)

    for filename in os.listdir(floorplans_dir):
        if filename.endswith('.fplan'):
            filepath = os.path.join(floorplans_dir, filename)
            floorplan = loadFloorPlan(filepath)
            sff = calculateSFF(floorplan)
            
            #export the sff as a npy file
            export_path = os.path.join(export_dir, filename.replace('.fplan', '_sff.npy'))
            with open(export_path, 'wb') as f:
                import numpy as np
                np.save(f, sff)
    
    visualizeFloorPlansWithSFF(floorplans_dir,export_dir, show_gradients=True, export_folder=visualizations_dir)
