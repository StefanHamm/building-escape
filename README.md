# Modeling and Simulation Project
Group 28 - Building Escape
Supervisor: Martin Bicher martin.bicher@tuwien.ac.at

- Stefan Hamm
- Manal Bourbigua
- Gabriel Kitzberger
- Manuel Kisser

## Description
Defined in [Building Escape.pdf](Building&#32;Escape.pdf)

## Setup conda environment
1. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   ```
## Floor planner tool
https://stefanhamm.github.io/building-escape/

## Task 1: Static Floor Field Calculation

Run following script to calculate static floor fields for all floor plans in the `data/floorPlans` folder:
```bash
python src/processFloorPlans.py
```
The SSF is saved as a numpy array in the `data/floorPlansSSF` folder.
Visualizations are automatically created and saved in the `data/floorPlanSSFVisualizations` folder.