import pyreadr
import numpy as np

# Convert RDS to numpy and save as .npy for Python spectral analysis
for artefact in ["spherical", "step"]:
    rds_path = f"results/r_btri_stage5_15000/{artefact}/{artefact}_mean_surface.rds"
    npy_path = f"results/r_btri_stage5_15000/{artefact}/{artefact}_mean_surface.npy"
    result = pyreadr.read_r(rds_path)
    # Assume the first object is the matrix
    arr = next(iter(result.values()))
    np.save(npy_path, arr)
    print(f"Saved {npy_path}")
