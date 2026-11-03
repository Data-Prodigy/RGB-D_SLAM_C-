import numpy as np

est = np.loadtxt('/home/uduakdon/Downloads/trajectory_tum_TRIAL.txt')
# If Y and Z axes are inverted relative to ground truth:
est[:, 2] *= -1.0  # Invert Z
est[:, 3] *= -1.0  # Invert Y (depending on your specific flip)
np.savetxt('/home/uduakdon/Downloads/trajectory_tum_FINALIZED.txt', est, fmt='%.6f')