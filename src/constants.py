# Global design parameters
GAMMA:      float = 1.4
EXIT_MACH:  float = 2.4
RAD_THROAT: float = 1.0
N_LINES:    int   = 25

# Global method switch for the inverse Prandtl-Meyer function
METHOD: str = 'newton'

# Display design information to the terminal
INFO: bool = False

# Save upper wall data to .csv
SAVE: bool = False
PATH: str  = '../data/example.csv'

# Plot nozzle contour
PLOT: bool = True
