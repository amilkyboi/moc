# module constants
'''
Nozzle design parameters and various other options.
'''

# Design parameters
GAMMA:      float = 1.4
EXIT_MACH:  float = 2.4
RAD_THROAT: float = 1.0
N_LINES:    int   = 25
# Pressure at the exit of the nozzle (kPa)
EXIT_PRES:  float = 200
# External back pressure (kPa)
BACK_PRES:  float = 100

# Method switch for the inverse Prandtl-Meyer function
METHOD: str = 'newton'

# Save upper wall data to .csv
SAVE:      bool = False
DATA_PATH: str  = '../../data/example.csv'

# Plot nozzle contour
PLOT:     bool = True
IMG_PATH: str  = '../../img/example.webp'

# Screen resolution for plotting
RES: tuple = (1920, 1080)
