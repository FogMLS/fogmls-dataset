# ================================================================
# SIMULATION TIME
# ================================================================
MAX_SIMULATION_TIME = 175000

# ================================================================
# DATASET SETTINGS
# ================================================================
# Maximum number of concurrent task positions per slot
MAX_JOBS = 200

# Number of features per task (task type, instruction count, data size)
JOB_ATTRIBUTES = 3

# Total resource classes: 0=Nothing, 1=Cloud, 2..MAX_RS-1=Fog nodes
# MAX_RS = number of fog nodes + 2
MAX_RS = 5

# Number of fog nodes derived from MAX_RS
NUM_FOG_NODES = MAX_RS - 2

# Total features per task position.
# Each task position records JOB_ATTRIBUTES task features only:
# task type, instruction count, data size.
# Capacity features are used internally by the GA for scheduling
# decisions but are NOT recorded in the dataset. This ensures the
# dataset captures task characteristics without directly encoding
# the scheduling decision state.
FEATURES_PER_JOB = JOB_ATTRIBUTES

# Number of sensors per fog cluster
SENSORS_PER_CLUSTER = 9

# ================================================================
# FITNESS PARAMETERS
# ================================================================
# Latency weight
ALPHA = 0.5
# Energy weight
BETA = 0.5
# Cost weight
GAMMA = 0.0

# ================================================================
# RESOURCE MANAGEMENT OPTIONS
# ================================================================
DEFAULT_RM = 0
GA_RM = 1
AI_RM = 2
RL_RM = 3
RR_RM = 4  # Capacity-aware Round Robin baseline

# Set resource management approach
RM_TYPE = 1

# ================================================================
# CLOUD SERVER
# ================================================================
CLOUD_COST_UNIT_TIME = 0.01
CLOUD_CPU_SPEED = 48000

# ================================================================
# FOG NODE HARDWARE SPECIFICATIONS
# To add a new fog node type:
#   1. Add new specs below (CPU_SPEED4, FD_RAM4, FD_BANDWIDTH4 etc)
#   2. Add CPU_SPEED4 to _cpu_speeds list in SLOT TIME section below
#   3. Add new entry to fog_node_configs list in devices.py
#   4. Update MAX_RS above
# ================================================================

# Bandwidth per fog node type (SU)
FD_BANDWIDTH1 = 2#1
FD_BANDWIDTH2 = 4#2
FD_BANDWIDTH3 = 8#3

# RAM capacity per fog node type (MB)
FD_RAM1 = 2000#1000
FD_RAM2 = 4000#2000
FD_RAM3 = 8000#3000

# CPU speed per fog node type (SU)
CPU_SPEED1 = 40#5
CPU_SPEED2 = 80#10
CPU_SPEED3 = 160#20

# Power consumption per fog node type (W)
BUSY_POWER1 = 105.0
BUSY_POWER2 = 110.0
BUSY_POWER3 = 115.339
IDLE_POWER1 = 80.0
IDLE_POWER2 = 82.0
IDLE_POWER3 = 85.0

# ================================================================
# NETWORK LATENCY (SU)
# ================================================================
LATENCY_CS_FR = 50
LATENCY_ED_FR = 1

# ================================================================
# RAM SWAP
# ================================================================
RAM_SWAP_UNIT = 1

# ================================================================
# TASK TYPE CONSTANTS
# ================================================================
SENSOR_TYPE = 1
ACTUATOR_TYPE = 5
FOG_TYPE = 6

# Application task types
CAMERA_SEN = 1
TRAFFIC_SEN = 2
VEHICLE_IDENTIFICATION = 3
TRAFFIC_LOCATION = 4
ACTION_ON_VEHICLE = 5
ACTION_CLEAR_TRAFFIC = 6

# ================================================================
# RANDOM SEED
# ================================================================
# Fixed seed for reproducibility.
# Change to None for fully random runs.
RANDOM_SEED = 42

# ================================================================
# TASK INSTRUCTION COUNT RANGES (SU)
# ================================================================
NO_INSTRUCTIONS1_MIN = 8
NO_INSTRUCTIONS1_MAX = 14
NO_INSTRUCTIONS2_MIN = 10
NO_INSTRUCTIONS2_MAX = 16
NO_INSTRUCTIONS3_MIN = 100#50#100
NO_INSTRUCTIONS3_MAX = 140#200#140
NO_INSTRUCTIONS4_MIN = 110#60#110
NO_INSTRUCTIONS4_MAX = 150#220#150
NO_INSTRUCTIONS5 = 0
NO_INSTRUCTIONS6 = 0

# ================================================================
# TASK DATA SIZE RANGES (SU)
# ================================================================
DATA_SIZE1_MIN = 400
DATA_SIZE1_MAX = 600
DATA_SIZE2_MIN = 800
DATA_SIZE2_MAX = 1200
DATA_SIZE3_MIN = 1200
DATA_SIZE3_MAX = 1800
DATA_SIZE4_MIN = 1600
DATA_SIZE4_MAX = 2400

# ================================================================
# GA PARAMETERS
# ================================================================
# Number of chromosomes must be even for crossover loop to work correctly
N_CHROMOSOMES = 8
MAX_ITERATIONS = 10
TOURNAMENT_SIZE = 3

# ================================================================
# SLOT TIME CALCULATION
# ================================================================
# SLOT_TIME is automatically calculated from hardware parameters.
# It represents the worst case execution time scaled by number of sensors.
#
# SLOT_TIME0 is a tuning constant. Adjust this value to achieve your
# desired slot duration.
# Current value gives SLOT_TIME = 35 SU.
#
# Formula to find your SLOT_TIME0:
# SLOT_TIME0 = desired_slot_time -
#              (INTERVAL_GAP * SENSORS_PER_CLUSTER + INTERVAL_GAP)
#
# When adding a new fog node type, add its CPU speed to _cpu_speeds below.

# Dynamically find worst case instruction count across all task types
_instruction_maxes = [
    NO_INSTRUCTIONS1_MAX,
    NO_INSTRUCTIONS2_MAX,
    NO_INSTRUCTIONS3_MAX,
    NO_INSTRUCTIONS4_MAX
]
MAX_INSTRUCTIONS_ANY_TASK = max(_instruction_maxes)

# Add new CPU speeds to this list when adding new fog node types
_cpu_speeds = [CPU_SPEED1, CPU_SPEED2, CPU_SPEED3]
MIN_CPU_SPEED = min(_cpu_speeds)

SLOT_TIME0 = -2.5
INTERVAL_GAP = MAX_INSTRUCTIONS_ANY_TASK / MIN_CPU_SPEED
SLOT_TIME = INTERVAL_GAP * SENSORS_PER_CLUSTER + INTERVAL_GAP + SLOT_TIME0

# ================================================================
# EXPLICIT DEVICE-TO-CLASS LABEL MAPPING
# ================================================================
# Maps device IDs to dataset class labels.
# This eliminates the fragile dependency on device creation order.
# Populated at runtime by IoTsimulation.py after all devices are created.
# Format: {device_id: class_label}
# Class labels: 0=Nothing, 1=Cloud, 2=Fog-1, 3=Fog-2, ...
DEVICE_TO_CLASS = {}