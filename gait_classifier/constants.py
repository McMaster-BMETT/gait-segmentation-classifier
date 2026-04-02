"""Constants for gait segmentation classifier."""

from enum import IntEnum


class GaitPhase(IntEnum):
    """6 gait phases with their ID mapping."""
    INITIAL_CONTACT = 0
    LOADING_RESPONSE = 1
    MID_STANCE = 2
    TERMINAL_STANCE = 3
    INITIAL_SWING = 4
    TERMINAL_SWING = 5


NUM_PHASES = len(GaitPhase)

# Gait cycle percentage boundaries for each phase
PHASE_BOUNDARIES = {
    GaitPhase.INITIAL_CONTACT: (0.00, 0.02),
    GaitPhase.LOADING_RESPONSE: (0.02, 0.12),
    GaitPhase.MID_STANCE: (0.12, 0.31),
    GaitPhase.TERMINAL_STANCE: (0.31, 0.50),
    GaitPhase.INITIAL_SWING: (0.50, 0.73),
    GaitPhase.TERMINAL_SWING: (0.73, 1.00),
}

PHASE_NAMES = [phase.name.replace("_", " ").title() for phase in GaitPhase]

# Sensor channel indices
CHANNEL_NAMES = [
    "imu_roll", "imu_pitch", "imu_yaw",
    "mag_x", "mag_y", "mag_z",
    "emg_quad", "emg_ham",
]
NUM_CHANNELS = len(CHANNEL_NAMES)

# Sensor baseline values (from sample data)
IMU_ROLL_BASELINE = 96.0    # degrees
IMU_PITCH_BASELINE = 306.0  # degrees
IMU_YAW_BASELINE = 180.0    # degrees

MAG_BASELINE = {"x": 0.0, "y": 0.0, "z": 0.0}  # mT

EMG_QUAD_BASELINE = 1900  # ADC counts (12-bit, 0-4095)
EMG_HAM_BASELINE = 130    # ADC counts

# Sensor ranges
EMG_ADC_MIN = 0
EMG_ADC_MAX = 4095
IMU_ANGLE_MIN = 0.0
IMU_ANGLE_MAX = 360.0

# BLE constants
BLE_DEVICE_NAME = "ESP32_SENSORS"
BLE_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
BLE_TX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

# Sampling
DEFAULT_SAMPLING_RATE = 50  # Hz

# Healing stages
HEALING_STAGES = [
    "stage1_injured",
    "stage2_early_rehab",
    "stage3_late_rehab",
    "stage4_healthy",
]
