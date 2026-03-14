"""Dashboard design constants — colors, targets, labels."""

# Stage display names (human-readable)
STAGE_DISPLAY = {
    "stage1_injured": "Injured",
    "stage2_early_rehab": "Early Rehab",
    "stage3_late_rehab": "Late Rehab",
    "stage4_healthy": "Healthy",
}

STAGE_ORDER = [
    "stage1_injured",
    "stage2_early_rehab",
    "stage3_late_rehab",
    "stage4_healthy",
]

# Gait phase color palette (6 distinct, non-rainbow)
PHASE_COLORS = [
    "#1E3A5F",  # Initial Contact  — navy
    "#2563EB",  # Loading Response  — blue
    "#7C3AED",  # Mid Stance        — purple
    "#059669",  # Terminal Stance    — green
    "#D97706",  # Initial Swing     — amber
    "#E04E39",  # Terminal Swing    — red-orange
]

# Design system colors
BG_COLOR = "#FAFAF9"
TEXT_PRIMARY = "#1A1A1A"
TEXT_SECONDARY = "#6B7280"
ACCENT = "#2563EB"
CARD_BG = "#FFFFFF"
GRID_COLOR = "#F3F4F6"

# Status colors
STATUS_GREEN = "#059669"
STATUS_AMBER = "#D97706"
STATUS_RED = "#DC2626"

# Clinical metric definitions
METRIC_DEFS = {
    "qii": {
        "label": "Quadriceps Inhibition Index",
        "short": "QII",
        "unit": "",
        "target": "< 0.10",
        "good_range": (None, 0.10),
        "warn_range": (0.10, 0.25),
        "format": ".2f",
        "radar_max": 0.5,
        "radar_invert": True,   # lower is better
    },
    "hq_functional": {
        "label": "H:Q Functional Ratio",
        "short": "H:Q Ratio",
        "unit": "",
        "target": "0.50 – 0.80",
        "good_range": (0.50, 0.80),
        "warn_range": (0.30, 0.50),
        "format": ".2f",
        "radar_max": 1.0,
        "radar_invert": False,
    },
    "swing_peak_flexion": {
        "label": "Swing Peak Flexion",
        "short": "Peak Flexion",
        "unit": "\u00b0",
        "target": "> 55\u00b0",
        "good_range": (55, None),
        "warn_range": (40, 55),
        "format": ".1f",
        "radar_max": 70.0,
        "radar_invert": False,
    },
    "lsi_temporal": {
        "label": "LSI Temporal",
        "short": "LSI",
        "unit": "%",
        "target": "> 90%",
        "good_range": (90, None),
        "warn_range": (75, 90),
        "format": ".1f",
        "radar_max": 110.0,
        "radar_invert": False,
    },
}

METRIC_ORDER = ["qii", "hq_functional", "swing_peak_flexion", "lsi_temporal"]

# Chart colors for progression lines (one per metric)
METRIC_LINE_COLORS = {
    "qii": "#DC2626",
    "hq_functional": "#2563EB",
    "swing_peak_flexion": "#7C3AED",
    "lsi_temporal": "#059669",
}

# EMG signal colors
EMG_QUAD_COLOR = "#2563EB"
EMG_HAM_COLOR = "#E04E39"
KNEE_FLEXION_COLOR = "#1A1A1A"
