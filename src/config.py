import os

# This will get the directory that contains your src folder
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Model configuration
DEFAULT_BASELINE_TIMESCALE = 365  # Annual cycle in days
DEFAULT_ELECTION_TIMESCALES = [30, 15]  # Pre-campaign (30 days) and official campaign (15 days)