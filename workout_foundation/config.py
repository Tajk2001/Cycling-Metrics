"""Central config with hardcoded rider constants, mirroring SprintV1 style."""

# Rider Parameters
FTP_WATTS: float = 290.0
LTHR_BPM: float = 181.0
RIDER_MASS_KG: float = 52.0

# Test harness (single file)
# Update this to point at a specific FIT file on your machine
TEST_INPUT_PATH: str = "/Users/tajkrieger/Projects/cycling_analysis/athletes/taj/thr/zwift-activity-1957539391362580528.fit"
TEST_OUTPUT_DIR: str = "/Users/tajkrieger/Projects/cycling_analysis/cycling_data/exports"


