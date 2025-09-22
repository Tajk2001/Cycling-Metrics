from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class AthleteProfile:
    name: str
    ftp_watts: float
    lthr_bpm: float
    weight_kg: float
    max_hr_bpm: Optional[float] = None
    resting_hr_bpm: Optional[float] = None
    notes: Optional[str] = None


def load_athlete_profile(athlete_dir: Path) -> AthleteProfile:
    """Load athlete profile from profile.json in their directory."""
    profile_path = athlete_dir / "profile.json"
    
    if not profile_path.exists():
        # Create default profile
        default_profile = AthleteProfile(
            name=athlete_dir.name,
            ftp_watts=250.0,  # Default FTP
            lthr_bpm=170.0,   # Default LTHR
            weight_kg=70.0,   # Default weight
            notes="Default profile - please update with actual values"
        )
        save_athlete_profile(athlete_dir, default_profile)
        return default_profile
    
    try:
        with open(profile_path, 'r') as f:
            data = json.load(f)
        return AthleteProfile(**data)
    except Exception as e:
        # If corrupted, create default
        default_profile = AthleteProfile(
            name=athlete_dir.name,
            ftp_watts=250.0,
            lthr_bpm=170.0,
            weight_kg=70.0,
            notes=f"Restored default profile due to error: {e}"
        )
        save_athlete_profile(athlete_dir, default_profile)
        return default_profile


def save_athlete_profile(athlete_dir: Path, profile: AthleteProfile) -> None:
    """Save athlete profile to profile.json in their directory."""
    profile_path = athlete_dir / "profile.json"
    athlete_dir.mkdir(parents=True, exist_ok=True)
    
    with open(profile_path, 'w') as f:
        json.dump(asdict(profile), f, indent=2)
