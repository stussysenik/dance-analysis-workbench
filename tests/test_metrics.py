from __future__ import annotations

from dance_analysis.contracts import Point
from dance_analysis.pipeline import compute_center_of_mass, score_on_beat


def test_compute_center_of_mass_weights_torso_lower_body_more() -> None:
    joints = {
        "head": Point(x=10, y=10),
        "shoulder_l": Point(x=8, y=20),
        "shoulder_r": Point(x=12, y=20),
        "hip_l": Point(x=9, y=40),
        "hip_r": Point(x=11, y=40),
        "foot_l": Point(x=8, y=70),
        "foot_r": Point(x=12, y=70),
    }
    com = compute_center_of_mass(joints)
    assert 9.0 < com.x < 11.0
    assert 30.0 < com.y < 45.0


def test_score_on_beat_rewards_close_motion_events() -> None:
    beat_times = [0.0, 0.5, 1.0, 1.5]
    aligned = score_on_beat(beat_times, [0.02, 0.48, 1.03], 0.5)
    misaligned = score_on_beat(beat_times, [0.25, 0.75, 1.25], 0.5)
    assert aligned > misaligned
    assert 0.0 <= aligned <= 1.0
