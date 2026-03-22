"""Unit tests for PAD helpers (no model weights required)."""

from __future__ import annotations

import numpy as np

from src.pad.liveness import (
    EAR_BLINK_THRESHOLD,
    EAR_CONSEC_FRAMES,
    BlinkDetector,
    MoireDetector,
)


def _landmarks_with_eye_geometry(
    *,
    left_vertical_span: float,
    right_vertical_span: float,
    horizontal_span: float = 10.0,
) -> np.ndarray:
    """Build (68, 3) landmarks with synthetic left/right eyes (iBUG ordering)."""
    lm = np.zeros((68, 3), dtype=np.float32)
    # Right eye 36-41: outer left, top outer, top inner, inner, bottom inner, bottom outer
    base_rx, base_ry = 30.0, 40.0
    half_h = horizontal_span / 2.0
    rv = right_vertical_span / 2.0
    r_idx = [36, 37, 38, 39, 40, 41]
    r_pts = [
        (base_rx - half_h, base_ry),
        (base_rx - half_h / 2, base_ry - rv),
        (base_rx + half_h / 2, base_ry - rv),
        (base_rx + half_h, base_ry),
        (base_rx + half_h / 2, base_ry + rv),
        (base_rx - half_h / 2, base_ry + rv),
    ]
    for i, p in zip(r_idx, r_pts, strict=True):
        lm[i, 0] = p[0]
        lm[i, 1] = p[1]
    # Left eye 42-47 — mirrored to the right side of the face
    base_lx = 60.0
    lv = left_vertical_span / 2.0
    l_idx = [42, 43, 44, 45, 46, 47]
    l_pts = [
        (base_lx - half_h, base_ry),
        (base_lx - half_h / 2, base_ry - lv),
        (base_lx + half_h / 2, base_ry - lv),
        (base_lx + half_h, base_ry),
        (base_lx + half_h / 2, base_ry + lv),
        (base_lx - half_h / 2, base_ry + lv),
    ]
    for i, p in zip(l_idx, l_pts, strict=True):
        lm[i, 0] = p[0]
        lm[i, 1] = p[1]
    return lm


def test_blink_detector_detects_closure_then_open_as_blink() -> None:
    """Two frames below EAR then an open eye should register one blink."""
    ear_t = EAR_BLINK_THRESHOLD
    consec = EAR_CONSEC_FRAMES
    blink = BlinkDetector(ear_threshold=ear_t, consec_frames=consec, timeout_frames=1000)

    closed = _landmarks_with_eye_geometry(left_vertical_span=1.0, right_vertical_span=1.0)
    open_eyes = _landmarks_with_eye_geometry(left_vertical_span=8.0, right_vertical_span=8.0)

    e1 = blink.update(closed)
    e2 = blink.update(closed)
    assert e1 < ear_t and e2 < ear_t
    assert not blink.blink_detected

    e3 = blink.update(open_eyes)
    assert e3 >= ear_t
    assert blink.blink_detected


def test_blink_detector_timeout_without_blink() -> None:
    blink = BlinkDetector(
        ear_threshold=EAR_BLINK_THRESHOLD,
        consec_frames=EAR_CONSEC_FRAMES,
        timeout_frames=3,
    )
    open_eyes = _landmarks_with_eye_geometry(left_vertical_span=8.0, right_vertical_span=8.0)
    for _ in range(3):
        blink.update(open_eyes)
    assert not blink.blink_detected
    assert blink.timed_out


def test_moire_detector_analyze_returns_bounded_metrics() -> None:
    det = MoireDetector()
    uniform = np.full((112, 112, 3), 128, dtype=np.uint8)
    rng = np.random.default_rng(0)
    noise = (rng.random((112, 112, 3)) * 255).astype(np.uint8)

    m_u, lbp_u, spec_u = det.analyze(uniform)
    m_n, lbp_n, spec_n = det.analyze(noise)

    for name, vals in (
        ("uniform", (m_u, lbp_u, spec_u)),
        ("noise", (m_n, lbp_n, spec_n)),
    ):
        moire, lbp, spec = vals
        assert np.isfinite(moire), name
        assert np.isfinite(lbp), name
        assert np.isfinite(spec), name
        assert 0.0 <= spec <= 1.0, name

    # Textured noise should not be identical to a flat field (FFT / LBP path smoke check)
    assert (m_u, lbp_u) != (m_n, lbp_n)
