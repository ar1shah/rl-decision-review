"""Field geometry constants and drawing (ROADMAP 1.5). No replay files needed."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rldr.viz import field


def test_outline_is_symmetric_octagon():
    pts = field.field_outline()
    assert pts.shape == (8, 2)
    as_set = {tuple(p) for p in pts}
    assert as_set == {(-x, y) for x, y in pts} == {(x, -y) for x, y in pts}
    assert np.abs(pts[:, 0]).max() == field.SIDE_WALL_X
    assert np.abs(pts[:, 1]).max() == field.BACK_WALL_Y


def test_corner_cuts_are_45_degrees_on_the_corner_plane():
    pts = field.field_outline()
    edges = list(zip(pts, np.roll(pts, -1, axis=0), strict=True))
    diagonal = [(a, b) for a, b in edges if a[0] != b[0] and a[1] != b[1]]
    assert len(diagonal) == 4
    for a, b in diagonal:
        dx, dy = b - a
        assert abs(dx) == pytest.approx(abs(dy))  # 45 degrees
        for p in (a, b):
            assert abs(p[0]) + abs(p[1]) == pytest.approx(field.CORNER_CUT)


def test_goal_dimensions():
    assert 2 * field.GOAL_HALF_WIDTH == pytest.approx(1786)
    for team in (0, 1):
        x0, y0, w, h = field.goal_box(team)
        assert (x0, w, h) == (-field.GOAL_HALF_WIDTH, 2 * field.GOAL_HALF_WIDTH, field.GOAL_DEPTH)
    assert field.goal_box(0)[1] + field.GOAL_DEPTH == -field.BACK_WALL_Y  # Blue at -y
    assert field.goal_box(1)[1] == field.BACK_WALL_Y


def test_standard_pads():
    pads = field.STANDARD_PADS
    assert len(pads) == 34
    assert (pads["size"] == "big").sum() == 6
    big = {(x, y) for x, y in pads.loc[pads["size"] == "big", ["x", "y"]].to_numpy()}
    assert big == {(sx * 3072.0, sy * 4096.0) for sx in (-1, 1) for sy in (-1, 1)} | {
        (-3584.0, 0.0),
        (3584.0, 0.0),
    }
    xy = {(x, y) for x, y in pads[["x", "y"]].to_numpy()}
    assert xy == {(-x, y) for x, y in xy} == {(x, -y) for x, y in xy}  # mirror-symmetric
    inside = np.abs(pads["x"]) + np.abs(pads["y"]) < field.CORNER_CUT
    assert inside.all()


def test_pads_from_meta_falls_back_to_standard():
    assert field.pads_from_meta({}).equals(field.STANDARD_PADS)
    one = [{"pad_id": "p", "x": 1.0, "y": 2.0, "z": 70.0, "size": "small"}]
    assert field.pads_from_meta({"boost_pads": one}).to_dict("records") == [
        {"x": 1.0, "y": 2.0, "z": 70.0, "size": "small"}
    ]


def test_hitboxes():
    assert set(field.HITBOXES) == {"Octane", "Dominus", "Plank", "Breakout", "Hybrid", "Merc"}
    assert field.hitbox_for("Dominus") == field.HITBOXES["Dominus"]
    assert field.hitbox_for("Mystery") == field.HITBOXES["Octane"]
    assert field.hitbox_for(None) == field.HITBOXES["Octane"]
    for length, width, height in field.HITBOXES.values():
        assert 110 < length < 135 and 75 < width < 86 and 28 < height < 43


def test_draw_field_headless(tmp_path):
    fig, (top, side) = plt.subplots(2, 1)
    field.draw_field(top)
    field.draw_side_field(side)
    out = tmp_path / "field.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.stat().st_size > 1000
