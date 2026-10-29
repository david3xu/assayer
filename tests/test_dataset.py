import pytest

from profiles.hylogger import dataset
from profiles.hylogger.reader import Sample

ROWS = [
    Sample(depth_m=10.5, valid=True, swir_group="CHLORITE", tir_group="SILICA"),
    Sample(depth_m=10.5, valid=False, swir_group=None, tir_group=None),
]


def test_round_trips_through_bytes(tmp_path):
    path = tmp_path / "out.csv"
    dataset.write(ROWS, path)
    assert dataset.read(path) == ROWS


def test_the_same_rows_always_give_the_same_bytes():
    # Rule 2: same plan and seed, same bytes. A float that formatted differently
    # on another machine would break the hash that every check result is bound to.
    assert dataset.to_csv_bytes(ROWS) == dataset.to_csv_bytes(list(ROWS))
    assert b"\r" not in dataset.to_csv_bytes(ROWS)
    assert dataset.to_csv_bytes([Sample(1.0, True, None, None)]).endswith(
        b"1.00000,true,,\n")


def test_a_null_survives_the_round_trip(tmp_path):
    path = tmp_path / "out.csv"
    dataset.write([Sample(1.0, True, None, "SILICA")], path)
    back, = dataset.read(path)
    assert back.swir_group is None
    assert back.tir_group == "SILICA"


def test_wrong_columns_are_refused(tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_text("depth_m,valid\n1.0,true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected columns"):
        dataset.read(path)
