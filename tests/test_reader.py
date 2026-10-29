import pytest

from profiles.hylogger.reader import NO_ANSWER, Sample, read, read_scalar_csv

HEADER = "StartDepth,EndDepth,Final_Mask,Grp1_sjCLST,Grp1_sTSAS\n"


def write_csv(tmp_path, body, name="hole-scalars.csv"):
    path = tmp_path / name
    path.write_text(HEADER + body, encoding="latin-1")
    return path


def test_reads_the_four_fields(tmp_path):
    path = write_csv(tmp_path, "10.5,10.5,1,CHLORITE,WHITE-MICA\n")
    assert read_scalar_csv(path) == [
        Sample(depth_m=10.5, valid=True, swir_group="WHITE-MICA", tir_group="CHLORITE")
    ]


def test_mask_other_than_one_is_invalid(tmp_path):
    path = write_csv(tmp_path, "1.0,1.0,0,CHLORITE,CHLORITE\n2.0,2.0,,SILICA,KAOLIN\n")
    assert [s.valid for s in read_scalar_csv(path)] == [False, False]


@pytest.mark.parametrize("spelling", sorted(NO_ANSWER - {""}))
def test_every_no_answer_spelling_becomes_none(tmp_path, spelling):
    path = write_csv(tmp_path, f"1.0,1.0,1,{spelling},{spelling.lower()}\n")
    sample, = read_scalar_csv(path)
    assert sample.swir_group is None
    assert sample.tir_group is None
    # The row itself survives: "logged here, no answer" is a fact about the hole.
    assert sample.valid is True


def test_a_group_name_is_not_truncated_or_case_folded(tmp_path):
    path = write_csv(tmp_path, "1.0,1.0,1, K-FELDSPAR ,OTHER-ALOH\n")
    sample, = read_scalar_csv(path)
    assert sample.tir_group == "K-FELDSPAR"      # surrounding spaces only
    assert sample.swir_group == "OTHER-ALOH"


def test_unreadable_depth_drops_the_row(tmp_path):
    path = write_csv(tmp_path, "1.0,1.0,1,CHLORITE,CHLORITE\n"
                               ",,1,CHLORITE,CHLORITE\n"
                               "-3.4028234663852886e+38,,1,CHLORITE,CHLORITE\n")
    assert [s.depth_m for s in read_scalar_csv(path)] == [1.0]


def test_repeated_depths_are_preserved(tmp_path):
    # A real continuous hole repeats a depth thousands of times. A reader that
    # silently de-duplicated would hide that from the profile.
    path = write_csv(tmp_path, "1.0,1.0,1,CHLORITE,CHLORITE\n"
                               "1.0,1.0,1,SILICA,CHLORITE\n")
    assert [s.depth_m for s in read_scalar_csv(path)] == [1.0, 1.0]


def test_read_dispatches_on_a_file(tmp_path):
    path = write_csv(tmp_path, "1.0,1.0,1,CHLORITE,CHLORITE\n")
    assert len(read(path)) == 1
