import pytest

from anpr.data.label_parser import parse_label_from_filename


def test_parse_label_from_clean_filename():
    assert parse_label_from_filename("AA04QZH.png") == "AA04QZH"


def test_parse_label_from_filename_with_space():
    assert parse_label_from_filename("AA04 QZH.jpg") == "AA04QZH"


def test_parse_label_from_lowercase_filename():
    assert parse_label_from_filename("aa04qzh.png") == "AA04QZH"


def test_parse_embedded_plate_when_allowed():
    assert (
        parse_label_from_filename(
            "AA04QZH_001.png",
            allow_embedded_plate=True,
        )
        == "AA04QZH"
    )


def test_reject_embedded_plate_by_default():
    with pytest.raises(ValueError):
        parse_label_from_filename("AA04QZH_001.png")