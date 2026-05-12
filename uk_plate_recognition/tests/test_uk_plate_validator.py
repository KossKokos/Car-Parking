import pytest

from anpr.validation.uk_plate import (
    clean_plate_text,
    is_valid_uk_plate,
    validate_uk_plate,
)


def test_clean_plate_text():
    assert clean_plate_text("aa04 qzh") == "AA04QZH"
    assert clean_plate_text("AA04-QZH") == "AA04QZH"


def test_valid_uk_plate():
    assert is_valid_uk_plate("AA04QZH") is True
    assert validate_uk_plate("aa04 qzh") == "AA04QZH"


@pytest.mark.parametrize(
    "label",
    [
        "A004QZH",   # wrong first two positions
        "AA0QZHH",   # position 4 should be digit
        "AA04QZ",    # too short
        "AA04QZH9",  # too long
        "1234567",   # wrong format
    ],
)
def test_invalid_uk_plate(label):
    assert is_valid_uk_plate(label) is False

    with pytest.raises(ValueError):
        validate_uk_plate(label)