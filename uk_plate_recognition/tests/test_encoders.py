import pytest

from anpr.data.encoders import (
    CLASS_COUNTS,
    decode_indices,
    encode_label,
)


def test_class_counts_match_fixed_uk_plate_format():
    assert CLASS_COUNTS == (26, 26, 10, 10, 26, 26, 26)


def test_encode_label():
    assert encode_label("AA04QZH") == [0, 0, 0, 4, 16, 25, 7]


def test_decode_indices():
    assert decode_indices([0, 0, 0, 4, 16, 25, 7]) == "AA04QZH"


def test_encode_decode_round_trip():
    label = "AB12CDE"
    assert decode_indices(encode_label(label)) == label


@pytest.mark.parametrize(
    "label",
    [
        "A004QZH",
        "AA0QZHH",
        "AA04QZ",
        "AA04QZH9",
        "1234567",
    ],
)
def test_encode_rejects_invalid_plate(label):
    with pytest.raises(ValueError):
        encode_label(label)


def test_decode_rejects_wrong_number_of_indices():
    with pytest.raises(ValueError):
        decode_indices([0, 1, 2])


def test_decode_rejects_invalid_index_for_position():
    with pytest.raises(ValueError):
        decode_indices([0, 0, 99, 4, 16, 25, 7])