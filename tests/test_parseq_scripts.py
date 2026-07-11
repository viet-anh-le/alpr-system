from pathlib import Path


def test_parseq_label_from_path_preserves_vietnamese_plate_chars() -> None:
    from ocr.parseq_dataset import label_from_path

    assert label_from_path("59-MĐ2-01265&2#0178.jpg") == "59-MĐ2-01265"
    assert label_from_path("60LD-02423&1#0457.jpg") == "60LD-02423"


def test_parseq_default_charset_contains_required_plate_tokens() -> None:
    from ocr.parseq_dataset import DEFAULT_PARSEQ_VN_CHARSET, unknown_chars

    assert "Đ" in DEFAULT_PARSEQ_VN_CHARSET
    assert "D" in DEFAULT_PARSEQ_VN_CHARSET
    assert "-" in DEFAULT_PARSEQ_VN_CHARSET
    assert "." in DEFAULT_PARSEQ_VN_CHARSET
    assert "[" in DEFAULT_PARSEQ_VN_CHARSET
    assert "]" in DEFAULT_PARSEQ_VN_CHARSET
    assert unknown_chars("59-MĐ2-01265", DEFAULT_PARSEQ_VN_CHARSET) == set()
    assert unknown_chars("60LD-02423", DEFAULT_PARSEQ_VN_CHARSET) == set()
    assert unknown_chars("59-T2[SEP]004.14", DEFAULT_PARSEQ_VN_CHARSET) == set()


def test_parseq_dataset_rejects_unknown_chars_without_normalizing_them(tmp_path: Path) -> None:
    from ocr.parseq_dataset import DEFAULT_PARSEQ_VN_CHARSET, FilenamePlateDataset

    image_path = tmp_path / "59-MĐ2-01265@&2#0178.jpg"
    image_path.write_bytes(b"not a real image")

    try:
        FilenamePlateDataset(tmp_path, charset=DEFAULT_PARSEQ_VN_CHARSET)
    except ValueError as exc:
        assert "@" in str(exc)
    else:
        raise AssertionError("Expected invalid label to be rejected")


def test_parseq_positional_char_accuracy_counts_dash_and_distinguishes_d() -> None:
    from ocr.train_parseq import positional_char_accuracy

    correct, total = positional_char_accuracy(
        ["59-MD2-01265", "60LD-02423"],
        ["59-MĐ2-01265", "60LD-02423"],
    )

    assert total == len("59-MĐ2-01265") + len("60LD-02423")
    assert correct == total - 1
