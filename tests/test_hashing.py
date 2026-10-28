from engine.hashing import canonical_json, sha256_file, sha256_json, short


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    assert sha256_json({"b": 1, "a": 2}) == sha256_json({"a": 2, "b": 1})


def test_file_hash_matches_known_value(tmp_path):
    p = tmp_path / "x.csv"
    p.write_bytes(b"depth_m,valid\n0.0,True\n")
    # sha256 of those exact bytes
    assert sha256_file(p) == __import__("hashlib").sha256(
        b"depth_m,valid\n0.0,True\n").hexdigest()


def test_short_is_a_prefix():
    d = sha256_json({"a": 1})
    assert d.startswith(short(d))
    assert len(short(d)) == 12
