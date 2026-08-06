"""MeCab から NJD までの公開フロントエンド契約を検証する。"""

# pyright: reportPrivateUsage=false

import copy
import subprocess
import sys
import textwrap
import unicodedata
from typing import Any, cast

import pytest
from phoneme_mapping_helpers import PHONEME_MAPPING_CORPUS, extract_label_phonemes

import pyopenjtalk


G2P_SNAPSHOT_CASES = [
    {
        "text": "こんにちは",
        "phonemes": ["k", "o", "N", "n", "i", "ch", "i", "w", "a"],
        "kana": "コンニチワ",
        "phonemes_use_vanilla": ["k", "o", "N", "n", "i", "ch", "i", "w", "a"],
    },
    {
        "text": "東京は日本の首都です",
        "phonemes": [
            "t",
            "o",
            "o",
            "ky",
            "o",
            "o",
            "w",
            "a",
            "n",
            "i",
            "h",
            "o",
            "N",
            "n",
            "o",
            "sh",
            "u",
            "t",
            "o",
            "d",
            "e",
            "s",
            "U",
        ],
        "kana": "トーキョーワニホンノシュトデス",
        "phonemes_use_vanilla": [
            "t",
            "o",
            "o",
            "ky",
            "o",
            "o",
            "w",
            "a",
            "n",
            "i",
            "h",
            "o",
            "N",
            "n",
            "o",
            "sh",
            "u",
            "t",
            "o",
            "d",
            "e",
            "s",
            "U",
        ],
    },
    {
        "text": "東京　大阪",
        "phonemes": ["t", "o", "o", "ky", "o", "o", "o", "o", "s", "a", "k", "a"],
        "kana": "トーキョーオーサカ",
        "phonemes_use_vanilla": [
            "t",
            "o",
            "o",
            "ky",
            "o",
            "o",
            "o",
            "o",
            "s",
            "a",
            "k",
            "a",
        ],
    },
    {
        "text": "（テスト・ケース）",
        "phonemes": ["t", "e", "s", "U", "t", "o", "pau", "k", "e", "e", "s", "u"],
        "kana": "（テスト・ケース）",
        "phonemes_use_vanilla": [
            "t",
            "e",
            "s",
            "U",
            "t",
            "o",
            "pau",
            "k",
            "e",
            "e",
            "s",
            "u",
        ],
    },
    {
        "text": "今日は2112年9月3日です",
        "phonemes": [
            "ky",
            "o",
            "o",
            "w",
            "a",
            "n",
            "i",
            "s",
            "e",
            "N",
            "hy",
            "a",
            "k",
            "u",
            "j",
            "u",
            "u",
            "n",
            "i",
            "n",
            "e",
            "N",
            "k",
            "u",
            "g",
            "a",
            "ts",
            "u",
            "m",
            "i",
            "cl",
            "k",
            "a",
            "d",
            "e",
            "s",
            "U",
        ],
        "kana": "キョーワニセンヒャクジューニネンクガツミッカデス",
        "phonemes_use_vanilla": [
            "ky",
            "o",
            "o",
            "w",
            "a",
            "n",
            "i",
            "s",
            "e",
            "N",
            "hy",
            "a",
            "k",
            "u",
            "j",
            "u",
            "u",
            "n",
            "i",
            "n",
            "e",
            "N",
            "k",
            "u",
            "g",
            "a",
            "ts",
            "u",
            "m",
            "i",
            "cl",
            "k",
            "a",
            "d",
            "e",
            "s",
            "U",
        ],
    },
    {
        "text": "電話番号は090-1234-5678です",
        "phonemes": [
            "d",
            "e",
            "N",
            "w",
            "a",
            "b",
            "a",
            "N",
            "g",
            "o",
            "o",
            "w",
            "a",
            "z",
            "e",
            "r",
            "o",
            "ky",
            "u",
            "u",
            "z",
            "e",
            "r",
            "o",
            "pau",
            "i",
            "ch",
            "i",
            "n",
            "i",
            "i",
            "s",
            "a",
            "N",
            "y",
            "o",
            "N",
            "pau",
            "g",
            "o",
            "o",
            "r",
            "o",
            "k",
            "u",
            "n",
            "a",
            "n",
            "a",
            "h",
            "a",
            "ch",
            "i",
            "d",
            "e",
            "s",
            "U",
        ],
        "kana": "デンワバンゴーワゼロキューゼロ−イチニーサンヨン−ゴーロクナナハチデス",
        "phonemes_use_vanilla": [
            "d",
            "e",
            "N",
            "w",
            "a",
            "b",
            "a",
            "N",
            "g",
            "o",
            "o",
            "w",
            "a",
            "z",
            "e",
            "r",
            "o",
            "ky",
            "u",
            "u",
            "z",
            "e",
            "r",
            "o",
            "pau",
            "i",
            "ch",
            "i",
            "n",
            "i",
            "i",
            "s",
            "a",
            "N",
            "y",
            "o",
            "N",
            "pau",
            "g",
            "o",
            "o",
            "r",
            "o",
            "k",
            "u",
            "n",
            "a",
            "n",
            "a",
            "h",
            "a",
            "ch",
            "i",
            "d",
            "e",
            "s",
            "U",
        ],
    },
    {
        "text": "つまみ出されようとした",
        "phonemes": [
            "ts",
            "u",
            "m",
            "a",
            "m",
            "i",
            "d",
            "a",
            "s",
            "a",
            "r",
            "e",
            "y",
            "o",
            "o",
            "t",
            "o",
            "sh",
            "I",
            "t",
            "a",
        ],
        "kana": "ツマミダサレヨートシタ",
        "phonemes_use_vanilla": [
            "ts",
            "u",
            "m",
            "a",
            "m",
            "i",
            "d",
            "a",
            "s",
            "a",
            "r",
            "e",
            "y",
            "o",
            "o",
            "t",
            "o",
            "sh",
            "I",
            "t",
            "a",
        ],
    },
    {
        "text": "学生々活",
        "phonemes": ["g", "a", "k", "U", "s", "e", "e", "s", "e", "e", "k", "a", "ts", "u"],
        "kana": "ガクセーセーカツ",
        "phonemes_use_vanilla": ["g", "a", "k", "U", "s", "e", "e", "pau", "k", "a", "ts", "u"],
    },
    {
        "text": "叙々々々苑",
        "phonemes": ["j", "o", "j", "o", "j", "o", "j", "o", "e", "N"],
        "kana": "ジョジョジョジョエン",
        "phonemes_use_vanilla": ["j", "o", "pau", "e", "N"],
    },
    {
        "text": "風がこんな風に吹く",
        "phonemes": [
            "k",
            "a",
            "z",
            "e",
            "g",
            "a",
            "k",
            "o",
            "N",
            "n",
            "a",
            "f",
            "u",
            "u",
            "n",
            "i",
            "f",
            "u",
            "k",
            "u",
        ],
        "kana": "カゼガコンナフウニフク",
        "phonemes_use_vanilla": [
            "k",
            "a",
            "z",
            "e",
            "g",
            "a",
            "k",
            "o",
            "N",
            "n",
            "a",
            "k",
            "a",
            "z",
            "e",
            "n",
            "i",
            "f",
            "u",
            "k",
            "u",
        ],
    },
    {
        "text": "何ですか",
        "phonemes": ["n", "a", "N", "d", "e", "s", "U", "k", "a"],
        "kana": "ナンデスカ",
        "phonemes_use_vanilla": ["n", "a", "n", "i", "d", "e", "s", "U", "k", "a"],
    },
    {
        "text": "今日は何をする",
        "phonemes": ["ky", "o", "o", "w", "a", "n", "a", "n", "i", "o", "s", "u", "r", "u"],
        "kana": "キョーワナニヲスル",
        "phonemes_use_vanilla": [
            "ky",
            "o",
            "o",
            "w",
            "a",
            "n",
            "a",
            "n",
            "i",
            "o",
            "s",
            "u",
            "r",
            "u",
        ],
    },
    {
        "text": "𰻞𰻞麺を食べた",
        "phonemes": ["m", "e", "N", "o", "t", "a", "b", "e", "t", "a"],
        "kana": "𰻞𰻞メンヲタベタ",
        "phonemes_use_vanilla": ["m", "e", "N", "o", "t", "a", "b", "e", "t", "a"],
    },
    {
        "text": "あーーーーーーーーあ",
        "phonemes": ["a", "a", "a", "a", "a", "a", "a", "a", "a", "a"],
        "kana": "アーーーーーーーーア",
        "phonemes_use_vanilla": ["a", "a", "a", "a", "a", "a", "a", "a", "a", "a"],
    },
    {
        "text": "しなじう",
        "phonemes": ["sh", "i", "n", "a", "j", "i", "u"],
        "kana": "シナジウ",
        "phonemes_use_vanilla": ["sh", "i", "n", "a", "j", "i", "i"],
    },
    {
        "text": "いみじう",
        "phonemes": ["i", "m", "i", "j", "i", "u"],
        "kana": "イミジウ",
        "phonemes_use_vanilla": ["i", "m", "i", "j", "i", "i"],
    },
]


@pytest.mark.parametrize("text", PHONEME_MAPPING_CORPUS)
def test_fullcontext_corpus_matches_split_frontend(text: str):
    """
    多様な語彙コーパスに対し、extract_fullcontext() と分割実行の全文脈ラベルが一致することを確認。

    要素ごとの zip 比較ではラベル数の差を見逃すため、リスト全体を比較する。
    """

    njd_features = pyopenjtalk.run_frontend(text)
    assert pyopenjtalk.extract_fullcontext(text) == pyopenjtalk.make_label(njd_features)


def test_extract_fullcontext_recovers_after_zero_phoneme_input():
    """音素を生成しない空入力の処理後も次の全文脈ラベル生成が正常に動作することを確認。"""

    assert pyopenjtalk.extract_fullcontext("") == []
    assert pyopenjtalk.extract_fullcontext("復帰") == pyopenjtalk.make_label(
        pyopenjtalk.run_frontend("復帰")
    )


@pytest.mark.parametrize("case", G2P_SNAPSHOT_CASES)
def test_g2p_snapshot_cases(case: dict[str, object]):
    text = cast(str, case["text"])
    expected_phonemes = cast(list[str], case["phonemes"])
    expected_kana = cast(str, case["kana"])
    expected_use_vanilla = cast(list[str], case["phonemes_use_vanilla"])

    assert pyopenjtalk.g2p(text, join=False) == expected_phonemes
    assert pyopenjtalk.g2p(text, kana=True) == expected_kana
    assert pyopenjtalk.g2p(text, join=False, use_vanilla=True) == expected_use_vanilla


@pytest.mark.parametrize("case", G2P_SNAPSHOT_CASES)
def test_g2p_snapshot_consistent_with_make_label(case: dict[str, object]):
    text = cast(str, case["text"])

    for is_use_vanilla in (False, True):
        njd_features = pyopenjtalk.run_frontend(text, use_vanilla=is_use_vanilla)
        labels = pyopenjtalk.make_label(njd_features)
        expected_phonemes = extract_label_phonemes(labels, keep_pause=True)

        assert pyopenjtalk.g2p(text, join=False, use_vanilla=is_use_vanilla) == expected_phonemes


def test_unicode_normalization_nfc():
    text = "か\u3099くせい"
    normalized_text = unicodedata.normalize("NFC", text)

    assert pyopenjtalk.g2p(text, kana=True, normalize_mode="NFC") == pyopenjtalk.g2p(
        normalized_text,
        kana=True,
    )


def test_unicode_normalization_nfkc():
    text = "ｶﾞｸｾｲ"
    normalized_text = unicodedata.normalize("NFKC", text)

    assert pyopenjtalk.g2p(text, kana=True, normalize_mode="NFKC") == pyopenjtalk.g2p(
        normalized_text,
        kana=True,
    )


def test_unicode_normalization_invalid_mode():
    invalid_mode: Any = "invalid"
    with pytest.raises(ValueError, match="normalize_mode must be one of"):
        pyopenjtalk.g2p("学生", normalize_mode=invalid_mode)


def test_unicode_normalization_combining_chars():
    """多様な結合文字が NFC 正規化で正しく処理されることを確認。"""

    combining_texts = [
        "\u304b\u3099",  # か + 結合濁点 → が
        "\u306f\u309a",  # は + 結合半濁点 → ぱ
        "\u30b3\u3099",  # コ + 結合濁点 → ゴ
        "\u0065\u0301",  # e + 結合アクセント → é
    ]
    for text in combining_texts:
        nfc_text = unicodedata.normalize("NFC", text)
        result_with_mode = pyopenjtalk.g2p(text, kana=True, normalize_mode="NFC")
        result_direct = pyopenjtalk.g2p(nfc_text, kana=True)
        assert result_with_mode == result_direct, (
            f"NFC normalization mismatch for {text!r}: "
            f"mode=NFC -> {result_with_mode}, direct -> {result_direct}"
        )


def test_run_mecab_long_input_should_not_segfault():
    command = [
        sys.executable,
        "-c",
        textwrap.dedent(
            """
            import sys
            import pyopenjtalk

            text = "😀" * 3000
            try:
                pyopenjtalk.run_mecab(text)
            except RuntimeError:
                sys.exit(0)
            except Exception:
                sys.exit(2)
            else:
                sys.exit(0)
            """
        ),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30.0)

    assert completed.returncode == 0


@pytest.mark.parametrize("method_name", ["run_mecab", "run_mecab_detailed"])
def test_run_mecab_utf8_buffer_boundary_and_recovery(method_name: str):
    """
    text2mecab の16,384バイト境界で UTF-8 を分断せず、容量超過後も次の解析へ復帰できることを確認。

    3バイト文字5461個は終端を含めて収まり、5462個では正規化後の出力が容量を超える。
    """

    method = getattr(pyopenjtalk, method_name)

    result = method("あ" * 5461)
    if method_name == "run_mecab_detailed":
        _, morphs = result
        assert len(morphs) > 0
    else:
        assert len(result) > 0
    with pytest.raises(RuntimeError, match="too long"):
        method("あ" * 5462)
    result = method("復帰")
    if method_name == "run_mecab_detailed":
        _, morphs = result
        assert len(morphs) > 0
    else:
        assert len(result) > 0


def test_run_frontend_empty_string():
    """空文字列を run_frontend に渡した場合、クラッシュせずリストを返すこと。"""
    features = pyopenjtalk.run_frontend("")
    assert isinstance(features, list)


def test_run_frontend_very_long_text():
    """非常に長いテキストを run_frontend に渡した場合、RuntimeError を送出するか返すこと（セグフォしないこと）。"""
    with pytest.raises(RuntimeError, match="too long"):
        pyopenjtalk.run_frontend("あ" * 10000)

    features = pyopenjtalk.run_frontend("こんにちは")
    assert len(features) > 0


def test_run_frontend_special_characters_only():
    """特殊文字のみを run_frontend に渡した場合、クラッシュしないこと。"""
    features = pyopenjtalk.run_frontend("!@#$%^&*()")
    assert isinstance(features, list)


def test_run_frontend_null_bytes_should_not_segfault():
    """null バイトを run_frontend に渡した場合、セグフォしないこと（例外を送出するか返す可能性あり）。"""
    command = [
        sys.executable,
        "-c",
        textwrap.dedent(
            """
            import pyopenjtalk

            try:
                features = pyopenjtalk.run_frontend("\\x00\\x01\\x02")
                assert isinstance(features, list)
            except Exception:
                pass
            """
        ),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30.0)

    assert completed.returncode == 0


def test_run_frontend_mixed_japanese_ascii():
    """日本語と ASCII が混在したテキストを run_frontend に渡した場合、正常に動作すること。"""
    features = pyopenjtalk.run_frontend("Hello世界123")
    assert isinstance(features, list)


def test_run_frontend_single_character():
    """1 文字を run_frontend に渡した場合、正常に動作すること。"""
    features = pyopenjtalk.run_frontend("あ")
    assert isinstance(features, list)
    assert len(features) > 0


def test_make_label_too_long_feature_should_not_crash():
    njd_features = pyopenjtalk.run_frontend("こんにちは")
    njd_features[0]["pron"] = "ア" * 400

    labels = pyopenjtalk.make_label(njd_features)

    assert isinstance(labels, list)


def test_make_label_empty_string_fields_should_not_crash():
    njd_features = pyopenjtalk.run_frontend("こんにちは")
    njd_features[0]["pron"] = ""
    njd_features[0]["pos"] = ""
    njd_features[0]["ctype"] = ""
    njd_features[0]["cform"] = ""

    labels = pyopenjtalk.make_label(njd_features)

    assert isinstance(labels, list)


def test_make_label_validation_error_should_not_break_next_call():
    njd_features = pyopenjtalk.run_frontend("こんにちは")
    invalid_njd_features = copy.deepcopy(njd_features)
    invalid_njd_features[0]["pron"] = 123  # type: ignore[assignment]

    with pytest.raises(TypeError, match="must be str"):
        pyopenjtalk.make_label(invalid_njd_features)

    labels = pyopenjtalk.make_label(njd_features)
    assert len(labels) > 0


def test_make_label_missing_field_should_not_break_next_call():
    njd_features = pyopenjtalk.run_frontend("こんにちは")
    invalid_njd_features = copy.deepcopy(njd_features)
    del invalid_njd_features[0]["pron"]  # type: ignore[assignment]

    with pytest.raises(KeyError):
        pyopenjtalk.make_label(invalid_njd_features)

    labels = pyopenjtalk.make_label(njd_features)
    assert len(labels) > 0


def test_make_label_null_character_should_not_break_next_call():
    njd_features = pyopenjtalk.run_frontend("こんにちは")
    invalid_njd_features = copy.deepcopy(njd_features)
    invalid_njd_features[0]["pron"] = "ア\x00イ"

    with pytest.raises(ValueError, match="contains null character"):
        pyopenjtalk.make_label(invalid_njd_features)

    labels = pyopenjtalk.make_label(njd_features)
    assert len(labels) > 0


def test_make_label_invalid_numeric_field_should_raise_type_error():
    njd_features = pyopenjtalk.run_frontend("こんにちは")
    invalid_njd_features = copy.deepcopy(njd_features)
    invalid_njd_features[0]["acc"] = "1"  # type: ignore[assignment]

    with pytest.raises(TypeError, match="must be int: acc"):
        pyopenjtalk.make_label(invalid_njd_features)


def test_g2p_large_digit_sequence_should_keep_place_reading():
    pron = pyopenjtalk.g2p("10000")

    assert pron == "i ch i m a N"


def test_g2p_large_digit_sequence_with_oku_should_keep_place_reading():
    pron = pyopenjtalk.g2p("100000000")

    assert pron == "i ch i o k u"


def test_run_mecab_runtime_error_should_not_break_next_call():
    with pytest.raises(RuntimeError, match="too long"):
        pyopenjtalk.run_mecab("😀" * 4096)

    morphs = pyopenjtalk.run_mecab("こんにちは")
    assert len(morphs) > 0


def test_run_njd_from_mecab_invalid_input_should_not_break_next_call():
    valid_mecab_features = pyopenjtalk.run_mecab("こんにちは")
    invalid_mecab_features = copy.deepcopy(valid_mecab_features)
    invalid_mecab_features[0] = 123  # type: ignore[assignment]

    with pytest.raises(TypeError, match="must be str"):
        pyopenjtalk.run_njd_from_mecab(invalid_mecab_features)

    njd_features = pyopenjtalk.run_njd_from_mecab(valid_mecab_features)
    assert len(njd_features) > 0


def test_run_njd_from_mecab_rule_exception_releases_njd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python 側規則が例外を送出しても、次の NJD 処理へノードを残さない。"""

    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    mecab_features = jtalk.run_mecab("こんにちは")
    openjtalk_module = pyopenjtalk.openjtalk
    original_rule = openjtalk_module.apply_original_rule_before_chaining

    def raise_from_rule(_features: list[Any]) -> list[Any]:
        """
        NJD の Python 側規則適用中に例外を送出する。

        Args:
            _features (list[Any]): NJD から変換した特徴列

        Raises:
            RuntimeError: 規則適用失敗を再現するため常に送出
        """

        raise RuntimeError("rule failure")

    monkeypatch.setattr(openjtalk_module, "apply_original_rule_before_chaining", raise_from_rule)
    with pytest.raises(RuntimeError, match="rule failure"):
        jtalk.run_njd_from_mecab(mecab_features)

    monkeypatch.setattr(openjtalk_module, "apply_original_rule_before_chaining", original_rule)
    assert jtalk.run_njd_from_mecab(mecab_features) == pyopenjtalk.run_frontend("こんにちは")


def test_misaligned_chained_orig_preserves_the_original_surface(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """熟字訓を誤分割した複数アクセント句でも元の表層を失わない。"""

    # `百舌鳥` は表層3文字と読み2モーラが対応しないが、単一アクセント句なら通常どおり処理できる
    regular_features = pyopenjtalk.run_njd_from_mecab(
        ["百舌鳥,名詞,固有名詞,一般,*,*,*,百舌鳥,モズ,モズ,0/2,C3"]
    )
    assert regular_features[0]["string"] == "百舌鳥"
    assert regular_features[0]["pron"] == "モズ"

    # 最終句だけ原形と表層が異なる活用語は、NAIST 辞書に含まれる正常な複数アクセント句
    conjugated_features = pyopenjtalk.run_njd_from_mecab(
        [
            "かんきわま,動詞,自立,*,*,五段・ラ行,体言接続特殊２,"
            "かん:きわまる,カン:キワマ,カン:キワマ,1/2:3/3,*"
        ]
    )
    assert [feature["string"] for feature in conjugated_features] == ["かん", "きわま"]

    # 正常な単一アクセント句と複数アクセント句では、安全策の警告を出さない
    assert capfd.readouterr().err == ""

    # 外部辞書が熟字訓の読みを複数句の orig に誤用した場合だけ、表層との対応が崩れる
    malformed_features = pyopenjtalk.run_njd_from_mecab(
        ["百舌鳥確認,名詞,一般,*,*,*,*,モズ:確認,モズ:カクニン,モズ:カクニン,0/2:0/4,C1"]
    )
    captured_stderr = capfd.readouterr().err

    # 言語的に不正な句分割の出力は固定せず、元の表層を読みへ置換しないことだけを検証する
    assert malformed_features[0]["string"] == "百舌鳥確認"
    assert captured_stderr.count("Chained orig does not match the surface prefix") == 1
    assert 'surface: "百舌鳥確認"' in captured_stderr
    assert 'orig: "モズ:確認"' in captured_stderr
    assert len(pyopenjtalk.make_label(malformed_features)) > 0


def test_run_mecab_detailed_known_word():
    """辞書に存在する単語が is_unknown=False で返されることを確認。"""

    _, morphs = pyopenjtalk.run_mecab_detailed("こんにちは")
    assert len(morphs) >= 1
    # 全てのフィールドが存在することを確認
    for morph in morphs:
        assert "surface" in morph
        assert "features" in morph
        assert "pos_id" in morph
        assert "left_id" in morph
        assert "right_id" in morph
        assert "word_cost" in morph
        assert "link_cost" in morph
        assert "node_cost" in morph
        assert "char_span" in morph
        assert "is_unknown" in morph
        assert "is_ignored" in morph
        assert "dictionary_index" in morph
    # 「こんにちは」は辞書に存在するので、少なくとも 1 つは既知語がある
    assert any(morph["is_unknown"] is False for morph in morphs)


def test_run_mecab_detailed_unknown_word():
    """辞書に存在しない造語が is_unknown=True で返されることを確認。"""

    # カタカナは辞書内の既知語に分割されてしまうため、
    # MeCab が確実に未知語と判定する ASCII 文字列を使用する
    _, morphs = pyopenjtalk.run_mecab_detailed("xtjq")
    assert any(morph["is_unknown"] is True for morph in morphs)


def test_run_mecab_detailed_splits_repeated_known_symbols():
    """未知語へ連結された既知記号を1文字ずつの形態素へ復元することを確認。"""

    _, morphs = pyopenjtalk.run_mecab_detailed("うおお！！！！！！！！！！！！！！！！")
    exclamation_morphs = [morph for morph in morphs if morph["surface"] == "！"]

    assert len(exclamation_morphs) == 16
    assert all(morph["is_unknown"] is False for morph in exclamation_morphs)
    # `symbols.csv` で選択される「！」の最小単語コストを、未知語チャンクからの復元後も保持
    assert all(morph["word_cost"] == 1525 for morph in exclamation_morphs)


def test_run_mecab_detailed_restored_symbol_metadata():
    """分解復元した既知記号が通常ノードと同じキー集合・値型を持つことを確認。"""

    _, normal_morphs = pyopenjtalk.run_mecab_detailed("こんにちは")
    normal_morph = normal_morphs[0]
    _, restored_morphs = pyopenjtalk.run_mecab_detailed("！？！？")
    restored_morph = restored_morphs[0]

    assert restored_morph.keys() == normal_morph.keys()
    assert isinstance(restored_morph["surface"], str)
    assert isinstance(restored_morph["features"], list)
    assert all(isinstance(feature, str) for feature in restored_morph["features"])
    assert isinstance(restored_morph["pos_id"], int)
    assert isinstance(restored_morph["left_id"], int)
    assert isinstance(restored_morph["right_id"], int)
    assert isinstance(restored_morph["word_cost"], int)
    assert isinstance(restored_morph["is_unknown"], bool)
    assert isinstance(restored_morph["is_ignored"], bool)


def test_run_mecab_detailed_includes_ignored():
    """通常の run_mecab ではフィルタされる記号,空白トークンも含まれることを確認。"""

    # 通常の run_mecab は記号,空白をフィルタする
    normal_morphs = pyopenjtalk.run_mecab("東京　大阪")
    # detailed は全トークンを返す
    _, detailed_morphs = pyopenjtalk.run_mecab_detailed("東京　大阪")
    # detailed の方がトークン数が多い（もしくは同じ）
    assert len(detailed_morphs) >= len(normal_morphs)


def test_run_mecab_detailed_feature_format():
    """feature 文字列が既存 run_mecab() と同じ "surface,品詞,..." フォーマットであることを確認。"""

    _, morphs = pyopenjtalk.run_mecab_detailed("こんにちは")
    for morph in morphs:
        # features の先頭要素は surface と一致する
        assert morph["features"][0] == morph["surface"]


def test_run_mecab_detailed_cost_types():
    """pos_id, left_id, right_id, word_cost が正しい型 (int) で返されることを確認。"""

    _, morphs = pyopenjtalk.run_mecab_detailed("東京は日本の首都です")
    for morph in morphs:
        assert isinstance(morph["pos_id"], int)
        assert isinstance(morph["left_id"], int)
        assert isinstance(morph["right_id"], int)
        assert isinstance(morph["word_cost"], int)


def test_run_mecab_detailed_empty_string():
    """空文字列入力でクラッシュしないことを確認。"""

    features, morphs = pyopenjtalk.run_mecab_detailed("")
    assert isinstance(features, list)
    assert isinstance(morphs, list)


def test_run_mecab_detailed_consistency_with_run_mecab():
    """run_mecab_detailed の features が run_mecab の結果と一致することを確認。"""

    text = "こんにちは世界"
    normal_features = pyopenjtalk.run_mecab(text)
    features, _morphs = pyopenjtalk.run_mecab_detailed(text)

    assert features == normal_features


def test_run_mecab_nbest_features_preserves_multiple_readings():
    """汎用 n-best API が同じ表層の複数読みを費用順で返すことを確認。"""

    paths = pyopenjtalk.run_mecab_nbest_features("最中を食べる", max_paths=3)

    # 呼び出し側が候補ごとの NJD 入力と比較費用をそのまま利用できる形を固定する
    assert len(paths) == 3
    assert [path["path_cost"] for path in paths] == sorted(path["path_cost"] for path in paths)
    first_features = [path["features"][0] for path in paths]
    assert any(",サイチュウ,サイチュー," in feature for feature in first_features)
    assert any(",モナカ,モナカ," in feature for feature in first_features)


def test_run_frontend_detailed_basic():
    """run_frontend_detailed がタプルを返し、NJDFeature が run_frontend と同一であることを確認。"""

    text = "こんにちは"
    njd_features, morphs = pyopenjtalk.run_frontend_detailed(text)
    njd_features_normal = pyopenjtalk.run_frontend(text)

    assert isinstance(njd_features, list)
    assert isinstance(morphs, list)
    assert njd_features == njd_features_normal
    assert len(morphs) >= 1


@pytest.mark.parametrize("text", ["÷÷÷÷", "！？" * 8])
def test_run_frontend_detailed_matches_normal_for_restored_symbols(text: str):
    """復元対象の連続記号でも NJDFeature が run_frontend と同一であることを確認。"""

    njd_features, morphs = pyopenjtalk.run_frontend_detailed(text)

    assert njd_features == pyopenjtalk.run_frontend(text)
    assert [morph["surface"] for morph in morphs] == list(text)


def test_run_frontend_detailed_morphs_fields():
    """run_frontend_detailed の morphs に全フィールドが含まれることを確認。"""

    _, morphs = pyopenjtalk.run_frontend_detailed("東京は日本の首都です")
    for morph in morphs:
        assert "surface" in morph
        assert "features" in morph
        assert "pos_id" in morph
        assert "left_id" in morph
        assert "right_id" in morph
        assert "word_cost" in morph
        assert "is_unknown" in morph
        assert "is_ignored" in morph


def test_run_frontend_detailed_empty_string():
    """空文字列で run_frontend_detailed がクラッシュしないことを確認。"""

    njd_features, morphs = pyopenjtalk.run_frontend_detailed("")
    assert isinstance(njd_features, list)
    assert isinstance(morphs, list)


def test_run_frontend_detailed_morphs_consistency():
    """run_frontend_detailed の morphs が run_mecab_detailed の結果と同一であることを確認。"""

    text = "東京は日本の首都です"
    _, morphs_from_frontend = pyopenjtalk.run_frontend_detailed(text)
    _, morphs_from_detailed = pyopenjtalk.run_mecab_detailed(text)
    assert morphs_from_frontend == morphs_from_detailed


RUN_FRONTEND_SPLIT_EQUIVALENCE_CASES = [
    "こんにちは",
    "明日は雨が降るでしょう",
    "焼きそばパン買ってこいや",
    "国境の長いトンネルを抜けると雪国であった。",
    "外国人参政権",
    "あのイーハトーヴォのすきとおった風、夏でも底に冷たさをもつ青いそら、",
    "うつくしい森で飾られたモリーオ市、郊外のぎらぎらひかる草の波。",
    "今日は2112年9月3日です",
    "電話番号は090-1234-5678です",
    "",
    "あ",
    "！？",
    "123456",
    "ABCabc",
    "日本語English123!",
    "The quick brown fox jumps over the lazy dog.",
]


@pytest.mark.parametrize("text", RUN_FRONTEND_SPLIT_EQUIVALENCE_CASES)
def test_run_frontend_split_equivalence(text: str):
    """run_frontend が分割実行 (run_mecab → run_njd_from_mecab → apply_postprocessing) と一致することを確認。"""

    original_result = pyopenjtalk.run_frontend(text)

    mecab_features = pyopenjtalk.run_mecab(text)
    njd_features = pyopenjtalk.run_njd_from_mecab(mecab_features)
    split_result = pyopenjtalk.apply_postprocessing(text, njd_features)

    assert original_result == split_result
