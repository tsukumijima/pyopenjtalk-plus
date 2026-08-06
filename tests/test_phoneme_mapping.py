"""形態素と音素の対応、および Haqumei (https://github.com/o24s/haqumei) から移植したテストケースを検証する。"""

from collections.abc import Mapping, Sequence

import pytest

import pyopenjtalk


PHONEME_MAPPING_CORPUS = [
    "こんにちは",
    "おはようございます",
    "東京は日本の首都です",
    "東京都知事が記者会見を行った。",
    "大阪",
    "外国人参政権",
    "学生生活",
    "学生々活は楽しい",
    "部分々々",
    "東京、大阪",
    "東京　大阪",
    "（テスト・ケース）",
    "今日は2112年9月3日です",
    "電話番号は090-1234-5678です",
    "明日は雨が降るでしょう",
    "ご遠慮ください",
    "お入りください",
    "食べよう",
    "見よう",
    "読もう",
    "書こう",
    "遊ぼう",
    "起きよう",
    "考えよう",
    "見せよう",
    "行こう",
    "入ろう",
    "来よう",
    "しよう",
    "食べている",
    "読んでいる",
    "書いている",
    "走っている",
    "見ている",
    "起きている",
    "つまみ出されようとした",
]


LONG_VOWEL_MERGE_CASES = [
    ("食べよう", "食べよう", "食べる"),
    ("見よう", "見よう", "見る"),
    ("読もう", "読もう", "読む"),
    ("書こう", "書こう", "書く"),
    ("遊ぼう", "遊ぼう", "遊ぶ"),
    ("起きよう", "起きよう", "起きる"),
    ("考えよう", "考えよう", "考える"),
    ("見せよう", "見せよう", "見せる"),
    ("行こう", "行こう", "行く"),
    ("入ろう", "入ろう", "入る"),
    ("来よう", "来よう", "来る"),
    ("つまみ出されようとした", "れよう", "れる"),
    # リテラルの長音記号 (ー) が吸収された場合は orig にも保持されること
    ("あーーーーーーーーあ", "あーーーーーーーー", "あーーーーーーーー"),
]


DOUNOJITEN_TEXT = "叙々々々々々々苑々々様々々要所々々々々々槇野々々々"


DOUNOJITEN_EXPECTED = [
    ("叙", ["j", "o"]),
    ("々々々々々々", ["j", "o", "j", "o", "j", "o", "j", "o", "j", "o", "j", "o"]),
    ("苑", ["e", "N"]),
    ("々々", ["e", "N", "e", "N"]),
    ("様々", ["s", "a", "m", "a", "z", "a", "m", "a"]),
    ("々", ["z", "a", "m", "a"]),
    ("要所々々", ["y", "o", "o", "sh", "o", "y", "o", "o", "sh", "o"]),
    ("々々", ["y", "o", "o", "sh", "o", "y", "o", "o", "sh", "o"]),
    ("々", ["y", "o", "o", "sh", "o"]),
    ("槇野々", ["m", "a", "k", "i", "n", "o", "n", "o"]),
    ("々々", ["n", "o", "n", "o"]),
]


NIGHTMARE_MAPPING_TEXT = (
    "つまみ出されようとしたが、「「八十五歳」」にもなる　長老　に助けられた。"
    "わーいです。そこで、𰻞𰻞麺とお冷を飲み食いしたです。"
    "ーっ、　𰻞ー𰻞。あ、はい。あーーーーーーーーあ"
    "叙々々々々々々苑々々様々々要所々々々々々槇野々々々"
)


NIGHTMARE_MAPPING_EXPECTED = [
    ("つまみ出さ", ["ts", "u", "m", "a", "m", "i", "d", "a", "s", "a"]),
    ("れよう", ["r", "e", "y", "o", "o"]),
    ("と", ["t", "o"]),
    ("し", ["sh", "I"]),
    ("た", ["t", "a"]),
    ("が", ["g", "a"]),
    ("、", ["pau"]),
    ("「", []),
    ("「", []),
    ("八", ["h", "a", "ch", "i"]),
    ("十", ["j", "u", "u"]),
    ("五", ["g", "o"]),
    ("歳", ["s", "a", "i"]),
    ("」", []),
    ("」", []),
    ("に", ["n", "i"]),
    ("も", ["m", "o"]),
    ("なる", ["n", "a", "r", "u"]),
    ("　", ["sp"]),
    ("長老", ["ch", "o", "o", "r", "o", "o"]),
    ("　", ["sp"]),
    ("に", ["n", "i"]),
    ("助け", ["t", "a", "s", "U", "k", "e"]),
    ("られ", ["r", "a", "r", "e"]),
    ("た", ["t", "a"]),
    ("。", ["pau"]),
    ("わーい", ["w", "a", "a", "i"]),
    ("です", ["d", "e", "s", "U"]),
    ("。", ["pau"]),
    ("そこで", ["s", "o", "k", "o", "d", "e"]),
    ("、", ["pau"]),
    ("𰻞𰻞", ["unk"]),
    ("麺", ["m", "e", "N"]),
    ("と", ["t", "o"]),
    ("お冷", ["o", "h", "i", "y", "a"]),
    ("を", ["o"]),
    ("飲み", ["n", "o", "m", "i"]),
    ("食い", ["g", "u", "i"]),
    ("し", ["sh", "I"]),
    ("た", ["t", "a"]),
    ("です", ["d", "e", "s", "U"]),
    ("。", ["pau"]),
    ("ー", ["unk"]),
    ("っ", ["cl"]),
    ("、", ["pau"]),
    ("　", ["sp"]),
    ("𰻞", ["unk"]),
    ("ー", ["unk"]),
    ("𰻞", ["unk"]),
    ("。", ["pau"]),
    ("あ", ["a"]),
    ("、", ["pau"]),
    ("はい", ["h", "a", "i"]),
    ("。", ["pau"]),
    ("あーーーーーーーー", ["a", "a", "a", "a", "a", "a", "a", "a", "a"]),
    ("あ", ["a"]),
    ("叙", ["j", "o"]),
    ("々々々々々々", ["j", "o", "j", "o", "j", "o", "j", "o", "j", "o", "j", "o"]),
    ("苑", ["e", "N"]),
    ("々々", ["e", "N", "e", "N"]),
    ("様々", ["s", "a", "m", "a", "z", "a", "m", "a"]),
    ("々", ["z", "a", "m", "a"]),
    ("要所々々", ["y", "o", "o", "sh", "o", "y", "o", "o", "sh", "o"]),
    ("々々", ["y", "o", "o", "sh", "o", "y", "o", "o", "sh", "o"]),
    ("々", ["y", "o", "o", "sh", "o"]),
    ("槇野々", ["m", "a", "k", "i", "n", "o", "n", "o"]),
    ("々々", ["n", "o", "n", "o"]),
]


FLAG_INVARIANT_CORPUS = [
    "吾輩は猫である。名前　はまだ無　い。",
    "𰻞𰻞麺を、　食べたい。",
    "学生々活7xyz七大阪",
    "ーっ、　𰻞ー𰻞。",
    NIGHTMARE_MAPPING_TEXT,
]


def _flatten_mapping_phonemes(
    mapping: Sequence[Mapping[str, object]],
    keep_pause: bool = False,
) -> list[str]:
    phonemes: list[str] = []
    for entry in mapping:
        entry_phonemes = entry["phonemes"]
        assert isinstance(entry_phonemes, list)
        if keep_pause is False and entry_phonemes in (["pau"], ["sp"]):
            continue
        if entry_phonemes == ["unk"]:
            continue
        phonemes.extend(entry_phonemes)
    return phonemes


def _extract_label_phonemes(labels: list[str], keep_pause: bool = False) -> list[str]:
    phonemes = [label.split("-")[1].split("+")[0] for label in labels[1:-1]]
    if keep_pause is False:
        phonemes = [phoneme for phoneme in phonemes if phoneme != "pau"]
    return phonemes


def _mapping_surface_phonemes(
    mapping: Sequence[Mapping[str, object]],
) -> list[tuple[str, list[str]]]:
    result: list[tuple[str, list[str]]] = []
    for entry in mapping:
        surface = entry["surface"]
        phonemes = entry["phonemes"]
        assert isinstance(surface, str)
        assert isinstance(phonemes, list)
        result.append((surface, phonemes))
    return result


def test_g2p_mapping_splits_alternating_pause_symbols():
    """交互に連続する感嘆符と疑問符が単一の未知語へ潰れないことを確認。"""

    text = "マジ！？！？！？！？！？！？！？！？"
    mapping = pyopenjtalk.g2p_mapping(text)

    assert "".join(entry["surface"] for entry in mapping) == text
    assert [entry["surface"] for entry in mapping[1:]] == list("！？" * 8)
    assert all(entry["is_unknown"] is False for entry in mapping[1:])


def test_g2p_mapping_splits_mixed_symbol_run():
    """既知記号と長音が混在する連続記号でも表層と既知判定を保持することを確認。"""

    text = "！？！？！？！？ー！￥／？ー！？！？"
    mapping = pyopenjtalk.g2p_mapping(text)

    assert "".join(entry["surface"] for entry in mapping) == text
    assert [entry["surface"] for entry in mapping] == list(text)
    assert [entry["surface"] for entry in mapping if entry["is_unknown"] is True] == ["ー", "ー"]


def test_make_phoneme_mapping_basic():
    """基本的な形態素-音素マッピングが返されることを確認。"""

    njd_features = pyopenjtalk.run_frontend("こんにちは")
    mapping = pyopenjtalk.make_phoneme_mapping(njd_features)
    assert len(mapping) >= 1
    for entry in mapping:
        assert "surface" in entry
        assert "phonemes" in entry
        assert "is_unknown" in entry
        assert "is_ignored" in entry
        assert len(entry["phonemes"]) > 0
        assert all(isinstance(phoneme, str) for phoneme in entry["phonemes"])
        # morphs なしの場合は is_unknown=False
        assert entry["is_unknown"] is False


def test_make_phoneme_mapping_with_punctuation():
    """句読点がポーズ音素 ['pau'] として扱われることを確認。"""

    njd_features = pyopenjtalk.run_frontend("東京、大阪")
    mapping = pyopenjtalk.make_phoneme_mapping(njd_features)
    pause_entries = [entry for entry in mapping if entry["phonemes"] == ["pau"]]
    assert len(pause_entries) >= 1


def test_make_phoneme_mapping_boundary_punctuation_end():
    """文末の句読点にも NJD の pron に基づいて pau が割り当てられることを確認。"""

    njd_features = pyopenjtalk.run_frontend("あ。")
    mapping = pyopenjtalk.make_phoneme_mapping(njd_features)

    assert mapping[0]["surface"] == "あ"
    assert mapping[0]["phonemes"] == ["a"]
    assert mapping[1]["surface"] == "。"
    assert mapping[1]["phonemes"] == ["pau"]
    assert mapping[1]["is_ignored"] is False


def test_make_phoneme_mapping_boundary_punctuation_start():
    """文頭の句読点にも NJD の pron に基づいて pau が割り当てられることを確認。"""

    njd_features = pyopenjtalk.run_frontend("。あ")
    mapping = pyopenjtalk.make_phoneme_mapping(njd_features)

    assert mapping[0]["surface"] == "。"
    assert mapping[0]["phonemes"] == ["pau"]
    assert mapping[0]["is_ignored"] is False
    assert mapping[1]["surface"] == "あ"
    assert mapping[1]["phonemes"] == ["a"]


def test_make_phoneme_mapping_pause_like_symbols():
    """pause-like 記号は、実際に pause がある箇所だけ ['pau'] を持つことを確認。"""

    njd_features = pyopenjtalk.run_frontend("（テスト・ケース）")
    mapping = pyopenjtalk.make_phoneme_mapping(njd_features)

    assert mapping[0]["surface"] == "（"
    assert mapping[0]["phonemes"] == []
    assert mapping[0]["is_ignored"] is True
    assert mapping[1]["surface"] == "テスト"
    assert mapping[1]["phonemes"] == ["t", "e", "s", "U", "t", "o"]
    assert mapping[2]["surface"] == "・"
    assert mapping[2]["phonemes"] == ["pau"]
    assert mapping[3]["surface"] == "ケース"
    assert mapping[3]["phonemes"] == ["k", "e", "e", "s", "u"]
    assert mapping[4]["surface"] == "）"
    assert mapping[4]["phonemes"] == []
    assert mapping[4]["is_ignored"] is True


def test_make_phoneme_mapping_prefers_explicit_pause_symbol_over_quote():
    """quote と読点が連続する場合、実際の pause は読点側へ関連付けられることを確認。"""

    njd_features = pyopenjtalk.run_frontend("「東京」、大阪")
    mapping = pyopenjtalk.make_phoneme_mapping(njd_features)

    assert mapping[0]["surface"] == "「"
    assert mapping[0]["phonemes"] == []
    assert mapping[2]["surface"] == "」"
    assert mapping[2]["phonemes"] == []
    assert mapping[3]["surface"] == "、"
    assert mapping[3]["phonemes"] == ["pau"]


def test_make_phoneme_mapping_phoneme_consistency_with_pause_retained():
    """句読点を含む入力では、通常音素列が make_label と一致しつつ pau が保持されることを確認。"""

    text = "東京、大阪"
    njd_features = pyopenjtalk.run_frontend(text)
    mapping = pyopenjtalk.make_phoneme_mapping(njd_features)

    mapping_phonemes = []
    for entry in mapping:
        if entry["phonemes"] != ["pau"]:
            mapping_phonemes.extend(entry["phonemes"])

    labels = pyopenjtalk.make_label(njd_features)
    label_phonemes = [
        phoneme
        for phoneme in map(lambda s: s.split("-")[1].split("+")[0], labels[1:-1])
        if phoneme != "pau"
    ]

    assert any(entry["phonemes"] == ["pau"] for entry in mapping)
    assert mapping_phonemes == label_phonemes


@pytest.mark.parametrize("text", PHONEME_MAPPING_CORPUS)
def test_make_phoneme_mapping_corpus_phoneme_consistency(text: str):
    """
    多様な語彙コーパスに対し、morphs なしの make_phoneme_mapping() でも音素列が安定していることを確認。

    Cython 側の Word-Mora-Phoneme マッピングが崩れると Python 側の補正以前に音素列が壊れるため、
    ベースマッピング単体でも広い語彙で検証する。
    """

    njd_features = pyopenjtalk.run_frontend(text)
    mapping = pyopenjtalk.make_phoneme_mapping(njd_features)
    labels = pyopenjtalk.make_label(njd_features)

    assert _flatten_mapping_phonemes(mapping) == _extract_label_phonemes(labels)


def test_make_phoneme_mapping_digit():
    """数字入力 (NJD でノード数が変わるケース) でクラッシュしないことを確認。"""

    njd_features = pyopenjtalk.run_frontend("123")
    mapping = pyopenjtalk.make_phoneme_mapping(njd_features)
    assert len(mapping) >= 1


def test_make_phoneme_mapping_empty_features():
    """空の NJDFeature リストで空リストが返されることを確認。"""

    mapping = pyopenjtalk.make_phoneme_mapping([])
    assert mapping == []


def test_make_phoneme_mapping_surface_correspondence():
    """
    make_phoneme_mapping() の surface フィールドが NJDFeature の string と 1:1 対応していることを確認
    NOTE: 長音吸収マージが発生するテキストでは len(mapping) < len(njd_features) となるため、
    このテストでは長音吸収が発生しない入力のみを使用している。
    """

    text = "今日も良い天気ですね"
    njd_features = pyopenjtalk.run_frontend(text)
    mapping = pyopenjtalk.make_phoneme_mapping(njd_features)

    assert len(mapping) == len(njd_features)
    for entry, feat in zip(mapping, njd_features):
        assert entry["surface"] == feat["string"]


def test_make_phoneme_mapping_long_vowel_merge_cython():
    """
    Cython レベルの make_phoneme_mapping() で長音吸収マージが正しく動作することを確認。
    "つまみ出されようとした" では NJD の長音処理により 'う' (pron='ー') が前方の Word に吸収される。
    OpenJTalk.make_phoneme_mapping() (Cython 直接呼び出し) で:
      - 吸収されたトークンが前方の Word に結合されること
      - 戻り値の長さが入力 features と異なる場合があること
      - 全エントリの phonemes が空でないこと
    を検証する。
    """

    jtalk = pyopenjtalk.openjtalk.OpenJTalk(pyopenjtalk.OPEN_JTALK_DICT_DIR)
    njd_features = jtalk.run_frontend("つまみ出されようとした")
    mapping = jtalk.make_phoneme_mapping(njd_features)

    # 長音吸収により mapping の長さが features より短くなる
    assert len(mapping) < len(njd_features), (
        f"Expected mapping length < features length due to long vowel merge, "
        f"got mapping: {len(mapping)}, features: {len(njd_features)}"
    )

    # 全エントリの phonemes が空でないこと
    for entry in mapping:
        assert len(entry["phonemes"]) > 0, f"Empty phonemes for word: {entry['surface']}"

    # 'れよう' がマージ結果として存在すること
    words = [entry["surface"] for entry in mapping]
    assert "れよう" in words, f"Expected 'れよう' in words, got: {words}"
    assert "う" not in words, f"'う' should be merged into 'れよう', got: {words}"


def test_make_phoneme_mapping_with_morphs_basic():
    """基本的な morphs 付きマッピングが返されることを確認。"""

    text = "こんにちは"
    njd_features, morphs = pyopenjtalk.run_frontend_detailed(text)
    detailed = pyopenjtalk.make_phoneme_mapping(njd_features, morphs=morphs)

    assert len(detailed) >= 1
    for entry in detailed:
        assert "surface" in entry
        assert "phonemes" in entry
        assert "is_unknown" in entry
        assert "is_ignored" in entry


def test_make_phoneme_mapping_with_morphs_unknown():
    """未知語が is_unknown=True で返されることを確認。"""

    # カタカナは辞書内の既知語に分割されてしまうため、
    # MeCab が確実に未知語と判定する ASCII 文字列を使用する
    text = "xtjqは最高"
    njd_features, morphs = pyopenjtalk.run_frontend_detailed(text)
    detailed = pyopenjtalk.make_phoneme_mapping(njd_features, morphs=morphs)

    assert any(entry["is_unknown"] is True for entry in detailed)


def test_make_phoneme_mapping_with_morphs_unknown_after_digit_normalization():
    """
    数字正規化が先行するケースで is_unknown が正しく伝播することを確認。
    "7xyz" は MeCab 上では "７"(既知) + "ｘｙｚ"(未知) だが、
    NJD 側で "７" → "七" に正規化されるため surface 不一致が発生する。
    バランスベースのアライメントにより、後続の未知語にも is_unknown が正しく伝播する。
    """

    text = "7xyz"
    njd_features, morphs = pyopenjtalk.run_frontend_detailed(text)
    detailed = pyopenjtalk.make_phoneme_mapping(njd_features, morphs=morphs)

    assert len(detailed) >= 2
    # 数字正規化されたエントリが存在すること
    assert any(entry["surface"] == "七" for entry in detailed)
    # 未知語トークンの is_unknown が正しく伝播していること
    xyz_entry = next(entry for entry in detailed if entry["surface"] == "ｘｙｚ")
    assert xyz_entry["is_unknown"] is True


def test_make_phoneme_mapping_with_morphs_known():
    """既知語が is_unknown=False で返されることを確認。"""

    text = "こんにちは"
    njd_features, morphs = pyopenjtalk.run_frontend_detailed(text)
    detailed = pyopenjtalk.make_phoneme_mapping(njd_features, morphs=morphs)

    assert any(entry["is_unknown"] is False for entry in detailed)


def test_make_phoneme_mapping_with_morphs_phonemes_match():
    """morphs 付きの phonemes が morphs なしの結果と (sp/unk を除き) 実質一致することを確認。"""

    text = "東京は日本の首都です"
    njd_features, morphs = pyopenjtalk.run_frontend_detailed(text)

    mapping_without = pyopenjtalk.make_phoneme_mapping(njd_features)
    mapping_with = pyopenjtalk.make_phoneme_mapping(njd_features, morphs=morphs)

    # morphs 付きの場合、sp/unk エントリが追加されることがあるため、
    # sp/unk を除いた通常エントリ同士を比較する
    normal_without = [e for e in mapping_without if e["phonemes"] not in (["sp"], ["unk"])]
    normal_with = [e for e in mapping_with if e["phonemes"] not in (["sp"], ["unk"])]

    assert len(normal_without) == len(normal_with)
    for entry_without, entry_with in zip(normal_without, normal_with):
        assert entry_without["surface"] == entry_with["surface"]
        assert entry_without["phonemes"] == entry_with["phonemes"]


def test_make_phoneme_mapping_with_morphs_digit():
    """数字入力 (NJD でノード数が変わるケース) でクラッシュしないことを確認。"""

    text = "123"
    njd_features, morphs = pyopenjtalk.run_frontend_detailed(text)
    detailed = pyopenjtalk.make_phoneme_mapping(njd_features, morphs=morphs)

    assert len(detailed) >= 1


@pytest.mark.parametrize(
    ("text", "expected_surfaces"),
    [
        ("２0ｉｔ", ["二", "十", "ｉｔ"]),
        ("２0　ｉｔ　日々", ["二", "十", "　", "ｉ", "ｔ", "　", "日々"]),
        ("1　0", ["十", "　"]),
        ("1　00", ["百", "　"]),
    ],
)
def test_make_phoneme_mapping_digit_alignment_is_local(
    text: str,
    expected_surfaces: list[str],
):
    """
    数字展開と後続ノードの粒度変化が混在しても、数字ブロック内だけで morph 消費数を決めることを確認。

    Haqumei で英単語結合と数字展開が相殺された回帰入力を使い、将来ノード結合を追加しても
    数字の対応判定が後続全体の要素数へ依存しない契約を固定する。
    """

    mapping = pyopenjtalk.g2p_mapping(text)

    assert [entry["surface"] for entry in mapping] == expected_surfaces
    if mapping[-1]["is_ignored"] is True:
        assert mapping[-1]["phonemes"] == ["sp"]
    else:
        assert mapping[-1]["features"][0] == expected_surfaces[-1]


@pytest.mark.parametrize("text", PHONEME_MAPPING_CORPUS)
def test_make_phoneme_mapping_with_morphs_corpus_phoneme_consistency(text: str):
    """
    多様な語彙コーパスに対し、make_phoneme_mapping() の音素列が make_label() と整合することを確認。

    `morphs` 付きのアライメント経路は数字正規化・踊り字展開・長音吸収などで壊れやすいため、
    さまざまな品詞・活用・句読点を含む入力でまとめて検証する。
    """

    njd_features, morphs = pyopenjtalk.run_frontend_detailed(text)
    mapping = pyopenjtalk.make_phoneme_mapping(njd_features, morphs=morphs)
    labels = pyopenjtalk.make_label(njd_features)

    assert _flatten_mapping_phonemes(mapping) == _extract_label_phonemes(labels)


@pytest.mark.parametrize("text", PHONEME_MAPPING_CORPUS)
def test_make_phoneme_mapping_with_morphs_corpus_features_consistency(text: str):
    """
    `features` を持つエントリは、常にその entry 自身の surface と一致することを確認。

    1:1 に対応しない merged node や正規化ノードに別 morph の features を紐づけると、
    downstream で誤った語彙情報を参照してしまうため、空リストにする必要がある。
    """

    njd_features, morphs = pyopenjtalk.run_frontend_detailed(text)
    mapping = pyopenjtalk.make_phoneme_mapping(njd_features, morphs=morphs)

    for entry in mapping:
        if len(entry["features"]) > 0:
            assert entry["features"][0] == entry["surface"]


@pytest.mark.parametrize(("text", "merged_surface", "expected_orig"), LONG_VOWEL_MERGE_CASES)
def test_make_phoneme_mapping_with_morphs_long_vowel_metadata(
    text: str,
    merged_surface: str,
    expected_orig: str,
):
    """
    長音吸収で merged された node の メタデータが破綻しないことを確認。

    代表的な意向形・助動詞連結を広く検証し、
    `features` が空リストになることと、`orig` が辞書の原形のまま保持されることを確認する。
    """

    njd_features, morphs = pyopenjtalk.run_frontend_detailed(text)
    mapping = pyopenjtalk.make_phoneme_mapping(njd_features, morphs=morphs)

    merged_entry = next(entry for entry in mapping if entry["surface"] == merged_surface)
    assert merged_entry["features"] == []
    assert merged_entry["orig"] == expected_orig


@pytest.mark.parametrize("text", ["÷÷÷÷", "！？" * 8])
def test_g2p_mapping_matches_normal_frontend_for_restored_symbols(text: str):
    """復元対象の連続記号でも通常の NJDFeature に基づく発音を返すことを確認。"""

    expected_mapping = pyopenjtalk.make_phoneme_mapping(pyopenjtalk.run_frontend(text))
    actual_mapping = pyopenjtalk.g2p_mapping(text)

    assert [entry["surface"] for entry in actual_mapping] == list(text)
    assert [phoneme for entry in actual_mapping for phoneme in entry["phonemes"]] == [
        phoneme for entry in expected_mapping for phoneme in entry["phonemes"]
    ]


def test_g2p_mapping_basic():
    """g2p_mapping の基本動作と全フィールドの存在・型を確認。"""

    mapping = pyopenjtalk.g2p_mapping("こんにちは")
    assert len(mapping) >= 1
    for entry in mapping:
        # SurfacePhonemeMapping の全フィールドが存在し正しい型であること
        assert isinstance(entry["surface"], str)
        assert isinstance(entry["char_span"], tuple)
        assert len(entry["char_span"]) == 2
        assert isinstance(entry["phonemes"], list)
        assert isinstance(entry["features"], list)
        for col in entry["features"]:
            assert isinstance(col, str)
        assert isinstance(entry["pos"], str)
        assert isinstance(entry["pos_group1"], str)
        assert isinstance(entry["pos_group2"], str)
        assert isinstance(entry["pos_group3"], str)
        assert isinstance(entry["ctype"], str)
        assert isinstance(entry["cform"], str)
        assert isinstance(entry["orig"], str)
        assert isinstance(entry["read"], str)
        assert isinstance(entry["pron"], str)
        assert isinstance(entry["accent_nucleus"], int)
        assert isinstance(entry["mora_count"], int)
        assert isinstance(entry["chain_rule"], str)
        assert isinstance(entry["chain_flag"], int)
        assert isinstance(entry["is_unknown"], bool)
        assert isinstance(entry["is_ignored"], bool)
        assert len(entry["phonemes"]) > 0
        # 既知語は features が8列以上
        if len(entry["features"]) > 0:
            assert len(entry["features"]) >= 8
            assert entry["features"][0] == entry["surface"]


def test_g2p_mapping_features_populated():
    """
    g2p_mapping で features が MeCab feature 文字列の分割リストとして返されることを確認。

    features の列数は MeCab の解析結果に依存する:
      - 既知語: 12 列 (surface, 品詞, ..., chain_rule)
      - 未知語: 8 列 (surface, 品詞, ..., 原形。読み/発音/acc/chain_rule がない)
      - アライメント不一致 (sp/数字展開等): 0 列 (空リスト)。
    """

    mapping = pyopenjtalk.g2p_mapping("東京は日本の首都です")
    for entry in mapping:
        # features は常に list[str] であること
        assert isinstance(entry["features"], list)
        for col in entry["features"]:
            assert isinstance(col, str)
        if len(entry["features"]) > 0:
            # 空でない場合、先頭要素が surface と一致すること
            assert entry["features"][0] == entry["surface"]
            # 既知語は 12 列以上、未知語は 8 列以上
            assert len(entry["features"]) >= 8, (
                f"Expected at least 8 feature columns, got {len(entry['features'])} for {entry['surface']}"
            )


def test_g2p_mapping_features_unknown_word():
    """未知語の features が 8 列 (読み/発音/acc/chain_rule なし) で返されることを確認。"""

    mapping = pyopenjtalk.g2p_mapping("xtjqは最高")
    xtjq = next(e for e in mapping if e["surface"] == "ｘｔｊｑ")
    assert xtjq["is_unknown"] is True
    assert isinstance(xtjq["features"], list)
    assert len(xtjq["features"]) == 8
    assert xtjq["features"][0] == "ｘｔｊｑ"

    # 既知語は 12 列
    saikou = next(e for e in mapping if e["surface"] == "最高")
    assert saikou["is_unknown"] is False
    assert len(saikou["features"]) == 12


@pytest.mark.parametrize(
    "text",
    [
        "彼は虎の威を借る狐のような人間だ。",
        "若さ故の無謀な行動は許されない。",
        "私は安定した会社に正社員として雇われたい。",
        "もう行っちゃう。",
    ],
)
def test_g2p_mapping_accepts_detailed_auxiliary_pos_without_warning(
    text: str,
    capfd: pytest.CaptureFixture[str],
):
    """システム辞書の詳細な助動詞・接尾辞品詞を JPCommon へ変換できることを確認。"""

    mapping = pyopenjtalk.g2p_mapping(text)
    captured = capfd.readouterr()

    assert len(mapping) > 0
    assert "convert_pos()" not in captured.err


def test_g2p_mapping_accepts_mixed_godan_sahen_conjugation_without_warning(
    capfd: pytest.CaptureFixture[str],
):
    """辞書に実在する「致す」の混合活用型を五段活用として変換できることを確認。"""

    mapping = pyopenjtalk.g2p_mapping("依頼を致す。")
    captured = capfd.readouterr()

    assert (
        next(entry for entry in mapping if entry["surface"] == "致す")["ctype"]
        == "五段・サ変・スル"
    )
    assert "convert_ctype()" not in captured.err


def test_make_label_accepts_dictionary_region_noun_without_warning(
    capfd: pytest.CaptureFixture[str],
):
    """システム辞書の「美術館」に付く地域分類を普通名詞として変換できることを確認。"""

    njd_features = pyopenjtalk.run_frontend("美術館")
    # 通常の one-best では一般名詞になるため、同梱辞書の別候補が持つ地域分類を再現
    njd_features[0]["pos_group1"] = "地域"
    labels = pyopenjtalk.make_label(njd_features)
    captured = capfd.readouterr()

    assert len(labels) > 0
    assert "convert_pos()" not in captured.err


def test_g2p_mapping_ignores_leading_quote_pause_without_warning(
    capfd: pytest.CaptureFixture[str],
):
    """文頭引用符の短ポーズを本文の音素へ混入させず処理できることを確認。"""

    mapping = pyopenjtalk.g2p_mapping("「今帝」という呼称だ。")
    captured = capfd.readouterr()

    assert mapping[0]["surface"] == "「"
    assert mapping[0]["phonemes"] == []
    assert len(mapping[1]["phonemes"]) > 0
    assert captured.err == ""


def test_g2p_mapping_features_space():
    """全角スペース (sp) の features が空リストであることを確認。"""

    mapping = pyopenjtalk.g2p_mapping("東京　大阪")
    sp_entries = [e for e in mapping if e["is_ignored"] is True]
    assert len(sp_entries) >= 1
    for sp in sp_entries:
        assert sp["features"] == []


def test_g2p_mapping_unknown_word():
    """
    g2p_mapping で未知語が is_unknown=True を持つことを確認。
    unk 音素への置換は、未知語かつ音素が空の場合のみ発生する。
    OpenJTalk が実際に音素を生成できた未知語は、is_unknown=True のまま
    生成された音素がそのまま保持される。
    """

    mapping = pyopenjtalk.g2p_mapping("xtjq")
    unknown_entries = [e for e in mapping if e["is_unknown"] is True]
    assert len(unknown_entries) >= 1
    # 未知語は必ず何らかの音素を持つ (空にはならない)
    for entry in unknown_entries:
        assert len(entry["phonemes"]) > 0


def test_g2p_mapping_unknown_pause_symbol():
    """NJD が読点扱いした未知記号を区切り記号と誤認せず unk に戻すことを確認。"""

    mapping = pyopenjtalk.g2p_mapping("東京ヶ大阪")

    assert mapping[0]["surface"] == "東京"
    assert mapping[1]["surface"] == "ヶ"
    assert mapping[1]["phonemes"] == ["unk"]
    assert mapping[1]["is_unknown"] is True
    assert mapping[2]["surface"] == "大阪"


def test_g2p_mapping_space_produces_sp():
    """g2p_mapping で全角空白が sp 音素を持つことを確認。"""

    mapping = pyopenjtalk.g2p_mapping("東京　大阪")
    sp_entries = [e for e in mapping if e["phonemes"] == ["sp"]]
    assert len(sp_entries) >= 1
    for entry in sp_entries:
        assert entry["is_ignored"] is True


def test_g2p_mapping_long_vowel_merge():
    """
    長音吸収マージにより語境界と音素対応が正しいことを確認。

    "つまみ出されようとした" では NJD の長音処理により 'う' (pron='ー') が
    前方の 'れよ' に吸収され、JPCommon Word としては 'れよう' が一つの Word になる。
    吸収されたトークンの word テキストは前方の Word に結合され、
    全エントリの phonemes が空でないことを保証する。
    """

    mapping = pyopenjtalk.g2p_mapping("つまみ出されようとした")
    for entry in mapping:
        # sp, unk, pau, 通常音素のいずれかが入っているはず
        assert len(entry["phonemes"]) > 0

    # 語境界の正確さを検証: 'う' が 'れよ' にマージされて 'れよう' になること
    words = [entry["surface"] for entry in mapping]
    assert "れよう" in words, f"Expected 'れよう' in words, got: {words}"
    # 'う' が独立エントリとして残っていないこと
    assert "う" not in words, f"'う' should be merged into 'れよう', got: {words}"

    # 'れよう' の音素が正しいこと (長音を含む)
    reyou_entry = next(entry for entry in mapping if entry["surface"] == "れよう")
    assert reyou_entry["phonemes"] == ["r", "e", "y", "o", "o"]

    # 'と' が正しい音素を持つこと (長音吸収前はオフバイワンで崩れていた)
    to_entry = next(entry for entry in mapping if entry["surface"] == "と")
    assert to_entry["phonemes"] == ["t", "o"]


def test_g2p_mapping_merged_internal_spaces():
    """スペースを挟んで分断された長音マークがマージされることを確認。"""

    mapping = pyopenjtalk.g2p_mapping("なる\u3000長老ー\u3000ー\u3000に")
    assert len(mapping) == 6

    surfaces = [entry["surface"] for entry in mapping]
    assert "なる" in surfaces
    assert "長老ーー" in surfaces
    assert "に" in surfaces
    # スペースが 3 つ含まれること (マージされたスペースも含む)
    sp_entries = [entry for entry in mapping if entry["is_ignored"] is True]
    assert len(sp_entries) == 3

    merged = next(entry for entry in mapping if entry["surface"] == "長老ーー")
    assert len(merged["phonemes"]) == 8


def test_g2p_mapping_triple_merge_with_spaces():
    """3 つの長音マークがスペースを挟んで 1 語にマージされることを確認。"""

    mapping = pyopenjtalk.g2p_mapping("あー\u3000ー\u3000ー")
    assert len(mapping) == 3

    surfaces = [entry["surface"] for entry in mapping]
    assert "あーーー" in surfaces
    sp_count = sum(1 for entry in mapping if entry["surface"] == "\u3000")
    assert sp_count == 2


def test_g2p_mapping_unknown_merged_with_space():
    """未知語とスペースと長音マークの組み合わせが崩れないことを確認。"""

    mapping = pyopenjtalk.g2p_mapping("𰻞\u3000ー")
    assert len(mapping) == 3

    surfaces = [entry["surface"] for entry in mapping]
    assert "𰻞" in surfaces
    assert "\u3000" in surfaces
    assert "ー" in surfaces

    rare_kanji = next(entry for entry in mapping if entry["surface"] == "𰻞")
    assert rare_kanji["phonemes"] == ["unk"]
    assert rare_kanji["is_unknown"] is True

    long_vowel = next(entry for entry in mapping if entry["surface"] == "ー")
    assert long_vowel["phonemes"] == ["unk"]
    assert long_vowel["is_unknown"] is True


def test_g2p_mapping_merged_word_boundary_spaces():
    """前後にスペースがある語の長音マージが正しく処理されることを確認。"""

    mapping = pyopenjtalk.g2p_mapping("\u3000あーー\u3000")
    assert len(mapping) == 3

    assert mapping[0]["surface"] == "\u3000"
    assert mapping[0]["is_ignored"] is True
    assert mapping[1]["surface"] == "あーー"
    assert mapping[2]["surface"] == "\u3000"
    assert mapping[2]["is_ignored"] is True


def test_g2p_mapping_complex_punctuation():
    """入れ子括弧・連続記号の pau 割り当てが崩れないことを確認。"""

    mapping = pyopenjtalk.g2p_mapping("「東京」、大阪」…、…あ")

    # 閉じ括弧は空音素
    kagi_close_entries = [entry for entry in mapping if entry["surface"] == "」"]
    assert len(kagi_close_entries) == 2
    for entry in kagi_close_entries:
        assert entry["phonemes"] == []

    # 最初の読点は pau
    touten_entries = [entry for entry in mapping if entry["surface"] == "、"]
    assert len(touten_entries) >= 1
    assert touten_entries[0]["phonemes"] == ["pau"]

    # 最初の省略記号は pau
    ellipsis_entries = [entry for entry in mapping if entry["surface"] == "…"]
    assert len(ellipsis_entries) == 2
    assert all(entry["phonemes"] == ["pau"] for entry in ellipsis_entries)

    # 連続する区切り記号は位置にかかわらず全て pau
    assert all(entry["phonemes"] == ["pau"] for entry in touten_entries)


def test_g2p_mapping_nested_quotes_have_stable_pause_assignment():
    """同じ閉じ括弧の音素が出現位置によって変わらないことを確認。"""

    mapping = pyopenjtalk.g2p_mapping("つまみ出されようとしたが、「「八十五歳」」にもなる")
    closing_quotes = [entry for entry in mapping if entry["surface"] == "」"]

    assert len(closing_quotes) == 2
    assert all(entry["phonemes"] == [] for entry in closing_quotes)


def test_g2p_mapping_keeps_internal_spaces_after_merged_word():
    """長音結合語の内部にあった空白を結合語の直後へ保持することを確認。"""

    text = "なる　長老ー　ー　に"
    mapping = pyopenjtalk.g2p_mapping(text)

    assert [entry["surface"] for entry in mapping] == [
        "なる",
        "　",
        "長老ーー",
        "　",
        "　",
        "に",
    ]


def test_g2p_mapping_empty_string():
    """空文字列で g2p_mapping が空リストを返すことを確認。"""

    mapping = pyopenjtalk.g2p_mapping("")
    assert mapping == []


def test_g2p_mapping_all_ignored():
    """
    全角スペースのみの入力で全エントリが is_ignored=True, phonemes=['sp'] となることを確認。

    全 morphs が ignored のケースに対応するテスト。
    """

    mapping = pyopenjtalk.g2p_mapping("　　　")
    assert len(mapping) >= 1
    for entry in mapping:
        assert entry["is_ignored"] is True
        assert entry["phonemes"] == ["sp"]


def test_make_phoneme_mapping_with_morphs_trailing_space():
    """テキスト末尾の空白トークンが sp として回収されることを確認。"""

    text = "こんにちは　"
    njd_features, morphs = pyopenjtalk.run_frontend_detailed(text)
    detailed = pyopenjtalk.make_phoneme_mapping(njd_features, morphs=morphs)

    # 末尾エントリが sp であること
    assert detailed[-1]["phonemes"] == ["sp"]
    assert detailed[-1]["is_ignored"] is True


def test_g2p_mapping_odori_resync():
    """
    踊り字展開で morph と NJD feature の粒度がずれるケースで、
    後続トークンのアライメントが正しく re-sync されることを確認。

    踊り字展開 (process_odori_features) は MeCab morphs を再構成するため、
    morphs=['学生', '々', '活', 'は', '楽しい'] に対して NJD=['学生', '生活', 'は', '楽しい']
    のように粒度がずれる。このとき不一致ブランチで morph_idx を適切に進めて re-sync し、
    後続の 'は' や '楽しい' が正しい morph と対応することを検証する。
    """

    mapping = pyopenjtalk.g2p_mapping("学生々活は楽しい")

    words = [entry["surface"] for entry in mapping]
    # 踊り字展開後の NJD feature 構成
    assert "学生" in words
    assert "生活" in words
    assert "は" in words
    assert "楽しい" in words

    # 全エントリの phonemes が空でないこと
    for entry in mapping:
        assert len(entry["phonemes"]) > 0, f"Empty phonemes for word: {entry['surface']}"

    # 生活の音素が正しいこと
    seikatsu = next(entry for entry in mapping if entry["surface"] == "生活")
    assert seikatsu["phonemes"] == ["s", "e", "e", "k", "a", "ts", "u"]

    # 楽しい が正しくマッピングされていること (re-sync が必要)
    tanoshii = next(entry for entry in mapping if entry["surface"] == "楽しい")
    assert len(tanoshii["phonemes"]) > 0


def test_g2p_mapping_odori_with_space():
    """踊り字展開 + 全角スペース併用で sp エントリが正しく出力されることを確認。"""

    mapping = pyopenjtalk.g2p_mapping("学生々活　大阪")

    words = [entry["surface"] for entry in mapping]
    assert "学生" in words
    assert "生活" in words
    assert "大阪" in words

    # 全角スペースが sp として出力されること
    sp_entries = [entry for entry in mapping if entry["phonemes"] == ["sp"]]
    assert len(sp_entries) >= 1
    for sp_entry in sp_entries:
        assert sp_entry["is_ignored"] is True


def test_g2p_mapping_odori_digit_unknown_combined():
    """
    踊り字展開 + 数字正規化 + 未知語の連続不一致で is_unknown が正しく伝播することを確認。

    "学生々活7xyz大阪" では:
      - 踊り字展開: morphs=['学生','々','活'] → NJD=['学生','生活'] (不一致 #1)
      - 数字正規化: morph='７' → NJD='七' (不一致 #2)
      - 未知語: morph='ｘｙｚ' (is_unknown=True)
    re-sync が過剰にスキップすると 'ｘｙｚ' の is_unknown=True が失われる回帰を防ぐ。
    """

    mapping = pyopenjtalk.g2p_mapping("学生々活7xyz大阪")

    # ｘｙｚ の is_unknown が正しく伝播していること
    xyz_entry = next(entry for entry in mapping if entry["surface"] == "ｘｙｚ")
    assert xyz_entry["is_unknown"] is True, (
        f"Expected ｘｙｚ to be is_unknown=True, got {xyz_entry}"
    )

    # 大阪 が正しくマッピングされていること
    osaka_entry = next(entry for entry in mapping if entry["surface"] == "大阪")
    assert osaka_entry["is_unknown"] is False
    assert osaka_entry["phonemes"] == ["o", "o", "s", "a", "k", "a"]


def test_g2p_mapping_odori_digit_unknown_duplicate_word():
    """
    踊り字展開 + 数字正規化 + next_base_word が後方に重複するケースで、
    is_unknown が正しく伝播し、間の morph が過剰にスキップされないことを確認。

    "学生々活7xyz七大阪" では:
      - 踊り字展開: morphs=['学生','々','活'] → NJD=['学生','生活'] (不一致 #1)
      - 数字正規化: morph='７' → NJD='七' (不一致 #2)
      - 未知語: morph='ｘｙｚ' (is_unknown=True)
      - NJD='七' が後方に再登場するため、probe 方式では後方の literal '七' に誤同期する回帰を防ぐ。
    """

    mapping = pyopenjtalk.g2p_mapping("学生々活7xyz七大阪")

    # ｘｙｚ の is_unknown が正しく伝播していること
    xyz_entry = next(entry for entry in mapping if entry["surface"] == "ｘｙｚ")
    assert xyz_entry["is_unknown"] is True, (
        f"Expected ｘｙｚ to be is_unknown=True, got {xyz_entry}"
    )

    # 大阪 が正しくマッピングされていること
    osaka_entry = next(entry for entry in mapping if entry["surface"] == "大阪")
    assert osaka_entry["is_unknown"] is False

    # 全エントリの phonemes が空でないこと
    for entry in mapping:
        assert len(entry["phonemes"]) > 0, f"Empty phonemes for word: {entry['surface']}"


def test_g2p_mapping_odori_digit_unknown_duplicate_word_with_space():
    """
    踊り字展開 + 全角スペース + 数字正規化 + 重複語。
    "学生々活　7xyz七大阪" は全角スペース (is_ignored) が踊り字展開と数字正規化の間に挟まるケース。
    """

    mapping = pyopenjtalk.g2p_mapping("学生々活　7xyz七大阪")

    # ｘｙｚ の is_unknown が正しく伝播していること
    xyz_entry = next(entry for entry in mapping if entry["surface"] == "ｘｙｚ")
    assert xyz_entry["is_unknown"] is True, (
        f"Expected ｘｙｚ to be is_unknown=True, got {xyz_entry}"
    )

    # 全角スペースが sp として出力されること
    sp_entries = [entry for entry in mapping if entry["phonemes"] == ["sp"]]
    assert len(sp_entries) >= 1

    # 大阪 が正しくマッピングされていること
    osaka_entry = next(entry for entry in mapping if entry["surface"] == "大阪")
    assert osaka_entry["is_unknown"] is False


def test_g2p_mapping_accent_phrase_boundary():
    """
    chain_flag でアクセント句境界が正しく識別できることを確認。
    chain_flag が -1 or 0 ならアクセント句の開始、1 なら前の語に連結。
    「東京都知事が」は 1 つのアクセント句を構成する。
    """

    mapping = pyopenjtalk.g2p_mapping("東京都知事が記者会見を行った。")

    tokyo = next(e for e in mapping if e["surface"] == "東京")
    tochiji = next(e for e in mapping if e["surface"] == "都知事")
    ga = next(e for e in mapping if e["surface"] == "が")
    kisha = next(e for e in mapping if e["surface"] == "記者")

    # 東京は先頭なので -1
    assert tokyo["chain_flag"] in [-1, 0]
    # 都知事・が は東京に連結
    assert tochiji["chain_flag"] == 1
    assert ga["chain_flag"] == 1
    # 記者は新しいアクセント句の開始
    assert kisha["chain_flag"] == 0


def test_g2p_mapping_accent_flat():
    """平板型 (acc=0) の語が正しく返されることを確認。"""

    mapping = pyopenjtalk.g2p_mapping("大阪")
    osaka = next(e for e in mapping if e["surface"] == "大阪")
    assert osaka["accent_nucleus"] == 0
    assert osaka["mora_count"] == 4  # オ・オ・サ・カ


def test_g2p_mapping_sp_entry_accent_defaults():
    """sp エントリ (is_ignored=True) のアクセント情報がデフォルト値を持つことを確認。"""

    mapping = pyopenjtalk.g2p_mapping("東京　大阪")
    sp_entries = [e for e in mapping if e["is_ignored"] is True]
    assert len(sp_entries) >= 1
    for sp in sp_entries:
        assert sp["accent_nucleus"] == 0
        assert sp["mora_count"] == 0
        assert sp["chain_flag"] == -1


def test_g2p_mapping_odori_accent_inherited():
    """踊り字展開後のアクセント情報が直前トークンから引き継がれることを確認。"""

    mapping = pyopenjtalk.g2p_mapping("部分々々")
    bubun = next(e for e in mapping if e["surface"] == "部分")
    odori = next(e for e in mapping if e["surface"] == "々々")
    assert odori["accent_nucleus"] == bubun["accent_nucleus"]
    assert odori["mora_count"] == bubun["mora_count"]


def test_g2p_mapping_odori_reanalysis_chain_flag():
    """
    踊り字展開で再解析された feature が直前の語に連結 (chain_flag=1) されることを確認。
    「学生々活」→「学生」「生活」で、「生活」は「学生」の一部を繰り返した語なので連結される。
    「学生生活」を直接入力した場合と同じ chain_flag=1 であるべき。
    """

    njd = pyopenjtalk.run_frontend("学生々活")
    seikatsu = next(f for f in njd if f["string"] == "生活")
    assert seikatsu["chain_flag"] == 1

    # 直接入力した場合と一致すること
    njd_direct = pyopenjtalk.run_frontend("学生生活")
    seikatsu_direct = next(f for f in njd_direct if f["string"] == "生活")
    assert seikatsu["chain_flag"] == seikatsu_direct["chain_flag"]


def test_g2p_mapping_morphs_none_has_all_fields():
    """morphs を渡さない場合でも全フィールドが含まれることを確認。"""

    njd = pyopenjtalk.run_frontend("東京")
    mapping = pyopenjtalk.make_phoneme_mapping(njd)
    assert len(mapping) >= 1
    for entry in mapping:
        # 全フィールドが存在すること
        assert isinstance(entry["surface"], str)
        assert isinstance(entry["phonemes"], list)
        assert isinstance(entry["features"], list)
        assert isinstance(entry["pos"], str)
        assert isinstance(entry["pos_group1"], str)
        assert isinstance(entry["pos_group2"], str)
        assert isinstance(entry["pos_group3"], str)
        assert isinstance(entry["ctype"], str)
        assert isinstance(entry["cform"], str)
        assert isinstance(entry["orig"], str)
        assert isinstance(entry["read"], str)
        assert isinstance(entry["pron"], str)
        assert isinstance(entry["accent_nucleus"], int)
        assert isinstance(entry["mora_count"], int)
        assert isinstance(entry["chain_rule"], str)
        assert isinstance(entry["chain_flag"], int)
        assert isinstance(entry["is_unknown"], bool)
        assert isinstance(entry["is_ignored"], bool)
        # morphs なしの場合 features は空リスト
        assert entry["features"] == []


def test_g2p_mapping_morphs_none_unknown_fallback():
    """morphs を渡さない場合でも NJDFeature ベースの未知語フォールバック判定が動作することを確認。"""

    njd = pyopenjtalk.run_frontend("xtjqは最高")
    mapping = pyopenjtalk.make_phoneme_mapping(njd)
    # フォールバック判定: pos=フィラー + chain_rule=* で is_unknown=True
    unknown_entries = [e for e in mapping if e["is_unknown"] is True]
    assert len(unknown_entries) >= 1


def test_g2p_mapping_integrity():
    """
    g2p_mapping() の surface を連結すると元の入力と一致することを確認。
    `run_frontend()` の surface 再構成とは別に、
    空白・未知語を含む公開 mapping API の出力契約として保持したい。
    """

    text = "吾輩は猫である。名前　はまだ無　い。𰻞𰻞麺を、　食べたい。"
    mapping = pyopenjtalk.g2p_mapping(text)
    reconstructed = "".join(entry["surface"] for entry in mapping)

    assert reconstructed == text


def test_g2p_mapping_unknown_word_rare_kanji_mix():
    """
    Unicode 拡張漢字の未知語と既知語が隣接するケースで、
    `unk` と通常音素が正しく分離されることを確認。
    """

    mapping = pyopenjtalk.g2p_mapping("𰻞𰻞麺")

    assert mapping[0]["surface"] == "𰻞𰻞"
    assert mapping[0]["phonemes"] == ["unk"]
    assert mapping[0]["is_unknown"] is True
    assert mapping[1]["surface"] == "麺"
    assert mapping[1]["phonemes"] == ["m", "e", "N"]
    assert mapping[1]["is_unknown"] is False


@pytest.mark.parametrize("text", FLAG_INVARIANT_CORPUS)
def test_g2p_mapping_flag_invariants(text: str):
    """
    混在コーパスに対し、未知語・無視トークンのフラグ不変条件が崩れないことを確認。

    Haqumei の `test_mapping_flags` の意図を、現在の pyopenjtalk-plus の
    `is_ignored` / `unk` セマンティクスに合わせて移植したもの。
    """

    mapping = pyopenjtalk.g2p_mapping(text)

    for entry in mapping:
        if entry["is_unknown"] is True:
            assert entry["phonemes"] != []
            assert entry["phonemes"] != ["pau"]
        if entry["phonemes"] == ["sp"]:
            assert entry["is_ignored"] is True
        if entry["phonemes"] == []:
            assert entry["is_ignored"] is True
            assert entry["is_unknown"] is False


def test_g2p_recovery_after_error():
    """公開 API でエラーが発生した後も次の g2p() 呼び出しが正常に動作することを確認。"""

    with pytest.raises(RuntimeError, match="too long"):
        pyopenjtalk.g2p("あ" * 10000)

    result = pyopenjtalk.g2p("復帰")
    assert result == "f u cl k i"


def test_g2p_symbols_and_control_chars():
    """記号と制御文字を含む入力でも g2p() がクラッシュしないことを確認。"""

    result = pyopenjtalk.g2p("#$%&'()\n\t")
    assert isinstance(result, str)
    assert len(result) > 0


def test_dounojiten_expansion():
    """
    展開済みの「々」をさらに後続の「々」が引き継ぐケースを確認。
    Haqumei から取り込んだ周期検出ロジックの回帰テスト。
    """

    mapping = pyopenjtalk.g2p_mapping(DOUNOJITEN_TEXT)

    assert _mapping_surface_phonemes(mapping) == DOUNOJITEN_EXPECTED


def test_g2p_mapping_nightmare_case():
    """
    長音吸収・未知語・空白・踊り字連鎖が混在する総合ケースを確認。
    個別テストでは見逃しやすい相互作用の崩れを 1 ケースで検出する。
    """

    mapping = pyopenjtalk.g2p_mapping(NIGHTMARE_MAPPING_TEXT)

    assert _mapping_surface_phonemes(mapping) == NIGHTMARE_MAPPING_EXPECTED


def test_g2p_mapping_compound_accent_phrase_keeps_surfaces():
    """コロン連結辞書による複数アクセント句の分裂で表層が欠けないことを確認。"""

    # naist-jdic の「ありがとうございました」は orig/read/pron のコロン連結により
    # 1つの MeCab エントリから2つの NJD ノードへ分裂する
    mapping = pyopenjtalk.g2p_mapping("本当にありがとうございました、また明日")

    assert [entry["surface"] for entry in mapping] == [
        "本当に",
        "ありがとう",
        "ございました",
        "、",
        "また",
        "明日",
    ]
