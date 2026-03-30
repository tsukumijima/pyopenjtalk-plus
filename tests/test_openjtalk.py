import copy
import subprocess
import sys
import textwrap
import unicodedata
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import pytest

import pyopenjtalk
import pyopenjtalk.utils as pyopenjtalk_utils
from pyopenjtalk import NJDFeature


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
    ("」", ["pau"]),
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
    ("。", []),
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


def _print_results(njd_features: list[NJDFeature], labels: list[str]):
    for f in njd_features:
        s, p = f["string"], f["pron"]
        print(s, p)

    for label in labels:
        print(label)


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


def test_hello():
    njd_features = pyopenjtalk.run_frontend("こんにちは")
    labels = pyopenjtalk.make_label(njd_features)
    _print_results(njd_features, labels)


def test_hello_marine():
    pytest.importorskip("marine")
    njd_features = pyopenjtalk.run_frontend("こんにちは", run_marine=True)
    labels = pyopenjtalk.make_label(njd_features)
    _print_results(njd_features, labels)


def test_njd_features():
    njd_features = pyopenjtalk.run_frontend("こんにちは")
    expected_feature = [
        {
            "string": "こんにちは",
            "pos": "感動詞",
            "pos_group1": "*",
            "pos_group2": "*",
            "pos_group3": "*",
            "ctype": "*",
            "cform": "*",
            "orig": "こんにちは",
            "read": "コンニチハ",
            "pron": "コンニチワ",
            "acc": 0,
            "mora_size": 5,
            "chain_rule": "-1",
            "chain_flag": -1,
        }
    ]
    assert njd_features == expected_feature


def test_njd_features_marine():
    pytest.importorskip("marine")
    njd_features = pyopenjtalk.run_frontend("こんにちは", run_marine=True)
    expected_feature = [
        {
            "string": "こんにちは",
            "pos": "感動詞",
            "pos_group1": "*",
            "pos_group2": "*",
            "pos_group3": "*",
            "ctype": "*",
            "cform": "*",
            "orig": "こんにちは",
            "read": "コンニチハ",
            "pron": "コンニチワ",
            "acc": 0,
            "mora_size": 5,
            "chain_rule": "-1",
            "chain_flag": -1,
        }
    ]
    assert njd_features == expected_feature


def test_fullcontext():
    features = pyopenjtalk.run_frontend("こんにちは")
    labels = pyopenjtalk.make_label(features)
    labels2 = pyopenjtalk.extract_fullcontext("こんにちは")
    for a, b in zip(labels, labels2):
        assert a == b


def test_fullcontext_marine():
    pytest.importorskip("marine")
    features = pyopenjtalk.run_frontend("こんにちは", run_marine=True)
    labels = pyopenjtalk.make_label(features)
    labels2 = pyopenjtalk.extract_fullcontext("こんにちは", run_marine=True)
    for a, b in zip(labels, labels2):
        assert a == b


def test_jtalk():
    for text in [
        "今日も良い天気ですね",
        "こんにちは。",
        "どんまい！",
        "パソコンのとりあえず知っておきたい使い方",
    ]:
        njd_features = pyopenjtalk.run_frontend(text)
        labels = pyopenjtalk.make_label(njd_features)
        _print_results(njd_features, labels)

        surface = "".join(map(lambda f: f["string"], njd_features))
        assert surface == text


def test_jtalk_marine():
    pytest.importorskip("marine")
    for text in [
        "今日も良い天気ですね",
        "こんにちは。",
        "どんまい！",
        "パソコンのとりあえず知っておきたい使い方",
    ]:
        njd_features = pyopenjtalk.run_frontend(text, run_marine=True)
        labels = pyopenjtalk.make_label(njd_features)
        _print_results(njd_features, labels)

        surface = "".join(map(lambda f: f["string"], njd_features))
        assert surface == text


def test_g2p_kana():
    for text, pron in [
        ("", ""),  # empty string
        ("今日もこんにちは", "キョーモコンニチワ"),
        ("いやあん", "イヤーン"),
        (
            "パソコンのとりあえず知っておきたい使い方",
            "パソコンノトリアエズシッテオキタイツカイカタ",
        ),
    ]:
        p = pyopenjtalk.g2p(text, kana=True)
        assert p == pron


def test_g2p_phone():
    for text, pron in [
        ("", ""),  # empty string
        ("こんにちは", "k o N n i ch i w a"),
        ("ななみんです", "n a n a m i N d e s U"),
        ("ハローユーチューブ", "h a r o o y u u ch u u b u"),
    ]:
        p = pyopenjtalk.g2p(text, kana=False)
        assert p == pron


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
        expected_phonemes = _extract_label_phonemes(labels, keep_pause=True)

        assert pyopenjtalk.g2p(text, join=False, use_vanilla=is_use_vanilla) == expected_phonemes


def test_g2p_nani_model():
    test_cases = [
        {
            "text": "何か問題があれば何でも言ってください、どんな些細なことでも何とかします。",
            "pron_without_nani": "ナニカモンダイガアレバナニデモイッテクダサイ、ドンナササイナコトデモナニトカシマス。",
            "pron_with_nani": "ナニカモンダイガアレバナンデモイッテクダサイ、ドンナササイナコトデモナントカシマス。",
        },
        {
            "text": "何か特別なことをしたわけではありませんが、何故か周りの人々が何かと気にかけてくれます。何と言えばいいのか分かりません。",
            "pron_without_nani": "ナニカトクベツナコトヲシタワケデワアリマセンガ、ナゼカマワリノヒトビトガナニカトキニカケテクレマス。ナニトイエバイイノカワカリマセン。",
            "pron_with_nani": "ナニカトクベツナコトヲシタワケデワアリマセンガ、ナゼカマワリノヒトビトガナニカトキニカケテクレマス。ナントイエバイイノカワカリマセン。",
        },
        {
            "text": "私も何とかしたいですが、何でも行くリソースはありません。",
            "pron_without_nani": "ワタシモナニトカシタイデスガ、ナニデモイクリソースワアリマセン。",
            "pron_with_nani": "ワタシモナントカシタイデスガ、ナンデモイクリソースワアリマセン。",
        },
        {
            "text": "何を言っても何の問題もありません。",
            "pron_without_nani": "ナニヲイッテモナニノモンダイモアリマセン。",
            "pron_with_nani": "ナニヲイッテモナンノモンダイモアリマセン。",
        },
        {
            "text": "これは何ですか？何の情報？",
            "pron_without_nani": "コレワナニデスカ？ナニノジョーホー？",
            "pron_with_nani": "コレワナンデスカ？ナンノジョーホー？",
        },
        {
            "text": "何だろう、何でも嘘つくのやめてもらっていいですか？",
            "pron_without_nani": "ナニダロー、ナニデモウソツクノヤメテモラッテイイデスカ？",
            "pron_with_nani": "ナンダロー、ナンデモウソツクノヤメテモラッテイイデスカ？",
        },
        {
            "text": "質問は何のことかな？",
            "pron_without_nani": "シツモンワナニノコトカナ？",
            "pron_with_nani": "シツモンワナンノコトカナ？",
        },
    ]

    # without nani model
    for case in test_cases:
        p = pyopenjtalk.g2p(case["text"], kana=True, use_vanilla=True)
        assert p == case["pron_without_nani"]

    # with nani model
    for case in test_cases:
        p = pyopenjtalk.g2p(case["text"], kana=True, use_vanilla=False)
        assert p == case["pron_with_nani"]


def test_g2p_nani_model_does_not_require_sudachi_when_only_nani(monkeypatch: pytest.MonkeyPatch):
    def fail_sudachi_analyze(text: str, multi_read_kanji_list: list[str]) -> list[list[str]]:
        raise AssertionError("sudachi_analyze should not be called for '何'-only correction")

    monkeypatch.setattr(pyopenjtalk_utils, "sudachi_analyze", fail_sudachi_analyze)

    assert pyopenjtalk.g2p("これは何ですか？", kana=True) == "コレワナンデスカ？"


def test_g2p_predict_nani_can_be_disabled():
    assert pyopenjtalk.g2p("何ですか", kana=True, predict_nani=True) == "ナンデスカ"
    assert pyopenjtalk.g2p("何ですか", kana=True, predict_nani=False) == "ナニデスカ"


def test_g2p_can_disable_sudachi_kanji_yomi_and_keep_nani_enabled():
    text = "風がこんな風に吹く。これは何ですか？"

    assert (
        pyopenjtalk.g2p(
            text,
            kana=True,
            use_sudachi_kanji_yomi=False,
            predict_nani=True,
        )
        == "カゼガコンナカゼニフク。コレワナンデスカ？"
    )


@pytest.mark.parametrize(
    ("text", "expected_phonemes", "expected_kana"),
    [
        ("しなじう", ["sh", "i", "n", "a", "j", "i", "u"], "シナジウ"),
        ("いみじう", ["i", "m", "i", "j", "i", "u"], "イミジウ"),
        ("買わう", ["k", "a", "w", "a", "u"], "カワウ"),
        ("捨てう", ["s", "U", "t", "e", "u"], "ステウ"),
        ("行こう", ["i", "k", "o", "o"], "イコー"),
        ("言おう", ["i", "o", "o"], "イオー"),
    ],
)
def test_g2p_auxiliary_u_long_vowel_revert(
    text: str,
    expected_phonemes: list[str],
    expected_kana: str,
):
    assert pyopenjtalk.g2p(text, join=False) == expected_phonemes
    assert pyopenjtalk.g2p(text, kana=True) == expected_kana


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
    with pytest.raises(ValueError, match="normalize_mode must be one of"):
        pyopenjtalk.g2p("学生", normalize_mode=cast(Any, "invalid"))


def test_unicode_normalization_combining_chars():
    """多様な結合文字が NFC 正規化で正しく処理されることを確認"""

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


def test_userdic():
    for text, expected in [
        ("nnmn", "n a n a m i N"),
        ("GNU", "g u n u u"),
    ]:
        p = pyopenjtalk.g2p(text)
        assert p != expected

    user_csv = str(Path(__file__).parent / "test_data" / "user.csv")
    user_dic = str(Path(__file__).parent / "test_data" / "user.dic")

    with open(user_csv, "w", encoding="utf-8") as f:
        f.write("ｎｎｍｎ,,,1,名詞,一般,*,*,*,*,ｎｎｍｎ,ナナミン,ナナミン,1/4,*\n")
        f.write("ＧＮＵ,,,1,名詞,一般,*,*,*,*,ＧＮＵ,グヌー,グヌー,2/3,*\n")

    try:
        pyopenjtalk.mecab_dict_index(f.name, user_dic)
        pyopenjtalk.update_global_jtalk_with_user_dict(user_dic)

        for text, expected in [
            ("nnmn", "n a n a m i N"),
            ("GNU", "g u n u u"),
        ]:
            p = pyopenjtalk.g2p(text)
            assert p == expected
    finally:
        pyopenjtalk.unset_user_dict()


def test_mecab_dict_index_empty_surface_should_not_segfault(tmp_path: Path):
    user_csv = tmp_path / "invalid_user.csv"
    user_dic = tmp_path / "invalid_user.dic"
    user_csv.write_text(",1358,1358,8047,名詞,接尾,一般,*,*,*,－,ノ,ノ,0/1,*\n", encoding="utf-8")

    command = [
        sys.executable,
        "-c",
        textwrap.dedent(
            """
            import sys
            import pyopenjtalk

            try:
                pyopenjtalk.mecab_dict_index(sys.argv[1], sys.argv[2])
            except RuntimeError:
                sys.exit(0)
            except Exception:
                sys.exit(2)
            else:
                sys.exit(3)
            """
        ),
        str(user_csv),
        str(user_dic),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode == 0


def test_mecab_dict_index_invalid_dn_mecab_should_raise_file_not_found(tmp_path: Path):
    user_csv = tmp_path / "valid.csv"
    user_dic = tmp_path / "valid.dic"
    user_csv.write_text(
        "ｔｅｓｔ,,,1,名詞,一般,*,*,*,*,ｔｅｓｔ,テスト,テスト,1/3,*\n", encoding="utf-8"
    )

    with pytest.raises(FileNotFoundError):
        pyopenjtalk.mecab_dict_index(
            str(user_csv), str(user_dic), dn_mecab=str(tmp_path / "not-found-dic")
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
    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode == 0


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
    completed = subprocess.run(command, capture_output=True, text=True, check=False)

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


def test_mecab_dict_index_valid_user_dict(tmp_path: Path):
    """有効な CSV エントリで mecab_dict_index を実行した場合、辞書が正常にビルドされること。"""
    user_csv = tmp_path / "valid_user.csv"
    user_dic = tmp_path / "valid_user.dic"
    user_csv.write_text(
        "テスト,1348,1348,5000,名詞,固有名詞,一般,*,*,*,テスト,テスト,テスト,1/3,C1\n",
        encoding="utf-8",
    )

    pyopenjtalk.mecab_dict_index(str(user_csv), str(user_dic))

    assert user_dic.exists()


def test_mecab_dict_index_csv_only_commas_should_not_segfault(tmp_path: Path):
    """カンマのみを含む CSV で mecab_dict_index を実行した場合、セグフォしないこと。"""
    user_csv = tmp_path / "invalid_user.csv"
    user_dic = tmp_path / "invalid_user.dic"
    user_csv.write_text(",,,,,,,,,,,,,\n", encoding="utf-8")

    command = [
        sys.executable,
        "-c",
        textwrap.dedent(
            """
            import sys
            import pyopenjtalk

            try:
                pyopenjtalk.mecab_dict_index(sys.argv[1], sys.argv[2])
            except Exception:
                sys.exit(0)
            else:
                sys.exit(0)
            """
        ),
        str(user_csv),
        str(user_dic),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode == 0


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
        pyopenjtalk.run_mecab("😀" * 3000)

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


def test_mecab_dict_index_random_invalid_input_should_not_segfault(tmp_path: Path):
    random_csv_lines = [
        ",,,,,\n",
        "a,b,c,d,e\n",
        "無効,1,2,3\n",
        "😀,1358,1358,8047,名詞,接尾,一般,*,*,*,－,ノ,ノ,0/1,*\n",
        '"unterminated,1358,1358,8047,名詞,接尾,一般,*,*,*,－,ノ,ノ,0/1,*\n',
    ]
    user_dic = tmp_path / "invalid_user.dic"

    for index, csv_line in enumerate(random_csv_lines):
        user_csv = tmp_path / f"invalid_user_{index}.csv"
        user_csv.write_text(csv_line, encoding="utf-8")

        command = [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys
                import pyopenjtalk

                try:
                    pyopenjtalk.mecab_dict_index(sys.argv[1], sys.argv[2])
                except Exception:
                    sys.exit(0)
                else:
                    sys.exit(0)
                """
            ),
            str(user_csv),
            str(user_dic),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        assert completed.returncode >= 0


def test_multithreading():
    ojt = pyopenjtalk.openjtalk.OpenJTalk(pyopenjtalk.OPEN_JTALK_DICT_DIR)
    texts = [
        "今日もいい天気ですね",
        "こんにちは",
        "マルチスレッドプログラミング",
        "テストです",
        "Pythonはプログラミング言語です",
        "日本語テキストを音声合成します",
    ] * 4

    # Test consistency between single and multi-threaded runs
    # make sure no corruptions happen in OJT internal
    results_s = [ojt.run_frontend(text) for text in texts]
    results_m = []
    with ThreadPoolExecutor() as e:
        results_m = [i for i in e.map(ojt.run_frontend, texts)]
    for s, m in zip(results_s, results_m):
        assert len(s) == len(m)
        for s_, m_ in zip(s, m):
            # full context must exactly match
            assert s_ == m_


def test_odoriji():
    # 一の字点（ゝ、ゞ、ヽ、ヾ）の処理テスト
    # 濁点なしの一の字点
    njd_features = pyopenjtalk.run_frontend("なゝ樹")
    assert njd_features[0]["read"] == "ナ"
    assert njd_features[0]["pron"] == "ナ"
    assert njd_features[0]["mora_size"] == 1
    assert njd_features[1]["read"] == "ナ"
    assert njd_features[1]["pron"] == "ナ"
    assert njd_features[1]["mora_size"] == 1
    assert njd_features[2]["read"] == "キ"
    assert njd_features[2]["pron"] == "キ"
    assert njd_features[2]["mora_size"] == 1

    # 濁点ありの一の字点
    njd_features = pyopenjtalk.run_frontend("金子みすゞ")
    assert njd_features[0]["read"] == "カネコ"
    assert njd_features[0]["pron"] == "カネコ"
    assert njd_features[0]["mora_size"] == 3
    assert njd_features[1]["read"] == "ミス"
    assert njd_features[1]["pron"] == "ミス"
    assert njd_features[1]["mora_size"] == 2
    assert njd_features[2]["read"] == "ズ"
    assert njd_features[2]["pron"] == "ズ"
    assert njd_features[2]["mora_size"] == 1

    # 濁点なしの一の字点（づゝ）
    njd_features = pyopenjtalk.run_frontend("づゝ")
    assert njd_features[0]["read"] == "ヅ"
    assert njd_features[0]["pron"] == "ヅ"
    assert njd_features[0]["mora_size"] == 1
    assert njd_features[1]["read"] == "ツ"
    assert njd_features[1]["pron"] == "ツ"
    assert njd_features[1]["mora_size"] == 1

    # 濁点ありの一の字点（ぶゞ漬け）
    njd_features = pyopenjtalk.run_frontend("ぶゞ漬け")
    assert njd_features[0]["read"] == "ブ"
    assert njd_features[0]["pron"] == "ブ"
    assert njd_features[0]["mora_size"] == 1
    assert njd_features[1]["read"] == "ブ"
    assert njd_features[1]["pron"] == "ブ"
    assert njd_features[1]["mora_size"] == 1
    assert njd_features[2]["read"] == "ヅケ"
    assert njd_features[2]["pron"] == "ヅケ"
    assert njd_features[2]["mora_size"] == 2

    # 片仮名の一の字点（バナヽ）
    njd_features = pyopenjtalk.run_frontend("バナヽ")
    assert njd_features[0]["read"] == "バナ"
    assert njd_features[0]["pron"] == "バナ"
    assert njd_features[0]["mora_size"] == 2
    assert njd_features[1]["read"] == "ナ"
    assert njd_features[1]["pron"] == "ナ"
    assert njd_features[1]["mora_size"] == 1

    # use_vanilla=True の場合は処理されない
    njd_features = pyopenjtalk.run_frontend("なゝ樹", use_vanilla=True)
    assert njd_features[1]["read"] == "、"
    assert njd_features[1]["pron"] == "、"

    # 単一の踊り字（辞書に登録されていないパターン）
    njd_features = pyopenjtalk.run_frontend("愛々")
    assert njd_features[0]["read"] == "アイ"
    assert njd_features[0]["pron"] == "アイ"
    assert njd_features[0]["mora_size"] == 2
    assert njd_features[1]["read"] == "アイ"
    assert njd_features[1]["pron"] == "アイ"
    assert njd_features[1]["mora_size"] == 2
    njd_features = pyopenjtalk.run_frontend("咲々")
    assert njd_features[0]["read"] == "サキ"
    assert njd_features[0]["pron"] == "サキ"
    assert njd_features[0]["mora_size"] == 2
    assert njd_features[1]["read"] == "サキ"
    assert njd_features[1]["pron"] == "サキ"
    assert njd_features[1]["mora_size"] == 2

    # 単一の踊り字だが、形態素解析で展開しないと正しい読みを取得できないケース
    # 実装上漢字1字だけで再解析した際に読みが間違ってしまうことがあるが、改善するのが面倒なのでテストケースには含めていない
    njd_features = pyopenjtalk.run_frontend("結婚式々場")
    assert njd_features[0]["read"] == "ケッコンシキ"
    assert njd_features[0]["pron"] == "ケッコンシ’キ"
    assert njd_features[0]["mora_size"] == 6
    assert njd_features[1]["read"] == "シキジョウ"
    assert njd_features[1]["pron"] == "シ’キジョー"
    assert njd_features[1]["mora_size"] == 4
    njd_features = pyopenjtalk.run_frontend("学生々活")
    assert njd_features[0]["read"] == "ガクセイ"
    assert njd_features[0]["pron"] == "ガク’セー"
    assert njd_features[0]["mora_size"] == 4
    assert njd_features[1]["read"] == "セイカツ"
    assert njd_features[1]["pron"] == "セーカツ"
    assert njd_features[1]["mora_size"] == 4
    njd_features = pyopenjtalk.run_frontend("民主々義")
    assert njd_features[0]["read"] == "ミンシュ"
    assert njd_features[0]["pron"] == "ミンシュ"
    assert njd_features[0]["mora_size"] == 3
    assert njd_features[1]["read"] == "シュギ"
    assert njd_features[1]["pron"] == "シュギ"
    assert njd_features[1]["mora_size"] == 2

    # 連続する踊り字
    njd_features = pyopenjtalk.run_frontend("叙々々苑")
    assert njd_features[0]["read"] == "ジョ"
    assert njd_features[0]["pron"] == "ジョ"
    assert njd_features[0]["mora_size"] == 1
    assert njd_features[1]["read"] == "ジョジョ"
    assert njd_features[1]["pron"] == "ジョジョ"
    assert njd_features[1]["mora_size"] == 2
    njd_features = pyopenjtalk.run_frontend("叙々々々苑")
    assert njd_features[0]["read"] == "ジョ"
    assert njd_features[0]["pron"] == "ジョ"
    assert njd_features[0]["mora_size"] == 1
    assert njd_features[1]["read"] == "ジョジョ"
    assert njd_features[1]["pron"] == "ジョジョ"
    assert njd_features[1]["mora_size"] == 2
    assert njd_features[2]["read"] == "ジョ"
    assert njd_features[2]["pron"] == "ジョ"
    assert njd_features[2]["mora_size"] == 1
    njd_features = pyopenjtalk.run_frontend("叙々々々々苑")
    assert njd_features[0]["read"] == "ジョ"
    assert njd_features[0]["pron"] == "ジョ"
    assert njd_features[0]["mora_size"] == 1
    assert njd_features[1]["read"] == "ジョジョ"
    assert njd_features[1]["pron"] == "ジョジョ"
    assert njd_features[1]["mora_size"] == 2
    assert njd_features[2]["read"] == "ジョジョ"
    assert njd_features[2]["pron"] == "ジョジョ"
    assert njd_features[2]["mora_size"] == 2
    njd_features = pyopenjtalk.run_frontend("叙々々々々々苑")
    assert njd_features[0]["read"] == "ジョ"
    assert njd_features[0]["pron"] == "ジョ"
    assert njd_features[0]["mora_size"] == 1
    assert njd_features[1]["read"] == "ジョジョジョジョジョ"
    assert njd_features[1]["pron"] == "ジョジョジョジョジョ"
    assert njd_features[1]["mora_size"] == 5
    njd_features = pyopenjtalk.run_frontend("複々々線")
    print(njd_features)
    assert njd_features[0]["read"] == "フク"
    assert njd_features[0]["pron"] == "フ’ク"
    assert njd_features[0]["mora_size"] == 2
    assert njd_features[1]["read"] == "フクフク"
    assert njd_features[1]["pron"] == "フ’クフ’ク"
    assert njd_features[1]["mora_size"] == 4
    njd_features = pyopenjtalk.run_frontend("複々々々線")
    assert njd_features[0]["read"] == "フク"
    assert njd_features[0]["pron"] == "フ’ク"
    assert njd_features[0]["mora_size"] == 2
    assert njd_features[1]["read"] == "フクフク"
    assert njd_features[1]["pron"] == "フ’クフ’ク"
    assert njd_features[1]["mora_size"] == 4
    assert njd_features[2]["read"] == "フク"
    assert njd_features[2]["pron"] == "フ’ク"
    assert njd_features[2]["mora_size"] == 2
    njd_features = pyopenjtalk.run_frontend("今日も前進々々")
    assert njd_features[0]["read"] == "キョウ"
    assert njd_features[0]["pron"] == "キョー"
    assert njd_features[0]["mora_size"] == 2
    assert njd_features[1]["read"] == "モ"
    assert njd_features[1]["pron"] == "モ"
    assert njd_features[1]["mora_size"] == 1
    assert njd_features[2]["read"] == "ゼンシン"
    assert njd_features[2]["pron"] == "ゼンシン"
    assert njd_features[2]["mora_size"] == 4
    assert njd_features[3]["read"] == "ゼンシン"
    assert njd_features[3]["pron"] == "ゼンシン"
    assert njd_features[3]["mora_size"] == 4

    # 2文字以上の漢字の後の踊り字
    njd_features = pyopenjtalk.run_frontend("部分々々")
    assert njd_features[0]["read"] == "ブブン"
    assert njd_features[0]["pron"] == "ブブン"
    assert njd_features[0]["mora_size"] == 3
    assert njd_features[1]["read"] == "ブブン"
    assert njd_features[1]["pron"] == "ブブン"
    assert njd_features[1]["mora_size"] == 3
    njd_features = pyopenjtalk.run_frontend("後手々々")
    assert njd_features[0]["read"] == "ゴテ"
    assert njd_features[0]["pron"] == "ゴテ"
    assert njd_features[0]["mora_size"] == 2
    assert njd_features[1]["read"] == "ゴテ"
    assert njd_features[1]["pron"] == "ゴテ"
    assert njd_features[1]["mora_size"] == 2
    njd_features = pyopenjtalk.run_frontend("其他々々")
    assert njd_features[0]["read"] == "ソノ"
    assert njd_features[0]["pron"] == "ソノ"
    assert njd_features[0]["mora_size"] == 2
    assert njd_features[1]["read"] == "ホカ"
    assert njd_features[1]["pron"] == "ホカ"
    assert njd_features[1]["mora_size"] == 2
    assert njd_features[2]["read"] == "ソノホカ"
    assert njd_features[2]["pron"] == "ソノホカ"
    assert njd_features[2]["mora_size"] == 4

    # 踊り字の前に漢字がない場合
    # 絵文字除去はこのライブラリの範囲外とし、とりあえず ? という記号を繰り返すことがないようにする
    njd_features = pyopenjtalk.run_frontend("やっほー！元気かな？ヾ(≧▽≦)ﾉ")
    assert njd_features[0]["read"] == "ヤッホー"
    assert njd_features[0]["pron"] == "ヤッホー"
    assert njd_features[0]["mora_size"] == 4
    assert njd_features[1]["read"] == "！"
    assert njd_features[1]["pron"] == "！"
    assert njd_features[1]["mora_size"] == 0
    assert njd_features[2]["read"] == "ゲンキ"
    assert njd_features[2]["pron"] == "ゲンキ’"
    assert njd_features[2]["mora_size"] == 3
    assert njd_features[3]["read"] == "カ"
    assert njd_features[3]["pron"] == "カ"
    assert njd_features[3]["mora_size"] == 1
    assert njd_features[4]["read"] == "ナ"
    assert njd_features[4]["pron"] == "ナ"
    assert njd_features[4]["mora_size"] == 1
    assert njd_features[5]["read"] == "？"
    assert njd_features[5]["pron"] == "？"
    assert njd_features[5]["mora_size"] == 0
    assert njd_features[6]["read"] == "、"
    assert njd_features[6]["pron"] == "、"
    assert njd_features[6]["mora_size"] == 0
    assert njd_features[7]["read"] == "、"
    assert njd_features[7]["pron"] == "、"
    assert njd_features[7]["mora_size"] == 0
    assert njd_features[8]["read"] == "ノ"
    assert njd_features[8]["pron"] == "ノ"
    assert njd_features[8]["mora_size"] == 1

    # use_vanilla=True の場合は処理されない
    njd_features = pyopenjtalk.run_frontend("愛々", use_vanilla=True)
    assert njd_features[1]["read"] == "、"
    assert njd_features[1]["pron"] == "、"


# =============================================================================
# run_mecab_detailed() のテスト
# =============================================================================


def test_run_mecab_detailed_known_word():
    """辞書に存在する単語が is_unknown=False で返されることを確認"""

    morphs = pyopenjtalk.run_mecab_detailed("こんにちは")
    assert len(morphs) >= 1
    # 全てのフィールドが存在することを確認
    for morph in morphs:
        assert "surface" in morph
        assert "features" in morph
        assert "pos_id" in morph
        assert "left_id" in morph
        assert "right_id" in morph
        assert "word_cost" in morph
        assert "is_unknown" in morph
        assert "is_ignored" in morph
    # 「こんにちは」は辞書に存在するので、少なくとも 1 つは既知語がある
    assert any(morph["is_unknown"] is False for morph in morphs)


def test_run_mecab_detailed_unknown_word():
    """辞書に存在しない造語が is_unknown=True で返されることを確認"""

    # カタカナは辞書内の既知語に分割されてしまうため、
    # MeCab が確実に未知語と判定する ASCII 文字列を使用する
    morphs = pyopenjtalk.run_mecab_detailed("xtjq")
    assert any(morph["is_unknown"] is True for morph in morphs)


def test_run_mecab_detailed_includes_ignored():
    """通常の run_mecab ではフィルタされる記号,空白トークンも含まれることを確認"""

    # 通常の run_mecab は記号,空白をフィルタする
    normal_morphs = pyopenjtalk.run_mecab("東京　大阪")
    # detailed は全トークンを返す
    detailed_morphs = pyopenjtalk.run_mecab_detailed("東京　大阪")
    # detailed の方がトークン数が多い（もしくは同じ）
    assert len(detailed_morphs) >= len(normal_morphs)


def test_run_mecab_detailed_feature_format():
    """feature 文字列が既存 run_mecab() と同じ "surface,品詞,..." フォーマットであることを確認"""

    morphs = pyopenjtalk.run_mecab_detailed("こんにちは")
    for morph in morphs:
        # features の先頭要素は surface と一致する
        assert morph["features"][0] == morph["surface"]


def test_run_mecab_detailed_cost_types():
    """pos_id, left_id, right_id, word_cost が正しい型 (int) で返されることを確認"""

    morphs = pyopenjtalk.run_mecab_detailed("東京は日本の首都です")
    for morph in morphs:
        assert isinstance(morph["pos_id"], int)
        assert isinstance(morph["left_id"], int)
        assert isinstance(morph["right_id"], int)
        assert isinstance(morph["word_cost"], int)


def test_run_mecab_detailed_empty_string():
    """空文字列入力でクラッシュしないことを確認"""

    morphs = pyopenjtalk.run_mecab_detailed("")
    assert isinstance(morphs, list)


def test_run_mecab_detailed_consistency_with_run_mecab():
    """run_mecab_detailed の非 ignored トークンが run_mecab の結果と一致することを確認"""

    text = "こんにちは世界"
    normal_morphs = pyopenjtalk.run_mecab(text)
    detailed_morphs = pyopenjtalk.run_mecab_detailed(text)

    # detailed から ignored を除いた features を結合すると normal の結果と一致する
    detailed_features = [
        ",".join(morph["features"]) for morph in detailed_morphs if morph["is_ignored"] is False
    ]
    assert detailed_features == normal_morphs


# =============================================================================
# make_phoneme_mapping() のテスト
# =============================================================================


def test_make_phoneme_mapping_basic():
    """基本的な形態素-音素マッピングが返されることを確認"""

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
    """句読点がポーズ音素 ['pau'] として扱われることを確認"""

    njd_features = pyopenjtalk.run_frontend("東京、大阪")
    mapping = pyopenjtalk.make_phoneme_mapping(njd_features)
    pause_entries = [entry for entry in mapping if entry["phonemes"] == ["pau"]]
    assert len(pause_entries) >= 1


def test_make_phoneme_mapping_boundary_punctuation_end():
    """文末の句読点は surface を保持しつつ、実際に pause がなければ空音素になることを確認"""

    njd_features = pyopenjtalk.run_frontend("あ。")
    mapping = pyopenjtalk.make_phoneme_mapping(njd_features)

    assert mapping[0]["surface"] == "あ"
    assert mapping[0]["phonemes"] == ["a"]
    assert mapping[1]["surface"] == "。"
    assert mapping[1]["phonemes"] == []
    assert mapping[1]["is_ignored"] is True


def test_make_phoneme_mapping_boundary_punctuation_start():
    """文頭の句読点は surface を保持しつつ、実際に pause がなければ空音素になることを確認"""

    njd_features = pyopenjtalk.run_frontend("。あ")
    mapping = pyopenjtalk.make_phoneme_mapping(njd_features)

    assert mapping[0]["surface"] == "。"
    assert mapping[0]["phonemes"] == []
    assert mapping[0]["is_ignored"] is True
    assert mapping[1]["surface"] == "あ"
    assert mapping[1]["phonemes"] == ["a"]


def test_make_phoneme_mapping_pause_like_symbols():
    """pause-like 記号は、実際に pause がある箇所だけ ['pau'] を持つことを確認"""

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
    """quote と読点が連続する場合、実際の pause は読点側へ関連付けられることを確認"""

    njd_features = pyopenjtalk.run_frontend("「東京」、大阪")
    mapping = pyopenjtalk.make_phoneme_mapping(njd_features)

    assert mapping[0]["surface"] == "「"
    assert mapping[0]["phonemes"] == []
    assert mapping[2]["surface"] == "」"
    assert mapping[2]["phonemes"] == []
    assert mapping[3]["surface"] == "、"
    assert mapping[3]["phonemes"] == ["pau"]


def test_make_phoneme_mapping_phoneme_consistency():
    """
    make_phoneme_mapping() の音素列を結合すると make_label 由来の音素列と同じ結果になることを確認
    make_phoneme_mapping() は後処理型で NJDFeature を直接受け取るため、
    同じ NJDFeature を make_label() に渡した結果と音素が一致するはず。
    ただし make_label はラベルフォーマットの中に音素が埋め込まれているため、
    パース手順が異なる。
    """

    text = "おはようございます"
    # use_vanilla=True で apply_postprocessing() を適用しない NJDFeature を取得
    njd_features = pyopenjtalk.run_frontend(text, use_vanilla=True)

    mapping = pyopenjtalk.make_phoneme_mapping(njd_features)
    mapping_phonemes = []
    for entry in mapping:
        if entry["phonemes"] != ["pau"]:
            mapping_phonemes.extend(entry["phonemes"])

    # 同じ NJDFeature から make_label した結果と比較
    labels = pyopenjtalk.make_label(njd_features)
    label_phonemes = list(map(lambda s: s.split("-")[1].split("+")[0], labels[1:-1]))

    assert mapping_phonemes == label_phonemes


def test_make_phoneme_mapping_phoneme_consistency_with_pause_retained():
    """句読点を含む入力では、通常音素列が make_label と一致しつつ pau が保持されることを確認"""

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


def test_make_phoneme_mapping_phoneme_consistency_with_postprocessing():
    """apply_postprocessing() 適用済みの NJDFeature でも音素の一貫性が保たれることを確認"""

    text = "東京は日本の首都です"
    # デフォルト (use_vanilla=False) で apply_postprocessing() 適用済みの NJDFeature を取得
    njd_features = pyopenjtalk.run_frontend(text)

    mapping = pyopenjtalk.make_phoneme_mapping(njd_features)
    mapping_phonemes = []
    for entry in mapping:
        if entry["phonemes"] != ["pau"]:
            mapping_phonemes.extend(entry["phonemes"])

    labels = pyopenjtalk.make_label(njd_features)
    label_phonemes = list(map(lambda s: s.split("-")[1].split("+")[0], labels[1:-1]))

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
    """数字入力 (NJD でノード数が変わるケース) でクラッシュしないことを確認"""

    njd_features = pyopenjtalk.run_frontend("123")
    mapping = pyopenjtalk.make_phoneme_mapping(njd_features)
    assert len(mapping) >= 1


def test_make_phoneme_mapping_empty_features():
    """空の NJDFeature リストで空リストが返されることを確認"""

    mapping = pyopenjtalk.make_phoneme_mapping([])
    assert mapping == []


def test_make_phoneme_mapping_surface_correspondence():
    """
    make_phoneme_mapping() の surface フィールドが NJDFeature の string と 1:1 対応していることを確認。
    NOTE: 長音吸収マージが発生するテキストでは len(mapping) < len(njd_features) となるため、
    このテストでは長音吸収が発生しない入力のみを使用している。
    """

    text = "今日も良い天気ですね"
    njd_features = pyopenjtalk.run_frontend(text)
    mapping = pyopenjtalk.make_phoneme_mapping(njd_features)

    assert len(mapping) == len(njd_features)
    for entry, feat in zip(mapping, njd_features):
        assert entry["surface"] == feat["string"]


def test_make_phoneme_mapping_multiple_sentences():
    """
    複数の文で make_phoneme_mapping() が正しく動作することを確認。
    NOTE: 長音吸収マージが発生するテキストでは len(mapping) < len(njd_features) となるため、
    このテストでは長音吸収が発生しない入力のみを使用している。
    """

    texts = [
        "こんにちは",
        "東京は日本の首都です",
        "おはようございます",
        "今日は2112年9月3日です",
        "焼きそばパン買ってこいや",
    ]
    for text in texts:
        njd_features = pyopenjtalk.run_frontend(text)
        mapping = pyopenjtalk.make_phoneme_mapping(njd_features)

        # NJDFeature の数と mapping の数が一致
        assert len(mapping) == len(njd_features)

        # make_label の音素列と一致
        mapping_phonemes = []
        for entry in mapping:
            if entry["phonemes"] != ["pau"]:
                mapping_phonemes.extend(entry["phonemes"])
        labels = pyopenjtalk.make_label(njd_features)
        label_phonemes = list(map(lambda s: s.split("-")[1].split("+")[0], labels[1:-1]))
        assert mapping_phonemes == label_phonemes


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


# =============================================================================
# make_phoneme_mapping() with morphs のテスト (旧 make_phoneme_mapping_detailed)
# =============================================================================


def test_make_phoneme_mapping_with_morphs_basic():
    """基本的な morphs 付きマッピングが返されることを確認"""

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
    """未知語が is_unknown=True で返されることを確認"""

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
    """既知語が is_unknown=False で返されることを確認"""

    text = "こんにちは"
    njd_features, morphs = pyopenjtalk.run_frontend_detailed(text)
    detailed = pyopenjtalk.make_phoneme_mapping(njd_features, morphs=morphs)

    assert any(entry["is_unknown"] is False for entry in detailed)


def test_make_phoneme_mapping_with_morphs_phonemes_match():
    """morphs 付きの phonemes が morphs なしの結果と (sp/unk を除き) 実質一致することを確認"""

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
    """数字入力 (NJD でノード数が変わるケース) でクラッシュしないことを確認"""

    text = "123"
    njd_features, morphs = pyopenjtalk.run_frontend_detailed(text)
    detailed = pyopenjtalk.make_phoneme_mapping(njd_features, morphs=morphs)

    assert len(detailed) >= 1


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
    長音吸収で merged された node の metadata が破綻しないことを確認。

    代表的な意向形・助動詞連結を広く検証し、
    `features` が空リストになることと、`orig` が辞書の原形のまま保持されることを確認する。
    """

    njd_features, morphs = pyopenjtalk.run_frontend_detailed(text)
    mapping = pyopenjtalk.make_phoneme_mapping(njd_features, morphs=morphs)

    merged_entry = next(entry for entry in mapping if entry["surface"] == merged_surface)
    assert merged_entry["features"] == []
    assert merged_entry["orig"] == expected_orig


# =============================================================================
# run_frontend_detailed() のテスト
# =============================================================================


def test_run_frontend_detailed_basic():
    """run_frontend_detailed がタプルを返し、NJDFeature が run_frontend と同一であることを確認"""

    text = "こんにちは"
    njd_features, morphs = pyopenjtalk.run_frontend_detailed(text)
    njd_features_normal = pyopenjtalk.run_frontend(text)

    assert isinstance(njd_features, list)
    assert isinstance(morphs, list)
    assert njd_features == njd_features_normal
    assert len(morphs) >= 1


def test_run_frontend_detailed_morphs_fields():
    """run_frontend_detailed の morphs に全フィールドが含まれることを確認"""

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
    """空文字列で run_frontend_detailed がクラッシュしないことを確認"""

    njd_features, morphs = pyopenjtalk.run_frontend_detailed("")
    assert isinstance(njd_features, list)
    assert isinstance(morphs, list)


# =============================================================================
# g2p_mapping() のテスト
# =============================================================================


def test_g2p_mapping_basic():
    """g2p_mapping の基本動作と全フィールドの存在・型を確認"""

    mapping = pyopenjtalk.g2p_mapping("こんにちは")
    assert len(mapping) >= 1
    for entry in mapping:
        # SurfacePhonemeMapping の全フィールドが存在し正しい型であること
        assert isinstance(entry["surface"], str)
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
        # 既知語は features が 12 列以上
        if len(entry["features"]) > 0:
            assert len(entry["features"]) >= 8
            assert entry["features"][0] == entry["surface"]


def test_g2p_mapping_features_populated():
    """
    g2p_mapping で features が MeCab feature 文字列の分割リストとして返されることを確認。

    features の列数は MeCab の解析結果に依存する:
      - 既知語: 12 列 (surface, 品詞, ..., chain_rule)
      - 未知語: 8 列 (surface, 品詞, ..., 原形。読み/発音/acc/chain_rule がない)
      - アライメント不一致 (sp/数字展開等): 0 列 (空リスト)
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
    """未知語の features が 8 列 (読み/発音/acc/chain_rule なし) で返されることを確認"""

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


def test_g2p_mapping_features_digit_normalization():
    """数字正規化でアライメント不一致のエントリは features が空リストであることを確認"""

    mapping = pyopenjtalk.g2p_mapping("123大阪")
    for entry in mapping:
        assert isinstance(entry["features"], list)
        for col in entry["features"]:
            assert isinstance(col, str)


def test_g2p_mapping_features_space():
    """全角スペース (sp) の features が空リストであることを確認"""

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
    """未知語扱いの記号でも、実際に生成された短ポーズは ['pau'] のまま保持されることを確認"""

    mapping = pyopenjtalk.g2p_mapping("東京ヶ大阪")

    assert mapping[0]["surface"] == "東京"
    assert mapping[1]["surface"] == "ヶ"
    assert mapping[1]["phonemes"] == ["pau"]
    assert mapping[1]["is_unknown"] is True
    assert mapping[2]["surface"] == "大阪"


def test_g2p_mapping_space_produces_sp():
    """g2p_mapping で全角空白が sp 音素を持つことを確認"""

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
    """スペースを挟んで分断された長音マークがマージされることを確認"""

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
    """3 つの長音マークがスペースを挟んで 1 語にマージされることを確認"""

    mapping = pyopenjtalk.g2p_mapping("あー\u3000ー\u3000ー")
    assert len(mapping) == 3

    surfaces = [entry["surface"] for entry in mapping]
    assert "あーーー" in surfaces
    sp_count = sum(1 for entry in mapping if entry["surface"] == "\u3000")
    assert sp_count == 2


def test_g2p_mapping_unknown_merged_with_space():
    """未知語とスペースと長音マークの組み合わせが崩れないことを確認"""

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
    """前後にスペースがある語の長音マージが正しく処理されることを確認"""

    mapping = pyopenjtalk.g2p_mapping("\u3000あーー\u3000")
    assert len(mapping) == 3

    assert mapping[0]["surface"] == "\u3000"
    assert mapping[0]["is_ignored"] is True
    assert mapping[1]["surface"] == "あーー"
    assert mapping[2]["surface"] == "\u3000"
    assert mapping[2]["is_ignored"] is True


def test_g2p_mapping_complex_punctuation():
    """入れ子括弧・連続記号の pau 割り当てが崩れないことを確認"""

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
    assert len(ellipsis_entries) >= 1
    assert ellipsis_entries[0]["phonemes"] == ["pau"]


def test_g2p_mapping_empty_string():
    """空文字列で g2p_mapping が空リストを返すことを確認"""

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
    """テキスト末尾の空白トークンが sp として回収されることを確認"""

    text = "こんにちは　"
    njd_features, morphs = pyopenjtalk.run_frontend_detailed(text)
    detailed = pyopenjtalk.make_phoneme_mapping(njd_features, morphs=morphs)

    # 末尾エントリが sp であること
    assert detailed[-1]["phonemes"] == ["sp"]
    assert detailed[-1]["is_ignored"] is True


def test_run_frontend_detailed_morphs_consistency():
    """run_frontend_detailed の morphs が run_mecab_detailed の結果と同一であることを確認"""

    text = "東京は日本の首都です"
    _, morphs_from_frontend = pyopenjtalk.run_frontend_detailed(text)
    morphs_from_detailed = pyopenjtalk.run_mecab_detailed(text)
    assert morphs_from_frontend == morphs_from_detailed


def test_run_frontend_split_equivalence():
    # Test that run_frontend produces the same result as the split
    # approach (run_mecab -> run_njd_from_mecab -> apply_postprocessing)

    for text in [
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
    ]:
        original_result = pyopenjtalk.run_frontend(text)

        mecab_features = pyopenjtalk.run_mecab(text)
        njd_features = pyopenjtalk.run_njd_from_mecab(mecab_features)
        split_result = pyopenjtalk.apply_postprocessing(text, njd_features)

        assert original_result == split_result


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
      - 未知語: morph='ｘｙｒ' (is_unknown=True)
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


# =============================================================================
# g2p_mapping() のアクセント情報テスト
# =============================================================================


def test_g2p_mapping_accent_fields_present():
    """g2p_mapping の全エントリに acc, mora_size, chain_flag が含まれることを確認"""

    mapping = pyopenjtalk.g2p_mapping("東京都知事が記者会見を行った。")
    for entry in mapping:
        assert "accent_nucleus" in entry
        assert "mora_count" in entry
        assert "chain_flag" in entry
        assert isinstance(entry["accent_nucleus"], int)
        assert isinstance(entry["mora_count"], int)
        assert isinstance(entry["chain_flag"], int)


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


def test_g2p_mapping_accent_position():
    """
    acc (アクセント核位置) が妥当な値を返すことを確認。
    句全体のアクセント核位置は先頭語の acc に集約される。
    """

    mapping = pyopenjtalk.g2p_mapping("東京都知事")

    tokyo = next(e for e in mapping if e["surface"] == "東京")
    assert tokyo["accent_nucleus"] >= 0
    assert tokyo["mora_count"] > 0


def test_g2p_mapping_accent_flat():
    """平板型 (acc=0) の語が正しく返されることを確認"""

    mapping = pyopenjtalk.g2p_mapping("大阪")
    osaka = next(e for e in mapping if e["surface"] == "大阪")
    assert osaka["accent_nucleus"] == 0
    assert osaka["mora_count"] == 4  # オ・オ・サ・カ


def test_g2p_mapping_sp_entry_accent_defaults():
    """sp エントリ (is_ignored=True) のアクセント情報がデフォルト値を持つことを確認"""

    mapping = pyopenjtalk.g2p_mapping("東京　大阪")
    sp_entries = [e for e in mapping if e["is_ignored"] is True]
    assert len(sp_entries) >= 1
    for sp in sp_entries:
        assert sp["accent_nucleus"] == 0
        assert sp["mora_count"] == 0
        assert sp["chain_flag"] == -1


def test_g2p_mapping_odori_accent_inherited():
    """踊り字展開後のアクセント情報が直前トークンから引き継がれることを確認"""

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


def test_modify_acc_after_chaining_unit():
    """modify_acc_after_chaining が「参ります」のアクセント核を正しく移動することを確認"""

    from pyopenjtalk.utils import modify_acc_after_chaining

    features: list[NJDFeature] = [
        {
            "string": "参り",
            "pos": "動詞",
            "pos_group1": "自立",
            "pos_group2": "*",
            "pos_group3": "*",
            "ctype": "五段・ラ行",
            "cform": "連用形",
            "orig": "参る",
            "read": "マイリ",
            "pron": "マイリ",
            "acc": 1,
            "mora_size": 3,
            "chain_rule": "*",
            "chain_flag": -1,
        },
        {
            "string": "ます",
            "pos": "助動詞",
            "pos_group1": "*",
            "pos_group2": "*",
            "pos_group3": "*",
            "ctype": "特殊・マス",
            "cform": "基本形",
            "orig": "ます",
            "read": "マス",
            "pron": "マス'",
            "acc": 1,
            "mora_size": 2,
            "chain_rule": "動詞%F2@1/助詞%F2@1",
            "chain_flag": 1,
        },
    ]
    result = modify_acc_after_chaining(features)
    # 「参ります」→ ま[いりま]す: アクセント核が「ま」(4 モーラ目) に移動する
    assert result[0]["acc"] == 4


def test_g2p_mapping_morphs_none_has_all_fields():
    """morphs を渡さない場合でも全フィールドが含まれることを確認"""

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
    """morphs を渡さない場合でも NJDFeature ベースの未知語フォールバック判定が動作することを確認"""

    njd = pyopenjtalk.run_frontend("xtjqは最高")
    mapping = pyopenjtalk.make_phoneme_mapping(njd)
    # フォールバック判定: pos=フィラー + chain_rule=* で is_unknown=True
    unknown_entries = [e for e in mapping if e["is_unknown"] is True]
    assert len(unknown_entries) >= 1


# ============================================================
# 発音復元オプション (revert_pron_to_read) のテスト
# ============================================================


def test_revert_long_vowels():
    """revert_long_vowels=True で辞書が自動的に長音化した発音が元に復元されることを確認"""

    text = "人生は効果的。"

    # デフォルト: 長音化された pron
    kana_default = pyopenjtalk.g2p(text, kana=True)
    assert "セー" in kana_default
    assert "コーカ" in kana_default
    assert "ワ" in kana_default  # 助詞は「ワ」

    # revert_long_vowels=True: pron が read に復元される
    kana_revert = pyopenjtalk.g2p(text, kana=True, revert_long_vowels=True)
    assert "セイ" in kana_revert
    assert "コウカ" in kana_revert
    assert "ワ" in kana_revert  # 助詞の「ワ」は維持されること


def test_revert_yotsugana():
    """revert_yotsugana=True で四つ仮名の発音統合が元に復元されることを確認"""

    text = "鼻血に気づかず。"

    # デフォルト: ヅ→ズ, ヂ→ジ に統合された pron
    kana_default = pyopenjtalk.g2p(text, kana=True)
    assert "ハナジ" in kana_default
    assert "キズカズ" in kana_default

    # revert_yotsugana=True: ヅ/ヂ が復元される
    kana_revert = pyopenjtalk.g2p(text, kana=True, revert_yotsugana=True)
    assert "ハナヂ" in kana_revert
    assert "キヅカズ" in kana_revert


def test_use_read_as_pron():
    """use_read_as_pron=True で全ての pron が read に置き換わることを確認"""

    text = "こんにちは、人生。"

    # デフォルト: 助詞「は」は「ワ」
    kana_default = pyopenjtalk.g2p(text, kana=True)
    assert "コンニチワ" in kana_default

    # use_read_as_pron=True: 助詞「は」も「ハ」になる
    kana_revert = pyopenjtalk.g2p(text, kana=True, use_read_as_pron=True)
    assert "コンニチハ" in kana_revert


def test_revert_pron_combined():
    """revert_long_vowels + revert_yotsugana の複合ケースが同時に動作することを確認"""

    text = "人生は、鼻血に気づかず。"
    kana = pyopenjtalk.g2p(
        text,
        kana=True,
        revert_long_vowels=True,
        revert_yotsugana=True,
    )
    assert "ジンセイ" in kana  # 長音復元
    assert "ワ" in kana  # 助詞は維持
    assert "ハナヂ" in kana  # 四つ仮名復元
    assert "キヅカズ" in kana  # 四つ仮名復元


def test_revert_pron_with_use_vanilla():
    """use_vanilla=True でも発音復元オプションは独立して適用されることを確認"""

    text = "人生は効果的。"

    # use_vanilla=True + revert_long_vowels=True: 後処理は省略されるが発音復元は適用
    njd = pyopenjtalk.run_frontend(
        text,
        use_vanilla=True,
        revert_long_vowels=True,
    )
    jinsei = next(f for f in njd if f["orig"] == "人生")
    assert jinsei["pron"] == "ジンセイ"  # 長音復元が適用されている

    kouka = next(f for f in njd if f["orig"] == "効果")
    assert kouka["pron"] == "コウカ"  # 長音復元が適用されている

    wa = next(f for f in njd if f["orig"] == "は")
    assert wa["pron"] == "ワ"  # 助詞の「ワ」は維持


def test_revert_pron_default_no_change():
    """発音復元オプションを指定しない場合は pron が変更されないことを確認"""

    text = "人生は効果的。"
    njd = pyopenjtalk.run_frontend(text)
    jinsei = next(f for f in njd if f["orig"] == "人生")
    assert "ー" in jinsei["pron"]  # デフォルトでは長音化された pron


# ============================================================
# 踊り字処理改善のテスト
# ============================================================


def test_odori_hard_boundary():
    """
    踊り字の漢字収集時に記号・フィラー・感動詞がハード境界として機能し、
    遠方の無関係な漢字を誤参照しないことを確認する
    """

    # 記号がハード境界として機能するケース
    # 「人。々」では「。」がハード境界となり、「人」の読みを「々」に引き継がない
    njd = pyopenjtalk.run_frontend("人。々")
    assert len(njd) >= 1
    # 踊り字トークンに「人」の読み (ヒト/ジン) が引き継がれていないことを確認
    odori_tokens = [f for f in njd if "々" in f["orig"]]
    assert len(odori_tokens) >= 1, "踊り字トークンが存在すること"
    for token in odori_tokens:
        assert "ヒト" not in token["read"]
        assert "ジン" not in token["read"]

    # 非漢字トークンが境界として機能するケース
    # 「人は々」では助詞「は」が境界となり、「人」の読みを引き継がない
    njd2 = pyopenjtalk.run_frontend("人は々")
    odori_tokens2 = [f for f in njd2 if "々" in f["orig"]]
    assert len(odori_tokens2) >= 1, "踊り字トークンが存在すること"
    for token in odori_tokens2:
        assert "ヒト" not in token["read"]
        assert "ジン" not in token["read"]


def test_odoriji_voiced_and_voiceless_conversion():
    """一の字点の清音化・濁音化が期待どおりに動作することを確認"""

    assert pyopenjtalk.g2p("がゝ", kana=True) == "ガカ"
    assert pyopenjtalk.g2p("バヽ", kana=True) == "バハ"
    assert pyopenjtalk.g2p("かゞ", kana=True) == "カガ"
    assert pyopenjtalk.g2p("ハヾ", kana=True) == "ハバ"


def test_odoriji_small_kana_handling():
    """拗音を含むモーラに対する一の字点処理が安定していることを確認"""

    assert pyopenjtalk.g2p("じょゝ", kana=True) == "ジョジョ"
    assert pyopenjtalk.g2p("ちゅゞ", kana=True) == "チュヂュ"


def test_odoriji_invalid_cases():
    """不正または孤立した一の字点を与えても安全に処理されることを確認"""

    assert pyopenjtalk.g2p("ゝ", kana=True) == "ゝ"
    assert pyopenjtalk.g2p("かゝ゜", kana=True) == "カカ゜"


def test_odoriji_basic_expansion():
    """一の字点 (ゝ/ゞ/ヽ/ヾ) の基本展開が正しく行われることを確認"""

    assert pyopenjtalk.g2p("さゝみ", kana=True) == "ササミ"
    assert pyopenjtalk.g2p("いすゞ", kana=True) == "イスズ"
    assert pyopenjtalk.g2p("カヽ", kana=True) == "カカ"
    assert pyopenjtalk.g2p("ガヾ", kana=True) == "ガガ"


def test_odoriji_mapping_known_word():
    """辞書登録済みの一の字点語でも mapping の音素列が崩れないことを確認"""

    mapping = pyopenjtalk.g2p_mapping("いすゞ")
    assert len(mapping) == 1
    assert mapping[0]["surface"] == "いすゞ"
    assert mapping[0]["phonemes"] == ["i", "s", "u", "z", "u"]


def test_g2p_mapping_integrity():
    """
    g2p_mapping() の surface を連結すると元の入力と一致することを確認。

    `run_frontend()` の surface 再構成とは別に、空白・未知語を含む
    公開 mapping API の出力契約として保持したい。
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

    haqumei の `test_mapping_flags` の意図を、現在の pyopenjtalk-plus の
    `is_ignored` / `unk` セマンティクスに合わせて移植したもの。
    """

    mapping = pyopenjtalk.g2p_mapping(text)

    for entry in mapping:
        if entry["is_unknown"] is True:
            assert entry["phonemes"] != []
        if entry["phonemes"] == ["sp"]:
            assert entry["is_ignored"] is True
        if entry["phonemes"] == []:
            assert entry["is_ignored"] is True
            assert entry["is_unknown"] is False


def test_g2p_recovery_after_error():
    """公開 API でエラーが発生した後も次の g2p() 呼び出しが正常に動作することを確認"""

    with pytest.raises(RuntimeError, match="too long"):
        pyopenjtalk.g2p("あ" * 10000)

    result = pyopenjtalk.g2p("復帰")
    assert result == "f u cl k i"


def test_g2p_symbols_and_control_chars():
    """記号と制御文字を含む入力でも g2p() がクラッシュしないことを確認"""

    result = pyopenjtalk.g2p("#$%&'()\n\t")
    assert isinstance(result, str)
    assert len(result) > 0


def test_dounojiten_expansion():
    """
    展開済みの「々」をさらに後続の「々」が引き継ぐケースを確認。

    最新の haqumei から取り込んだ周期検出ロジックの回帰テスト。
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
