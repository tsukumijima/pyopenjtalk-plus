"""OpenJTalk 用のユーザー辞書の構築と入力検証を確認する。"""

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

import pyopenjtalk
from pyopenjtalk.types import UserDictionaryEntry


def _run_mecab_dict_index_without_native_crash(user_csv: Path, user_dic: Path) -> None:
    """
    不正なユーザー辞書入力を子プロセスで実行し、ネイティブ異常終了を検査する。

    Args:
        user_csv (Path): 検査するユーザー辞書 CSV
        user_dic (Path): 辞書の出力先
    """

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
                pass
            """
        ),
        str(user_csv),
        str(user_dic),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=30.0,
    )

    # Python 例外は成功終了へ変換済みなので、非0ならネイティブ側の異常終了とみなす
    assert completed.returncode == 0


def test_mecab_dict_index_empty_surface_should_not_segfault(tmp_path: Path) -> None:
    """空の表層を含む CSV で辞書構築がネイティブ異常終了しない。"""

    user_csv = tmp_path / "invalid_user.csv"
    user_dic = tmp_path / "invalid_user.dic"
    user_csv.write_text(",1358,1358,8047,名詞,接尾,一般,*,*,*,－,ノ,ノ,0/1,*\n", encoding="utf-8")

    _run_mecab_dict_index_without_native_crash(user_csv, user_dic)


def test_mecab_dict_index_invalid_dn_mecab_should_raise_file_not_found(tmp_path: Path) -> None:
    """存在しないシステム辞書パスを FileNotFoundError で拒否する。"""

    user_csv = tmp_path / "valid.csv"
    user_dic = tmp_path / "valid.dic"
    user_csv.write_text(
        "ｔｅｓｔ,,,1,名詞,一般,*,*,*,*,ｔｅｓｔ,テスト,テスト,1/3,*\n", encoding="utf-8"
    )

    with pytest.raises(FileNotFoundError):
        pyopenjtalk.mecab_dict_index(
            str(user_csv), str(user_dic), dn_mecab=str(tmp_path / "not-found-dic")
        )


def test_mecab_dict_index_valid_user_dict(tmp_path: Path) -> None:
    """有効な CSV エントリで mecab_dict_index() を実行した場合、辞書が正常にビルドされること。"""
    user_csv = tmp_path / "valid_user.csv"
    user_dic = tmp_path / "valid_user.dic"
    user_csv.write_text(
        "テスト,1348,1348,5000,名詞,固有名詞,一般,*,*,*,テスト,テスト,テスト,1/3,C1\n",
        encoding="utf-8",
    )

    pyopenjtalk.mecab_dict_index(str(user_csv), str(user_dic))

    assert user_dic.exists()


def test_mecab_dict_index_csv_only_commas_should_not_segfault(tmp_path: Path) -> None:
    """カンマのみを含む CSV で mecab_dict_index() を実行した場合、セグフォしないこと。"""
    user_csv = tmp_path / "invalid_user.csv"
    user_dic = tmp_path / "invalid_user.dic"
    user_csv.write_text(",,,,,,,,,,,,,\n", encoding="utf-8")

    _run_mecab_dict_index_without_native_crash(user_csv, user_dic)


def test_mecab_dict_index_random_invalid_input_should_not_segfault(tmp_path: Path) -> None:
    """複数種類の不正 CSV で辞書構築がネイティブ異常終了しない。"""

    random_csv_lines = [
        ",,,,,\n",
        "a,b,c,d,e\n",
        "無効,1,2,3\n",
        "😀,1358,1358,8047,名詞,接尾,一般,*,*,*,－,ノ,ノ,0/1,*\n",
        '"unterminated,1358,1358,8047,名詞,接尾,一般,*,*,*,－,ノ,ノ,0/1,*\n',
    ]
    for index, csv_line in enumerate(random_csv_lines):
        user_csv = tmp_path / f"invalid_user_{index}.csv"
        user_dic = tmp_path / f"invalid_user_{index}.dic"
        user_csv.write_text(csv_line, encoding="utf-8")

        _run_mecab_dict_index_without_native_crash(user_csv, user_dic)


def test_g2p_mapping_user_dict_multi_accent_phrase_keeps_surfaces(tmp_path: Path) -> None:
    """OpenJTalk 用のユーザー辞書の1表層複数アクセント句でも表層列が崩れないことを確認。"""

    # 人名を意図的に2アクセント句へ分ける実運用形式 (orig/read/pron/acc をコロンで連結) を再現する
    user_csv = tmp_path / "multi_accent.csv"
    user_dic = tmp_path / "multi_accent.dic"
    user_csv.write_text(
        "山下清悟,,,1,名詞,固有名詞,人名,一般,*,*,山下:清悟,ヤマシタ:シンゴ,ヤマシタ:シンゴ,2/4:1/3,C1\n",
        encoding="utf-8",
    )

    try:
        pyopenjtalk.mecab_dict_index(str(user_csv), str(user_dic))
        pyopenjtalk.update_global_jtalk_with_user_dict(str(user_dic))

        # 分裂の前後に数字変換を混在させ、morph 消費カーソルが後続へずれないことも確認する
        mapping = pyopenjtalk.g2p_mapping("１２３円を山下清悟さんが払った")

        assert [entry["surface"] for entry in mapping] == [
            "百",
            "二",
            "十",
            "三",
            "円",
            "を",
            "山下",
            "清悟",
            "さん",
            "が",
            "払っ",
            "た",
        ]
    finally:
        pyopenjtalk.unset_user_dict()


@pytest.mark.parametrize(
    ("surface", "reading", "text", "expected"),
    [
        ("方", "ホウ", "先生方", "センセーホウ"),
        ("等", "ホト", "機器等", "キキホト"),
        ("ヴィクトリーヌ", "ビクトリーヌ", "ヴィクトリーヌ", "ビクトリーヌ"),
    ],
)
def test_protected_user_dictionary_reading_survives_postprocessing(
    tmp_path: Path,
    surface: str,
    reading: str,
    text: str,
    expected: str,
) -> None:
    """読み保護を有効にしたユーザー辞書のエントリ（「方」「等」「ヴィクトリーヌ」など）の読みが、Python 側の各種後処理（文脈読み補正や外来語補正など）によって上書きされず、登録通りの読みで維持されることを確認する。"""

    user_csv = tmp_path / "protected.csv"
    user_dic = tmp_path / "protected.dic"
    user_csv.write_text(
        f"{surface},1348,1348,1,名詞,接尾,一般,*,*,*,{surface},{reading},{reading},0/2,C1\n",
        encoding="utf-8",
    )

    try:
        pyopenjtalk.mecab_dict_index(str(user_csv), str(user_dic))
        pyopenjtalk.update_global_jtalk_with_user_dict(
            [
                UserDictionaryEntry(
                    dic_path=str(user_dic),
                    is_reading_protected=True,
                )
            ]
        )

        assert pyopenjtalk.g2p(text, kana=True) == expected
    finally:
        pyopenjtalk.unset_user_dict()


def test_protected_repeated_placeholder_reading_survives_njd_rules(tmp_path: Path) -> None:
    """読み保護を有効にしたユーザー辞書に登録された連続伏字（「〇〇」など）の読みが、NJD 前処理による「マル」への読み補正で上書きされず、ユーザー辞書の登録値（「カク」）のまま維持されることを確認する。"""

    user_csv = tmp_path / "protected_placeholder.csv"
    user_dic = tmp_path / "protected_placeholder.dic"
    user_csv.write_text(
        "〇,1345,1345,1,名詞,一般,*,*,*,*,〇,カク,カク,1/2,C1\n",
        encoding="utf-8",
    )

    try:
        pyopenjtalk.mecab_dict_index(str(user_csv), str(user_dic))
        pyopenjtalk.update_global_jtalk_with_user_dict(
            [
                UserDictionaryEntry(
                    dic_path=str(user_dic),
                    is_reading_protected=True,
                )
            ]
        )

        features = pyopenjtalk.run_frontend("〇〇町")
        placeholders = [feature for feature in features if feature["string"] == "〇"]

        assert pyopenjtalk.g2p("〇〇町", kana=True) == "カクカクマチ"
        assert len(placeholders) == 2
        assert all(feature.get("is_reading_protected") is True for feature in placeholders)
        assert all(feature["read"] == "カク" for feature in placeholders)
        assert all(feature["pron"] == "カク" for feature in placeholders)
        assert all(feature["mora_size"] == 2 for feature in placeholders)
    finally:
        pyopenjtalk.unset_user_dict()


@pytest.mark.parametrize(
    (
        "text",
        "expected",
        "expected_seven_pronunciations",
        "expected_reading_protections",
    ),
    [
        ("７、七", "ナナ、ナナナ", ["ナナ", "ナナナ"], [False, True]),
        ("７ 七", "ナナナナナ", ["ナナ", "ナナナ"], [False, True]),
        ("１０日七", "トーカナナナ", ["ナナナ"], [True]),
    ],
)
def test_protected_user_dictionary_reading_uses_character_position_after_digit(
    tmp_path: Path,
    text: str,
    expected: str,
    expected_seven_pronunciations: list[str],
    expected_reading_protections: list[bool],
) -> None:
    """空白や読点を挟んで同一の表層（「七」）が現れる場合でも、文字位置（文字スパン）に基づいて判定することで、数字変換された形態素と保護対象のユーザー辞書エントリが正確に区別され、ユーザー辞書側の読みだけが保護されることを確認する。"""

    user_csv = tmp_path / "protected_seven.csv"
    user_dic = tmp_path / "protected_seven.dic"
    user_csv.write_text(
        "七,1345,1345,1,名詞,一般,*,*,*,*,七,ナナナ,ナナナ,0/3,C2\n",
        encoding="utf-8",
    )

    try:
        pyopenjtalk.mecab_dict_index(str(user_csv), str(user_dic))
        pyopenjtalk.update_global_jtalk_with_user_dict(
            [
                UserDictionaryEntry(
                    dic_path=str(user_dic),
                    is_reading_protected=True,
                )
            ]
        )

        # 全角数字の変換後も、区切りの後ろにあるユーザー辞書語だけを保護する
        features = pyopenjtalk.run_frontend(text)
        seven_features = [feature for feature in features if feature["string"] == "七"]

        assert pyopenjtalk.g2p(text, kana=True) == expected
        assert [feature["pron"] for feature in seven_features] == expected_seven_pronunciations
        assert [feature.get("is_reading_protected", False) for feature in seven_features] == [
            *expected_reading_protections,
        ]
    finally:
        pyopenjtalk.unset_user_dict()


def test_protected_multi_accent_user_dictionary_reading_survives_postprocessing(
    tmp_path: Path,
) -> None:
    """複数アクセント句に分割されるユーザー辞書のエントリ（「先生:方」など）において、後処理が適用された後でも各アクセント句の登録された読みが正しく維持されることを確認する。"""

    user_csv = tmp_path / "protected_multi_accent.csv"
    user_dic = tmp_path / "protected_multi_accent.dic"
    user_csv.write_text(
        "先生方,,,1,名詞,接尾,一般,*,*,*,先生:方,センセイ:ホウ,センセイ:ホウ,0/4:0/2,C1\n",
        encoding="utf-8",
    )

    try:
        pyopenjtalk.mecab_dict_index(str(user_csv), str(user_dic))
        pyopenjtalk.update_global_jtalk_with_user_dict(
            [
                UserDictionaryEntry(
                    dic_path=str(user_dic),
                    is_reading_protected=True,
                )
            ]
        )

        assert pyopenjtalk.g2p("先生方", kana=True) == "センセイホウ"
    finally:
        pyopenjtalk.unset_user_dict()


def test_protected_suffix_toki_keeps_registered_attributes(tmp_path: Path) -> None:
    """読み保護を有効にした接尾辞の「時」（読み: 「ジ」）を登録した場合、前接語がない文脈であっても「トキ」への文脈読み補正がスキップされ、登録された読み・品詞・アクセントがそのまま維持されることを確認する。"""

    user_csv = tmp_path / "protected_toki.csv"
    user_dic = tmp_path / "protected_toki.dic"
    user_csv.write_text(
        "時,1348,1348,1,名詞,接尾,一般,*,*,*,時,ジ,ジ,0/1,C1\n",
        encoding="utf-8",
    )

    try:
        pyopenjtalk.mecab_dict_index(str(user_csv), str(user_dic))
        pyopenjtalk.update_global_jtalk_with_user_dict(
            [
                UserDictionaryEntry(
                    dic_path=str(user_dic),
                    is_reading_protected=True,
                )
            ]
        )

        # 読み保護が復元しない品詞・アクセントまで、「トキ」補正の対象から外れて登録値のまま残る
        features = pyopenjtalk.run_frontend("ぐるぐる時")
        toki = next(feature for feature in features if feature["string"] == "時")

        assert toki["read"] == "ジ"
        assert toki["pron"] == "ジ"
        assert toki["pos_group1"] == "接尾"
        assert toki["acc"] == 0
        assert toki["mora_size"] == 1
    finally:
        pyopenjtalk.unset_user_dict()


def test_protected_user_dictionary_keeps_unvoiced_vowel(tmp_path: Path) -> None:
    """読み保護を有効にしたユーザー辞書の単語であっても、NJD の無声化処理で推定された母音の無声化（「草」の [U] など）が失われずに発音ラベルへ反映されることを確認する。"""

    user_csv = tmp_path / "protected_unvoiced.csv"
    user_dic = tmp_path / "protected_unvoiced.dic"
    # 登録読みは辞書 CSV の慣行どおり無声化記号を含まない（「クサ」の u は k と s に挟まれ無声化する）
    user_csv.write_text(
        "草,1345,1345,1,名詞,一般,*,*,*,*,草,クサ,クサ,2/2,C3\n",
        encoding="utf-8",
    )

    try:
        pyopenjtalk.mecab_dict_index(str(user_csv), str(user_dic))
        pyopenjtalk.update_global_jtalk_with_user_dict(
            [
                UserDictionaryEntry(
                    dic_path=str(user_dic),
                    is_reading_protected=True,
                )
            ]
        )

        assert pyopenjtalk.g2p("草") == "k U s a"
    finally:
        pyopenjtalk.unset_user_dict()


def test_protected_user_dictionary_accent_survives_accent_postprocessing(tmp_path: Path) -> None:
    """読み保護を有効にしたユーザー辞書のエントリにおいて、長音上にアクセント核が指定されている場合（「ローン」の2モーラ目など）でも、後処理のアクセント補正（核の後退処理）によって書き換えられず、登録されたアクセント核の位置が維持されることを確認する。"""

    user_csv = tmp_path / "protected_accent.csv"
    user_dic = tmp_path / "protected_accent.dic"
    # 核を長音上 (2モーラ目) に置いた登録は、保護がなければ retreat_acc_nuc に1モーラ前へ引き戻される
    user_csv.write_text(
        "ローン,1345,1345,1,名詞,一般,*,*,*,*,ローン,ローン,ローン,2/3,C1\n",
        encoding="utf-8",
    )

    try:
        pyopenjtalk.mecab_dict_index(str(user_csv), str(user_dic))
        pyopenjtalk.update_global_jtalk_with_user_dict(
            [
                UserDictionaryEntry(
                    dic_path=str(user_dic),
                    is_reading_protected=True,
                )
            ]
        )

        features = pyopenjtalk.run_frontend("ローンを組む")
        loan = next(feature for feature in features if feature["string"] == "ローン")

        assert loan["acc"] == 2
    finally:
        pyopenjtalk.unset_user_dict()


def test_unknown_katakana_does_not_override_user_dictionary(tmp_path: Path) -> None:
    """未知カタカナ語として判定されうる単語であっても、ユーザー辞書に登録されていれば生起コストに関わらずユーザー辞書のエントリが優先され、登録された読みとアクセントが外来語の規則で上書きされずに反映されることを確認する。"""

    user_csv = tmp_path / "katakana.csv"
    user_dic = tmp_path / "katakana.dic"
    user_csv.write_text(
        "ヌメロワール,1345,1345,10000,名詞,一般,*,*,*,*,ヌメロワール,ユーザーワール,ユーザーワール,3/7,C1\n",
        encoding="utf-8",
    )

    try:
        pyopenjtalk.mecab_dict_index(str(user_csv), str(user_dic))
        pyopenjtalk.update_global_jtalk_with_user_dict(str(user_dic))

        assert pyopenjtalk.g2p("ヌメロワール", kana=True) == "ユーザーワール"
        # 外来語の規則なら後ろから3モーラ目の「ワ」(5) に核が来るので、登録した「ザ」(3) のままかで上書きの有無が分かる
        features = pyopenjtalk.run_frontend("ヌメロワール")
        assert [(feature["pron"], feature["acc"]) for feature in features] == [
            ("ユーザーワール", 3)
        ]
    finally:
        pyopenjtalk.unset_user_dict()


def test_itaiji_normalization_keeps_user_dictionary_word(tmp_path: Path) -> None:
    """同じ異体字がユーザー辞書の語と未知語の両方に現れても、置き換えるのは未知語の字だけで、ユーザー辞書の語の読みは保たれることを確認する。"""

    user_csv = tmp_path / "itaiji.csv"
    user_dic = tmp_path / "itaiji.dic"
    user_csv.write_text(
        "𠮷野家,1345,1345,-5000,名詞,一般,*,*,*,*,𠮷野家,トクベツ,トクベツ,0/4,C1\n",
        encoding="utf-8",
    )

    try:
        pyopenjtalk.mecab_dict_index(str(user_csv), str(user_dic))
        pyopenjtalk.update_global_jtalk_with_user_dict(str(user_dic))

        assert pyopenjtalk.g2p("𠮷野家、𠮷", kana=True) == "トクベツ、ヨシ"
    finally:
        pyopenjtalk.unset_user_dict()


def test_openjtalk_rejects_mismatched_user_dictionary_protection_count() -> None:
    """OpenJTalk 用のユーザー辞書数と読み保護フラグ数の不一致を初期化前に拒否する。"""

    with pytest.raises(ValueError, match="same number of entries"):
        pyopenjtalk.OpenJTalk(
            userdic=b"first.dic,second.dic",
            userdic_reading_protection=[False],
        )


def test_openjtalk_rejects_non_boolean_user_dictionary_protection() -> None:
    """読み保護フラグへ bool 以外を受け入れない。"""

    with pytest.raises(TypeError, match="entries must be bool"):
        invalid_protection: Any = [1]
        pyopenjtalk.OpenJTalk(
            userdic=b"user.dic",
            userdic_reading_protection=invalid_protection,
        )


def test_high_level_user_dictionary_rejects_mixed_entry_types(tmp_path: Path) -> None:
    """従来文字列と UserDictionaryEntry を同じリストへ混在させない。"""

    with pytest.raises(TypeError, match="must not mix"):
        mixed_paths: Any = [
            str(tmp_path / "plain.dic"),
            {
                "dic_path": str(tmp_path / "protected.dic"),
                "is_reading_protected": True,
            },
        ]
        pyopenjtalk.update_global_jtalk_with_user_dict(mixed_paths)


def test_high_level_user_dictionary_rejects_invalid_list_entry_type(tmp_path: Path) -> None:
    """文字列と辞書以外のリスト要素を混在エラーと区別する。"""

    with pytest.raises(TypeError, match="only strings or UserDictionaryEntry values"):
        invalid_paths: Any = [str(tmp_path / "plain.dic"), 1]
        pyopenjtalk.update_global_jtalk_with_user_dict(invalid_paths)


@pytest.mark.parametrize(
    "paths",
    [
        ["first,second.dic"],
        [{"dic_path": "first,second.dic", "is_reading_protected": True}],
    ],
)
def test_high_level_user_dictionary_rejects_comma_in_list_path(
    paths: list[str] | list[UserDictionaryEntry],
) -> None:
    """リスト内のカンマを辞書区切りとして解釈させない。"""

    with pytest.raises(ValueError, match="must not contain commas"):
        pyopenjtalk.update_global_jtalk_with_user_dict(paths)


def test_high_level_user_dictionary_rejects_non_string_dictionary_path() -> None:
    """UserDictionaryEntry の辞書パスに文字列以外を受け入れない。"""

    with pytest.raises(TypeError, match="dic_path must be a non-empty string"):
        invalid_entry_paths: Any = [
            {
                "dic_path": 1,
                "is_reading_protected": False,
            }
        ]
        pyopenjtalk.update_global_jtalk_with_user_dict(invalid_entry_paths)
