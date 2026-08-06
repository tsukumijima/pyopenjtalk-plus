"""OpenJTalk 用ユーザー辞書の構築・入力検証を検証する。"""

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, cast

import pytest

import pyopenjtalk
from pyopenjtalk.types import UserDictionaryEntry


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


def test_g2p_mapping_user_dict_multi_accent_phrase_keeps_surfaces(tmp_path: Path):
    """ユーザー辞書の1表層複数アクセント句エントリでも表層列が崩れないことを確認。"""

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


def test_openjtalk_rejects_mismatched_user_dictionary_protection_count() -> None:
    """ユーザー辞書数と読み保護フラグ数の不一致を初期化前に拒否する。"""

    with pytest.raises(ValueError, match="same number of entries"):
        pyopenjtalk.OpenJTalk(
            userdic=b"first.dic,second.dic",
            userdic_reading_protection=[False],
        )


def test_openjtalk_rejects_non_boolean_user_dictionary_protection() -> None:
    """読み保護フラグへ bool 以外を受け入れない。"""

    with pytest.raises(TypeError, match="entries must be bool"):
        pyopenjtalk.OpenJTalk(
            userdic=b"user.dic",
            userdic_reading_protection=cast(Any, [1]),
        )


def test_high_level_user_dictionary_rejects_mixed_entry_types(tmp_path: Path) -> None:
    """従来文字列と UserDictionaryEntry を同じリストへ混在させない。"""

    with pytest.raises(TypeError, match="must not mix"):
        pyopenjtalk.update_global_jtalk_with_user_dict(
            cast(
                Any,
                [
                    str(tmp_path / "plain.dic"),
                    {
                        "dic_path": str(tmp_path / "protected.dic"),
                        "is_reading_protected": True,
                    },
                ],
            )
        )


def test_high_level_user_dictionary_rejects_invalid_list_entry_type(tmp_path: Path) -> None:
    """文字列と辞書以外のリスト要素を混在エラーと区別する。"""

    with pytest.raises(TypeError, match="only strings or UserDictionaryEntry values"):
        pyopenjtalk.update_global_jtalk_with_user_dict(cast(Any, [str(tmp_path / "plain.dic"), 1]))


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
        pyopenjtalk.update_global_jtalk_with_user_dict(
            cast(
                Any,
                [
                    {
                        "dic_path": 1,
                        "is_reading_protected": False,
                    }
                ],
            )
        )
