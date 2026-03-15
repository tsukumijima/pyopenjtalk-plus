import copy
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import pyopenjtalk
from pyopenjtalk import NJDFeature


def _print_results(njd_features: list[NJDFeature], labels: list[str]):
    for f in njd_features:
        s, p = f["string"], f["pron"]
        print(s, p)

    for label in labels:
        print(label)


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

    pyopenjtalk.mecab_dict_index(f.name, user_dic)
    pyopenjtalk.update_global_jtalk_with_user_dict(user_dic)

    for text, expected in [
        ("nnmn", "n a n a m i N"),
        ("GNU", "g u n u u"),
    ]:
        p = pyopenjtalk.g2p(text)
        assert p == expected


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
        assert "feature" in morph
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
        # feature は "surface," で始まる
        assert morph["feature"].startswith(morph["surface"] + ",")


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

    # detailed から ignored を除いた feature が normal の結果と一致する
    detailed_features = [
        morph["feature"] for morph in detailed_morphs if morph["is_ignored"] is False
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
        assert "word" in entry
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
    """文末の句読点でもポーズ音素情報が保持されることを確認"""

    njd_features = pyopenjtalk.run_frontend("あ。")
    mapping = pyopenjtalk.make_phoneme_mapping(njd_features)

    assert mapping[0]["word"] == "あ"
    assert mapping[0]["phonemes"] == ["a"]
    assert mapping[1]["word"] == "。"
    assert mapping[1]["phonemes"] == ["pau"]


def test_make_phoneme_mapping_boundary_punctuation_start():
    """文頭の句読点でもポーズ音素情報が保持されることを確認"""

    njd_features = pyopenjtalk.run_frontend("。あ")
    mapping = pyopenjtalk.make_phoneme_mapping(njd_features)

    assert mapping[0]["word"] == "。"
    assert mapping[0]["phonemes"] == ["pau"]
    assert mapping[1]["word"] == "あ"
    assert mapping[1]["phonemes"] == ["a"]


def test_make_phoneme_mapping_pause_like_symbols():
    """読点へ正規化される各種記号もポーズ音素 ['pau'] として保持されることを確認"""

    njd_features = pyopenjtalk.run_frontend("（テスト・ケース）")
    mapping = pyopenjtalk.make_phoneme_mapping(njd_features)

    assert mapping[0]["word"] == "（"
    assert mapping[0]["phonemes"] == ["pau"]
    assert mapping[1]["word"] == "テスト"
    assert mapping[1]["phonemes"] == ["t", "e", "s", "U", "t", "o"]
    assert mapping[2]["word"] == "・"
    assert mapping[2]["phonemes"] == ["pau"]
    assert mapping[3]["word"] == "ケース"
    assert mapping[3]["phonemes"] == ["k", "e", "e", "s", "u"]
    assert mapping[4]["word"] == "）"
    assert mapping[4]["phonemes"] == ["pau"]


def test_make_phoneme_mapping_phoneme_consistency():
    """make_phoneme_mapping の音素列を結合すると make_label 由来の音素列と同じ結果になることを確認

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


def test_make_phoneme_mapping_digit():
    """数字入力 (NJD でノード数が変わるケース) でクラッシュしないことを確認"""

    njd_features = pyopenjtalk.run_frontend("123")
    mapping = pyopenjtalk.make_phoneme_mapping(njd_features)
    assert len(mapping) >= 1


def test_make_phoneme_mapping_empty_features():
    """空の NJDFeature リストで空リストが返されることを確認"""

    mapping = pyopenjtalk.make_phoneme_mapping([])
    assert mapping == []


def test_make_phoneme_mapping_word_correspondence():
    """make_phoneme_mapping の word フィールドが NJDFeature の string と 1:1 対応していることを確認。

    Note: 長音吸収マージが発生するテキストでは len(mapping) < len(njd_features) となるため、
    このテストでは長音吸収が発生しない入力のみを使用している。
    """

    text = "今日も良い天気ですね"
    njd_features = pyopenjtalk.run_frontend(text)
    mapping = pyopenjtalk.make_phoneme_mapping(njd_features)

    assert len(mapping) == len(njd_features)
    for entry, feat in zip(mapping, njd_features):
        assert entry["word"] == feat["string"]


def test_make_phoneme_mapping_multiple_sentences():
    """複数の文で make_phoneme_mapping が正しく動作することを確認。

    Note: 長音吸収マージが発生するテキストでは len(mapping) < len(njd_features) となるため、
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
    """Cython レベルの make_phoneme_mapping() で長音吸収マージが正しく動作することを確認。

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
        assert len(entry["phonemes"]) > 0, f"Empty phonemes for word: {entry['word']}"

    # 'れよう' がマージ結果として存在すること
    words = [entry["word"] for entry in mapping]
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
        assert "word" in entry
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
    """数字正規化が先行するケースで is_unknown が正しく伝播することを確認。

    "7xyz" は MeCab 上では "７"(既知) + "ｘｙｚ"(未知) だが、
    NJD 側で "７" → "七" に正規化されるため surface 不一致が発生する。
    バランスベースのアライメントにより、後続の未知語にも is_unknown が正しく伝播する。
    """

    text = "7xyz"
    njd_features, morphs = pyopenjtalk.run_frontend_detailed(text)
    detailed = pyopenjtalk.make_phoneme_mapping(njd_features, morphs=morphs)

    assert len(detailed) >= 2
    # 数字正規化されたエントリが存在すること
    assert any(entry["word"] == "七" for entry in detailed)
    # 未知語トークンの is_unknown が正しく伝播していること
    xyz_entry = next(entry for entry in detailed if entry["word"] == "ｘｙｚ")
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
        assert entry_without["word"] == entry_with["word"]
        assert entry_without["phonemes"] == entry_with["phonemes"]


def test_make_phoneme_mapping_with_morphs_digit():
    """数字入力 (NJD でノード数が変わるケース) でクラッシュしないことを確認"""

    text = "123"
    njd_features, morphs = pyopenjtalk.run_frontend_detailed(text)
    detailed = pyopenjtalk.make_phoneme_mapping(njd_features, morphs=morphs)

    assert len(detailed) >= 1


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
        assert "feature" in morph
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
    """g2p_mapping の基本動作を確認"""

    mapping = pyopenjtalk.g2p_mapping("こんにちは")
    assert len(mapping) >= 1
    for entry in mapping:
        assert "word" in entry
        assert "phonemes" in entry
        assert "is_unknown" in entry
        assert "is_ignored" in entry
        assert len(entry["phonemes"]) > 0


def test_g2p_mapping_unknown_word():
    """g2p_mapping で未知語が is_unknown=True を持つことを確認。

    unk 音素への置換は、未知語かつ音素が空 or ['pau'] の場合のみ発生する。
    OpenJTalk が実際に音素を生成できた未知語は、is_unknown=True のまま
    生成された音素がそのまま保持される。
    """

    mapping = pyopenjtalk.g2p_mapping("xtjq")
    unknown_entries = [e for e in mapping if e["is_unknown"] is True]
    assert len(unknown_entries) >= 1
    # 未知語は必ず何らかの音素を持つ (空にはならない)
    for entry in unknown_entries:
        assert len(entry["phonemes"]) > 0


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
    words = [entry["word"] for entry in mapping]
    assert "れよう" in words, f"Expected 'れよう' in words, got: {words}"
    # 'う' が独立エントリとして残っていないこと
    assert "う" not in words, f"'う' should be merged into 'れよう', got: {words}"

    # 'れよう' の音素が正しいこと (長音を含む)
    reyou_entry = next(entry for entry in mapping if entry["word"] == "れよう")
    assert reyou_entry["phonemes"] == ["r", "e", "y", "o", "o"]

    # 'と' が正しい音素を持つこと (長音吸収前はオフバイワンで崩れていた)
    to_entry = next(entry for entry in mapping if entry["word"] == "と")
    assert to_entry["phonemes"] == ["t", "o"]


def test_g2p_mapping_empty_string():
    """空文字列で g2p_mapping が空リストを返すことを確認"""

    mapping = pyopenjtalk.g2p_mapping("")
    assert mapping == []


def test_g2p_mapping_all_ignored():
    """全角スペースのみの入力で全エントリが is_ignored=True, phonemes=['sp'] となることを確認。

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

    words = [entry["word"] for entry in mapping]
    # 踊り字展開後の NJD feature 構成
    assert "学生" in words
    assert "生活" in words
    assert "は" in words
    assert "楽しい" in words

    # 全エントリの phonemes が空でないこと
    for entry in mapping:
        assert len(entry["phonemes"]) > 0, f"Empty phonemes for word: {entry['word']}"

    # 生活の音素が正しいこと
    seikatsu = next(entry for entry in mapping if entry["word"] == "生活")
    assert seikatsu["phonemes"] == ["s", "e", "e", "k", "a", "ts", "u"]

    # 楽しい が正しくマッピングされていること (re-sync が必要)
    tanoshii = next(entry for entry in mapping if entry["word"] == "楽しい")
    assert len(tanoshii["phonemes"]) > 0


def test_g2p_mapping_odori_with_space():
    """踊り字展開 + 全角スペース併用で sp エントリが正しく出力されることを確認。"""

    mapping = pyopenjtalk.g2p_mapping("学生々活　大阪")

    words = [entry["word"] for entry in mapping]
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
    xyz_entry = next(entry for entry in mapping if entry["word"] == "ｘｙｚ")
    assert xyz_entry["is_unknown"] is True, (
        f"Expected ｘｙｚ to be is_unknown=True, got {xyz_entry}"
    )

    # 大阪 が正しくマッピングされていること
    osaka_entry = next(entry for entry in mapping if entry["word"] == "大阪")
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
    xyz_entry = next(entry for entry in mapping if entry["word"] == "ｘｙｚ")
    assert xyz_entry["is_unknown"] is True, (
        f"Expected ｘｙｚ to be is_unknown=True, got {xyz_entry}"
    )

    # 大阪 が正しくマッピングされていること
    osaka_entry = next(entry for entry in mapping if entry["word"] == "大阪")
    assert osaka_entry["is_unknown"] is False

    # 全エントリの phonemes が空でないこと
    for entry in mapping:
        assert len(entry["phonemes"]) > 0, f"Empty phonemes for word: {entry['word']}"


def test_g2p_mapping_odori_digit_unknown_duplicate_word_with_space():
    """
    踊り字展開 + 全角スペース + 数字正規化 + 重複語。
    "学生々活　7xyz七大阪" は全角スペース (is_ignored) が踊り字展開と数字正規化の間に挟まるケース。
    """

    mapping = pyopenjtalk.g2p_mapping("学生々活　7xyz七大阪")

    # ｘｙｚ の is_unknown が正しく伝播していること
    xyz_entry = next(entry for entry in mapping if entry["word"] == "ｘｙｚ")
    assert xyz_entry["is_unknown"] is True, (
        f"Expected ｘｙｚ to be is_unknown=True, got {xyz_entry}"
    )

    # 全角スペースが sp として出力されること
    sp_entries = [entry for entry in mapping if entry["phonemes"] == ["sp"]]
    assert len(sp_entries) >= 1

    # 大阪 が正しくマッピングされていること
    osaka_entry = next(entry for entry in mapping if entry["word"] == "大阪")
    assert osaka_entry["is_unknown"] is False
