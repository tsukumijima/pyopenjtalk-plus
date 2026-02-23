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
