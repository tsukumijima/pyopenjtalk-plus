#!/usr/bin/env python3
"""
辞書エントリの到達性 (自分自身が最良経路に出てくるか) を監査し、必要なコスト調整量を計算する。

unidic-csj.csv は OpenJTalk のメンテナが過去に UniDic-CSJ 2.2.0 から変換した際、おそらく機械的にコストを変換した関係で
コスト分布が naist-jdic と揃っておらず、エントリは存在するのにコスト負けして一度も選ばれない「死にエントリ」を含む。
（実例: 「未病（ミビョー、コスト: 11340）」が「未+病」の分割経路に負け、ミヤマイと誤読される）

このスクリプトは各エントリについて表層単独または指定した代表文を入力し、以下の通り動作する。
1. 最良経路が自分自身 (1形態素で表層・文脈 ID・単語コストが一致) を含むなら到達可能と判定
2. 負けている場合は n-best から自分を通る経路を探し、勝者経路との総コスト差 delta を実測
3. 単語コストを (delta + margin) だけ下げた推奨値を出力

Viterbi の経路総コストは単語コストに線形なので、必要な調整量は二分探索や辞書の再ビルドなしに
1回の n-best 解析から閉形式で決まる。辞書のビルドが必要になるのは推奨値を適用した後の1回だけである。

制限: 指定した入力での到達性は、未指定の文脈でも常に勝つことを保証しない。
推奨値の本適用前には、対象語を含む代表文と、対象語を含まない対照文の回帰検査を必ず行うこと。

解釈上の注意: dead 判定は「単独入力で自分が勝たない」ことしか意味しない。unidic-csj.csv のような
1表層1読みの一般語では dead = 一度も選ばれ得ない真の死にエントリだが、
heteronyms.csv のような同形異音語辞書では、同表層の劣後読み (例: 風=フー) が既定読み (風=カゼ) に
単独入力で負けるのは設計どおりの正常な状態である。
同形異音語辞書で問題になるのは、同一表層の全読みが表層グループ外の分割経路に負ける場合だけである。

Usage:
    # 特定の表層だけを検査する (デバッグ・単発症例向け)
    uv run python scripts/audit_dictionary_entry_reachability.py \
        --csv pyopenjtalk/dictionary/unidic-csj.csv --surface 未病

    # CSV 全行を検査し、死にエントリだけを TSV で保存する
    uv run python scripts/audit_dictionary_entry_reachability.py \
        --csv pyopenjtalk/dictionary/unidic-csj.csv --dead-only --output dead_entries.tsv

    # 代表文すべてで勝つ最大コストを求める
    uv run python scripts/audit_dictionary_entry_reachability.py \
        --csv pyopenjtalk/dictionary/naist-jdic.csv --surface 未病 \
        --context '{surface}について説明します。' --context '以前から{surface}でした。'
"""

import argparse
import csv
import sys
from pathlib import Path

from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyopenjtalk
from pyopenjtalk.types import MeCabNBestPath


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DICTIONARY_DIR = REPO_ROOT / "pyopenjtalk" / "dictionary"


class DictionaryEntry:
    """辞書 CSV 1行分のエントリ (到達性判定に使う列だけを保持する)。"""

    def __init__(self, row: list[str], line_number: int) -> None:
        """
        CSV 行からエントリを構築する。

        Args:
            row (list[str]): 15列の辞書 CSV 行
            line_number (int): CSV 内の行番号 (1始まり)
        """

        self.surface = row[0]
        self.left_id = int(row[1])
        self.right_id = int(row[2])
        self.word_cost = int(row[3])
        self.part_of_speech = row[4]
        self.pronunciation = row[12] if len(row) > 12 else ""
        self.line_number = line_number


class ReachabilityAuditor:
    """n-best 解析で、入力文における辞書エントリの到達性とコスト調整量を測る。"""

    def __init__(self, jtalk: "pyopenjtalk.OpenJTalk", max_paths: int, margin: int) -> None:
        """
        監査器を構築する。

        Args:
            jtalk (pyopenjtalk.OpenJTalk): 検査対象の辞書だけを読み込んだフロントエンド
            max_paths (int): n-best の探索深さ (1〜512)
            margin (int): 推奨コストへ上乗せする勝ち幅
        """

        self.jtalk = jtalk
        self.max_paths = max_paths
        self.margin = margin

    def audit(self, entry: DictionaryEntry, text: str | None = None) -> dict[str, object]:
        """
        1エントリの到達性を判定し、死にエントリなら必要調整量を計算する。

        Args:
            entry (DictionaryEntry): 検査するエントリ
            text (str | None): エントリを含む検査文。None の場合は表層単独

        Returns:
            dict[str, object]: status (reachable / dead / not_in_nbest / analysis_error) と明細
        """

        # 文を指定しない従来経路では、表層そのものを単独入力する
        analysis_text = entry.surface if text is None else text
        target_start = analysis_text.find(entry.surface)
        if target_start < 0:
            return {
                "entry": entry,
                "text": analysis_text,
                "status": "analysis_error",
                "detail": "entry surface is absent from analysis text",
            }
        target_char_span = (target_start, target_start + len(entry.surface))
        try:
            paths = self.jtalk.run_mecab_nbest_features(analysis_text, self.max_paths)
        except RuntimeError as ex:
            return {
                "entry": entry,
                "text": analysis_text,
                "status": "analysis_error",
                "detail": str(ex),
            }
        if len(paths) == 0:
            return {
                "entry": entry,
                "text": analysis_text,
                "status": "analysis_error",
                "detail": "no path returned",
            }

        best_path = paths[0]
        # 最良経路が自分自身ならコスト調整は不要
        if self._contains_entry(entry, best_path, target_char_span) is True:
            return {
                "entry": entry,
                "text": analysis_text,
                "status": "reachable",
                "best_cost": best_path["path_cost"],
            }

        # n-best の中から自分を通る経路を探し、勝者との総コスト差を実測する
        for path in paths[1:]:
            if self._contains_entry(entry, path, target_char_span) is True:
                delta = path["path_cost"] - best_path["path_cost"]
                return {
                    "entry": entry,
                    "text": analysis_text,
                    "status": "dead",
                    "best_cost": best_path["path_cost"],
                    "self_cost": path["path_cost"],
                    "delta": delta,
                    # 経路コストは単語コストに線形なので、delta + margin 下げれば必ず勝つ
                    "recommended_cost": entry.word_cost - delta - self.margin,
                    "winner": self._describe_path(best_path),
                    # 勝者の連結発音が自発音と違う場合だけ実害 (発音が壊れる誤読) になる
                    "pronunciation_broken": self._is_pronunciation_broken(
                        entry,
                        best_path,
                        target_char_span,
                    ),
                }

        # n-best 深さの範囲では自経路が現れなかった。差の下限だけ報告する
        return {
            "entry": entry,
            "text": analysis_text,
            "status": "not_in_nbest",
            "best_cost": best_path["path_cost"],
            "delta_lower_bound": paths[-1]["path_cost"] - best_path["path_cost"],
            "winner": self._describe_path(best_path),
        }

    def _contains_entry(
        self,
        entry: DictionaryEntry,
        path: MeCabNBestPath,
        target_char_span: tuple[int, int],
    ) -> bool:
        """
        経路が検査対象エントリを1形態素として通るかを判定する。

        Args:
            entry (DictionaryEntry): 検査中のエントリ
            path (MeCabNBestPath): n-best 経路
            target_char_span (tuple[int, int]): 検査文における対象表層の半開区間

        Returns:
            bool: 表層・文脈 ID・単語コストが一致する形態素を含めば True
        """

        # 文中の前後形態素は無視し、対象エントリと同じ辞書識別値を持つ形態素だけを探す
        return any(
            morph["is_ignored"] is False
            and morph["surface"] == entry.surface
            and morph["left_id"] == entry.left_id
            and morph["right_id"] == entry.right_id
            and morph["word_cost"] == entry.word_cost
            and morph["char_span"] == target_char_span
            for morph in path["morphs"]
        )

    def _is_pronunciation_broken(
        self,
        entry: DictionaryEntry,
        winner_path: MeCabNBestPath,
        target_char_span: tuple[int, int],
    ) -> bool:
        """
        勝者経路の連結発音がエントリ発音と異なる (誤読の実害がある) かを判定する。

        Args:
            entry (DictionaryEntry): 検査中のエントリ
            winner_path (MeCabNBestPath): 現在の勝者経路
            target_char_span (tuple[int, int]): 検査文における対象表層の半開区間

        Returns:
            bool: 長音転写差を吸収した上で発音が一致しなければ True
        """

        parts: list[str] = []
        for morph in winner_path["morphs"]:
            if (
                morph["is_ignored"] is True
                or morph["char_span"][1] <= target_char_span[0]
                or target_char_span[1] <= morph["char_span"][0]
            ):
                continue
            features = morph["features"]
            # 既知語の発音列は末尾から3番目 (発音, アクセント, 連接規則)。未知語は発音を持たない
            parts.append(features[-3] if len(features) >= 11 else morph["surface"])
        return self._normalize_reading("".join(parts)) != self._normalize_reading(
            entry.pronunciation
        )

    @staticmethod
    def _normalize_reading(reading: str) -> str:
        """
        長音の転写差 (オウ表記とー表記) を吸収した比較用の読みへ正規化する。

        Args:
            reading (str): カタカナ読み

        Returns:
            str: ウ段長音をーへ畳んだ読み
        """

        result: list[str] = []
        previous = ""
        for character in reading:
            # 直前がオ段 (拗音を含む) のウは長音記号と同じ発音になる
            if character == "ウ" and previous in "オコソトノホモヨロヲゴゾドボポョ":
                result.append("ー")
            else:
                result.append(character)
            previous = character
        return "".join(result)

    @staticmethod
    def _describe_path(path: MeCabNBestPath) -> str:
        """
        経路を「表層(発音)」の連結で人間可読に整形する。

        Args:
            path (MeCabNBestPath): 整形する経路

        Returns:
            str: 例 "未(ミ)+病(ヤマイ)"
        """

        parts: list[str] = []
        for morph in path["morphs"]:
            if morph["is_ignored"] is True:
                continue
            # 発音列は features の末尾から2番目 (発音,アクセント,連接規則 の並び)。未知語は列が無いので表層だけ示す
            features = morph["features"]
            pronunciation = features[-3] if len(features) >= 11 else "?"
            parts.append(f"{morph['surface']}({pronunciation})")
        return "+".join(parts)


def load_entries(
    csv_path: Path, surfaces: set[str] | None, limit: int | None
) -> list[DictionaryEntry]:
    """
    辞書 CSV から検査対象エントリを読み込む。

    Args:
        csv_path (Path): 辞書 CSV のパス
        surfaces (set[str] | None): この表層集合だけに絞る (None なら全行)
        limit (int | None): 先頭から検査する最大行数

    Returns:
        list[DictionaryEntry]: 検査対象エントリ
    """

    entries: list[DictionaryEntry] = []
    with csv_path.open(encoding="utf-8", newline="") as file:
        for line_number, row in enumerate(csv.reader(file), start=1):
            if len(row) < 13:
                continue
            if surfaces is not None and row[0] not in surfaces:
                continue
            entries.append(DictionaryEntry(row, line_number))
            if limit is not None and len(entries) >= limit:
                break
    return entries


def main() -> None:
    """引数を解釈し、対象エントリの到達性監査を実行して結果を表示・保存する。"""

    parser = argparse.ArgumentParser(description="辞書エントリの到達性監査とコスト調整量の計算")
    parser.add_argument("--csv", type=Path, required=True, help="検査する辞書 CSV")
    parser.add_argument(
        "--dictionary-dir",
        type=Path,
        default=DEFAULT_DICTIONARY_DIR,
        help="ビルド済み辞書ディレクトリ",
    )
    parser.add_argument("--surface", action="append", help="この表層だけを検査する (複数指定可)")
    parser.add_argument("--limit", type=int, default=None, help="検査する最大エントリ数")
    parser.add_argument("--nbest", type=int, default=64, help="n-best の探索深さ (1〜512)")
    parser.add_argument("--margin", type=int, default=1, help="推奨コストへ上乗せする勝ち幅")
    parser.add_argument(
        "--context",
        action="append",
        help="{surface} を対象表層へ置換する検査文テンプレート (複数指定可)",
    )
    parser.add_argument("--dead-only", action="store_true", help="死にエントリだけを表示する")
    parser.add_argument("--output", type=Path, default=None, help="明細 TSV の保存先")
    args = parser.parse_args()

    context_templates = tuple(args.context or ("{surface}",))
    if any(template.count("{surface}") != 1 for template in context_templates):
        parser.error("every --context must contain {surface} exactly once")

    # ユーザー辞書は読み込まない (デフォルト辞書単体の到達性を測るため)
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=str(args.dictionary_dir).encode("utf-8"))
    auditor = ReachabilityAuditor(jtalk, args.nbest, args.margin)

    surfaces = set(args.surface) if args.surface else None
    entries = load_entries(args.csv, surfaces, args.limit)
    print(f"auditing {len(entries)} entries from {args.csv}", file=sys.stderr)

    results = []
    status_counts: dict[str, int] = {}
    for entry in tqdm(entries, desc="auditing", unit=" entries", file=sys.stderr):
        # 同じ辞書行を全代表文で測り、文脈に依存する最も厳しいコスト差を残す
        for context_template in context_templates:
            result = auditor.audit(entry, context_template.format(surface=entry.surface))
            results.append(result)
            status_counts[str(result["status"])] = status_counts.get(str(result["status"]), 0) + 1

    print(f"summary: {status_counts}")
    output_rows: list[list[str]] = []
    for result in results:
        entry = result["entry"]
        assert isinstance(entry, DictionaryEntry)
        text = str(result["text"])
        if args.dead_only is True and result["status"] == "reachable":
            continue
        if result["status"] == "dead":
            # 発音が壊れる実害エントリと、発音は変わらない分割負けを見分けられるよう区別して表示する
            harm = "broken" if result.get("pronunciation_broken") is True else "same_pron"
            line = (
                f"[dead:{harm}] {entry.surface}={entry.pronunciation} cost={entry.word_cost} "
                f"delta={result['delta']} recommended={result['recommended_cost']} "
                f"winner={result['winner']} text={text!r} (line {entry.line_number})"
            )
        elif result["status"] == "not_in_nbest":
            line = (
                f"[not_in_nbest] {entry.surface}={entry.pronunciation} cost={entry.word_cost} "
                f"delta>{result['delta_lower_bound']} winner={result['winner']} "
                f"text={text!r} (line {entry.line_number})"
            )
        elif result["status"] == "reachable":
            line = (
                f"[reachable] {entry.surface}={entry.pronunciation} "
                f"cost={entry.word_cost} text={text!r}"
            )
        else:
            line = f"[analysis_error] {entry.surface}={entry.pronunciation}: {result['detail']}"
        print(line)
        output_rows.append(
            [
                str(result["status"]),
                "broken" if result.get("pronunciation_broken") is True else "",
                entry.surface,
                entry.pronunciation,
                str(entry.word_cost),
                str(result.get("delta", result.get("delta_lower_bound", ""))),
                str(result.get("recommended_cost", "")),
                str(result.get("winner", "")),
                text,
                str(entry.line_number),
            ]
        )

    # 全代表文で到達させる場合は、文ごとの推奨値のうち最も低い値が安全な上限になる
    for entry in entries:
        entry_results = [result for result in results if result["entry"] is entry]
        recommended_costs = [
            int(result["recommended_cost"])
            for result in entry_results
            if result["status"] == "dead"
        ]
        if len(recommended_costs) > 0:
            print(
                f"[recommended_for_all_contexts] {entry.surface} "
                f"cost={min(recommended_costs)} checks={len(entry_results)}"
            )

    # TSV には判定、実害区分、辞書行、コスト差、勝者経路、検査文を残す
    if args.output is not None:
        header = (
            "status\tpronunciation_harm\tsurface\tpronunciation\tword_cost\tdelta\t"
            "recommended_cost\twinner\ttext\tline_number\n"
        )
        args.output.write_text(
            header + "\n".join("\t".join(row) for row in output_rows) + "\n", encoding="utf-8"
        )
        print(f"saved: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
