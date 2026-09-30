"""
形態素-音素マッピングに、フルコンテキストラベルから読み取ったモーラの高低とアクセント句の区切りを重ね、韻律付きの音素列として整形する処理。
pyopenjtalk.g2p_mapping_prosody() と pyopenjtalk.g2p_prosody() が利用する。
"""

from collections.abc import Sequence
from typing import cast

from .types import (
    ProsodicMarker,
    ProsodicMarkerKind,
    ProsodicPhone,
    ProsodicPhoneme,
    ProsodyFormat,
    ProsodyPitch,
    SurfacePhonemeMapping,
    SurfaceProsodyMapping,
)


# g2p_prosody() の出力で、韻律の区切りの種類ごとに使う記号
_PROSODIC_MARKER_SYMBOLS: dict[ProsodicMarkerKind, str] = {
    "AccentPhraseBoundary": "#",
    "Pause": "_",
    "Interrogative": "?",
    "Exclamatory": "!",
}
# アクセント句の区切りを置いてよい、句の最後のモーラの終わりの音素 (母音・撥音・促音)
_ACCENT_BOUNDARY_FINAL_PHONEMES = frozenset(
    {"a", "e", "i", "o", "u", "A", "E", "I", "O", "U", "N", "cl"}
)


def make_prosody_mapping(
    mapping: Sequence[SurfacePhonemeMapping],
    labels: Sequence[str],
) -> list[SurfaceProsodyMapping]:
    """
    詳細音素マッピングへ HTS ラベル由来のピッチと韻律境界を対応付ける。

    Args:
        mapping (Sequence[SurfacePhonemeMapping]): 表層と通常音素の詳細マッピング
        labels (Sequence[str]): 同じ NJD features から生成したフルコンテキストラベル

    Returns:
        list[SurfaceProsodyMapping]: 表層・入力座標を保った韻律情報付きマッピング

    Raises:
        RuntimeError: 通常音素列とフルコンテキストラベルの対応が崩れた場合
    """

    parsed_labels = _parse_fullcontext_prosody(labels)
    expected_label_phonemes = [
        phoneme
        for entry in mapping
        for phoneme in entry["phonemes"]
        if phoneme not in {"pau", "sp", "unk"}
    ]
    prosody_labels = [parsed_label for parsed_label in parsed_labels if parsed_label[0] != "pau"]
    actual_label_phonemes = [label_phoneme for label_phoneme, *_ in prosody_labels]
    if expected_label_phonemes != actual_label_phonemes:
        raise RuntimeError("phoneme mapping and full-context labels do not align")

    prosody_mapping: list[SurfaceProsodyMapping] = []
    label_index = 0
    flattened_mapping_phonemes = [phoneme for entry in mapping for phoneme in entry["phonemes"]]
    mapping_phoneme_index = 0
    for entry in mapping:
        prosodic_phonemes: list[ProsodicPhoneme] = []
        for phoneme in entry["phonemes"]:
            # 空白と未知語はフルコンテキストラベルに現れないため、既存の詳細マッピングの値を維持する
            if phoneme in {"sp", "unk"}:
                prosodic_phonemes.append(ProsodicPhone(kind="Phoneme", phoneme=phoneme, pitch=None))
                mapping_phoneme_index += 1
                continue

            # 句読点の pau は音声区間ではなく韻律境界として表し、疑問符と感嘆符は区別を保持する
            if phoneme == "pau":
                surface = entry["surface"]
                if "？" in surface or "?" in surface:
                    pause_kind: ProsodicMarkerKind = "Interrogative"
                elif "！" in surface or "!" in surface:
                    pause_kind = "Exclamatory"
                else:
                    pause_kind = "Pause"
                prosodic_phonemes.append(ProsodicMarker(kind=pause_kind))
                mapping_phoneme_index += 1
                continue

            label_phoneme, accent_nucleus, mora_position, remaining_mora_count = prosody_labels[
                label_index
            ]
            if phoneme != label_phoneme:
                raise RuntimeError(
                    "phoneme mapping and full-context labels diverged during alignment"
                )
            label_index += 1

            if accent_nucleus is None or mora_position is None or remaining_mora_count is None:
                raise RuntimeError("OpenJTalk phoneme label lacks prosody information")
            # 平板型は第2モーラ以降、頭高型は第1モーラだけが高くなる
            if accent_nucleus == 0:
                pitch: ProsodyPitch = "High" if mora_position >= 2 else "Low"
            elif accent_nucleus == 1:
                pitch = "High" if mora_position == 1 else "Low"
            else:
                pitch = "High" if 2 <= mora_position <= accent_nucleus else "Low"
            prosodic_phonemes.append(
                ProsodicPhone(
                    kind="Phoneme",
                    phoneme=phoneme,
                    pitch=pitch,
                )
            )

            # 母音・撥音・促音の終端から次の句の先頭モーラへ移る位置だけをアクセント句境界にする
            next_mapping_phoneme = (
                flattened_mapping_phonemes[mapping_phoneme_index + 1]
                if mapping_phoneme_index + 1 < len(flattened_mapping_phonemes)
                else None
            )
            if label_index < len(prosody_labels) and next_mapping_phoneme not in {
                None,
                "pau",
                "sp",
                "unk",
            }:
                _, _, next_mora_position, _ = prosody_labels[label_index]
                if (
                    remaining_mora_count == 1
                    and next_mora_position == 1
                    and phoneme in _ACCENT_BOUNDARY_FINAL_PHONEMES
                ):
                    prosodic_phonemes.append(ProsodicMarker(kind="AccentPhraseBoundary"))
            mapping_phoneme_index += 1

        prosody_mapping.append(
            cast(
                SurfaceProsodyMapping,
                {
                    **entry,
                    "phonemes": prosodic_phonemes,
                },
            )
        )

    if label_index != len(prosody_labels):
        raise RuntimeError("not every full-context label was assigned to a phoneme mapping")
    return prosody_mapping


def format_prosody_phonemes(
    prosody_mapping: Sequence[SurfaceProsodyMapping],
    format: ProsodyFormat,
) -> list[str]:
    """
    韻律付きのマッピングを、g2p_prosody() が返す記号付きの音素列に並べ直す。

    Args:
        prosody_mapping (Sequence[SurfaceProsodyMapping]): make_prosody_mapping() が返したマッピング
        format (ProsodyFormat): ピッチの表記方式 (`Default` は高低が変わる位置に `[` `]`、`Prefix` は `H_` / `L_`、`Numeric` は `:1` / `:0` を使う)

    Returns:
        list[str]: 先頭の ^ と末尾の $、ピッチの記号、アクセント句の区切り、句読点の記号を含む音素列
    """

    mapping = prosody_mapping
    if len(mapping) == 0:
        return []

    formatted_phonemes = ["^"]
    previous_pitch: ProsodyPitch | None = None
    for entry in mapping:
        if entry["is_unknown"] is True:
            formatted_phonemes.append("{")
        for prosodic_phoneme in entry["phonemes"]:
            if prosodic_phoneme["kind"] == "Phoneme":
                phoneme = prosodic_phoneme["phoneme"]
                pitch = prosodic_phoneme["pitch"]
                if format == "Default" and pitch is not None:
                    if previous_pitch == "Low" and pitch == "High":
                        formatted_phonemes.append("[")
                    elif previous_pitch == "High" and pitch == "Low":
                        formatted_phonemes.append("]")
                    previous_pitch = pitch
                if format == "Prefix":
                    formatted_phonemes.append(
                        f"{'H_' if pitch == 'High' else 'L_' if pitch == 'Low' else ''}{phoneme}"
                    )
                elif format == "Numeric":
                    formatted_phonemes.append(
                        f"{phoneme}{':1' if pitch == 'High' else ':0' if pitch == 'Low' else ''}"
                    )
                else:
                    formatted_phonemes.append(phoneme)
                continue

            kind = prosodic_phoneme["kind"]
            marker = _PROSODIC_MARKER_SYMBOLS[kind]
            formatted_phonemes.append(marker)
            # 区切り記号の後は別のアクセント句としてピッチ変化を判定する
            if format == "Default":
                previous_pitch = None
        if entry["is_unknown"] is True:
            formatted_phonemes.append("}")
    formatted_phonemes.append("$")
    return formatted_phonemes


def _parse_fullcontext_prosody(
    labels: Sequence[str],
) -> list[tuple[str, int | None, int | None, int | None]]:
    """
    HTS フルコンテキストラベルから中心音素とアクセント句内の位置を取り出す。

    Args:
        labels (Sequence[str]): OpenJTalk が生成したフルコンテキストラベル

    Returns:
        list[tuple[str, int | None, int | None, int | None]]: 中心音素、アクセント核、句内モーラ位置、残りモーラ数

    Raises:
        RuntimeError: OpenJTalk のラベル形式を解釈できない場合
    """

    parsed_labels: list[tuple[str, int | None, int | None, int | None]] = []
    for label in labels:
        try:
            phoneme_context = label.split("/", maxsplit=1)[0]
            phoneme = phoneme_context.split("-", maxsplit=1)[1].split("+", maxsplit=1)[0]
        except (IndexError, ValueError) as ex:
            raise RuntimeError(
                "OpenJTalk full-context label does not contain a central phoneme"
            ) from ex

        # 文頭・文末の sil とダミーの xx は形態素に対応しないため、詳細マッピングとの照合から除外する
        if phoneme in {"sil", "xx"}:
            continue
        if phoneme == "pau":
            parsed_labels.append((phoneme, None, None, None))
            continue

        try:
            mora_context = label.split("/A:", maxsplit=1)[1].split("/", maxsplit=1)[0]
            _, mora_position, remaining_mora_count = mora_context.split("+")
            accent_context = label.split("/F:", maxsplit=1)[1].split("/", maxsplit=1)[0]
            accent_nucleus = accent_context.split("#", maxsplit=1)[0].split("_", maxsplit=1)[1]
            parsed_labels.append(
                (
                    phoneme,
                    int(accent_nucleus),
                    int(mora_position),
                    int(remaining_mora_count),
                )
            )
        except (IndexError, ValueError) as ex:
            raise RuntimeError(
                "OpenJTalk full-context label does not contain prosody fields"
            ) from ex
    return parsed_labels
