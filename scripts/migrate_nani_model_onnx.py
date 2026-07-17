import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def migrate_nani_model(source_path: Path, output_path: Path) -> None:
    """
    旧 ONNX の暗黙的な二値確率を明示的な二クラス確率へ変換する。

    Args:
        source_path (Path): skl2onnx が出力した旧 ONNX モデル
        output_path (Path): 変換後の ONNX モデルの保存先

    Raises:
        ValueError: 入力モデルが想定している旧 nani_model の構造と異なる場合
    """

    model = onnx.load(source_path)
    classifier_nodes = [
        node
        for node in model.graph.node
        if node.op_type == "TreeEnsembleClassifier" and node.domain == "ai.onnx.ml"
    ]
    if len(classifier_nodes) != 1:
        raise ValueError("Expected exactly one ai.onnx.ml.TreeEnsembleClassifier node")

    classifier_node = classifier_nodes[0]
    attributes = {attribute.name: attribute for attribute in classifier_node.attribute}
    class_labels = list(attributes["classlabels_int64s"].ints)
    class_ids = list(attributes["class_ids"].ints)
    class_tree_ids = list(attributes["class_treeids"].ints)
    class_node_ids = list(attributes["class_nodeids"].ints)
    class_weights = list(attributes["class_weights"].floats)

    # 旧 skl2onnx はクラスラベルを二つ持ちながら、各葉にはクラス 1 の確率だけを class_id=0 として格納する
    ## ONNX Runtime 1.25 以前は残りを 1 - score で補っていたが、この暗黙補完は 1.26 で削除された
    if class_labels != [0, 1] or set(class_ids) != {0}:
        raise ValueError("Expected a legacy binary classifier with labels [0, 1] and class_id=0")
    if not (len(class_ids) == len(class_tree_ids) == len(class_node_ids) == len(class_weights)):
        raise ValueError("Classifier leaf attribute arrays have inconsistent lengths")

    tree_count = len(set(class_tree_ids))
    if tree_count == 0:
        raise ValueError("Classifier does not contain any trees")
    tree_weight = 1.0 / tree_count

    explicit_class_ids: list[int] = []
    explicit_class_tree_ids: list[int] = []
    explicit_class_node_ids: list[int] = []
    explicit_class_weights: list[float] = []
    for class_tree_id, class_node_id, class_weight in zip(
        class_tree_ids,
        class_node_ids,
        class_weights,
    ):
        # RandomForest の各木は 1 / 木数の確率を持つため、保存済み確率の補数を同じ葉へ明示する
        explicit_class_ids.extend([0, 1])
        explicit_class_tree_ids.extend([class_tree_id, class_tree_id])
        explicit_class_node_ids.extend([class_node_id, class_node_id])
        explicit_class_weights.extend([tree_weight - class_weight, class_weight])

    attributes["class_ids"].ints[:] = explicit_class_ids
    attributes["class_treeids"].ints[:] = explicit_class_tree_ids
    attributes["class_nodeids"].ints[:] = explicit_class_node_ids
    attributes["class_weights"].floats[:] = explicit_class_weights

    # ORT の二値分類ラベル生成にもバージョン間差があるため、確率テンソルだけを Python 側へ公開する
    ## 後続の Cast と ZipMap はラベルと辞書形式の確率を作る処理なので、分類器本体だけを残す
    model.graph.ClearField("node")
    model.graph.node.extend([classifier_node])
    model.graph.ClearField("output")
    model.graph.output.extend(
        [
            helper.make_tensor_value_info(
                classifier_node.output[1],
                TensorProto.FLOAT,
                [None, 2],
            )
        ]
    )

    onnx.checker.check_model(model)
    onnx.save(model, output_path)


def main() -> None:
    """コマンドライン引数で指定された旧 ONNX モデルを変換する。"""

    parser = argparse.ArgumentParser(
        description="Migrate the nani RandomForest ONNX model to explicit binary probabilities.",
    )
    parser.add_argument("source", type=Path, help="Legacy nani_model.onnx path")
    parser.add_argument("output", type=Path, help="Migrated nani_model.onnx path")
    args = parser.parse_args()
    migrate_nani_model(args.source, args.output)


if __name__ == "__main__":
    main()
