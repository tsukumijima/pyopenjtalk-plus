"""辞書交換と共有インスタンスの並行実行契約を検証する。"""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
from time import sleep
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

import pyopenjtalk
import pyopenjtalk.tsqyomi as tsqyomi
from pyopenjtalk import NJDFeature


def test_replacement_waits_for_global_jtalk_frontend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """フロントエンド借り出し中はグローバル OpenJTalk の交換が待たされる。"""

    original_global_jtalk = cast(Any, pyopenjtalk)._global_jtalk
    original_instance = original_global_jtalk._instance
    is_postprocessing_started = Event()
    can_finish_postprocessing = Event()
    is_swap_started = Event()
    is_swap_finished = Event()
    original_apply = pyopenjtalk.apply_postprocessing

    def slow_apply(*args: Any, **kwargs: Any) -> list[NJDFeature]:
        is_postprocessing_started.set()
        assert can_finish_postprocessing.wait(timeout=5.0) is True
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(pyopenjtalk, "apply_postprocessing", slow_apply)

    def hold_frontend() -> None:
        pyopenjtalk.run_frontend("テストです。")

    def swap_global_jtalk() -> None:
        assert is_postprocessing_started.wait(timeout=5.0) is True
        is_swap_started.set()
        pyopenjtalk.unset_user_dict()
        is_swap_finished.set()

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            frontend_future = executor.submit(hold_frontend)
            swap_future = executor.submit(swap_global_jtalk)
            assert is_swap_started.wait(timeout=5.0) is True
            assert is_swap_finished.wait(timeout=0.1) is False
            can_finish_postprocessing.set()
            frontend_future.result(timeout=10.0)
            swap_future.result(timeout=10.0)
    finally:
        monkeypatch.setattr(pyopenjtalk, "_global_jtalk", original_global_jtalk)
        if original_instance is not None:
            original_global_jtalk.replace(original_instance)

    assert is_swap_finished.is_set() is True


def test_replace_waits_for_all_active_leases() -> None:
    """`replace()` は複数の借り出しが全て返却されるまで待機する。"""

    manager = cast(Any, pyopenjtalk)._ReplaceableInstanceManager(lambda: "old")
    is_first_lease_started = Event()
    is_second_lease_started = Event()
    can_finish_first_lease = Event()
    can_finish_second_lease = Event()
    is_first_lease_finished = Event()
    is_replacement_started = Event()
    is_replacement_finished = Event()

    def hold_lease(is_started: Event, can_finish: Event, is_finished: Event | None = None) -> None:
        with manager():
            is_started.set()
            assert can_finish.wait(timeout=5.0) is True
        if is_finished is not None:
            is_finished.set()

    def replace_after_all_leases() -> None:
        is_replacement_started.set()
        manager.replace("new")
        is_replacement_finished.set()

    with ThreadPoolExecutor(max_workers=3) as executor:
        first_lease_future = executor.submit(
            hold_lease,
            is_first_lease_started,
            can_finish_first_lease,
            is_first_lease_finished,
        )
        second_lease_future = executor.submit(
            hold_lease,
            is_second_lease_started,
            can_finish_second_lease,
        )
        assert is_first_lease_started.wait(timeout=5.0) is True
        assert is_second_lease_started.wait(timeout=5.0) is True

        replacement_future = executor.submit(replace_after_all_leases)
        assert is_replacement_started.wait(timeout=5.0) is True
        can_finish_first_lease.set()
        assert is_first_lease_finished.wait(timeout=5.0) is True
        assert is_replacement_finished.wait(timeout=0.1) is False

        can_finish_second_lease.set()
        assert is_replacement_finished.wait(timeout=5.0) is True
        first_lease_future.result(timeout=10.0)
        second_lease_future.result(timeout=10.0)
        replacement_future.result(timeout=10.0)


def test_waiting_lease_uses_replaced_instance() -> None:
    """交換待機中に開始した借り出しは新しいインスタンスを取得する。"""

    manager = cast(Any, pyopenjtalk)._ReplaceableInstanceManager(lambda: "old")
    is_initial_lease_started = Event()
    can_finish_initial_lease = Event()
    is_replacement_started = Event()
    is_replacement_finished = Event()
    is_new_lease_started = Event()
    is_new_lease_entered = Event()
    borrowed_instances: list[str] = []

    def hold_initial_lease() -> None:
        with manager():
            is_initial_lease_started.set()
            assert can_finish_initial_lease.wait(timeout=5.0) is True

    def replace_instance() -> None:
        is_replacement_started.set()
        manager.replace("new")
        is_replacement_finished.set()

    def borrow_during_replacement() -> None:
        is_new_lease_started.set()
        with manager() as instance:
            borrowed_instances.append(instance)
            is_new_lease_entered.set()

    with ThreadPoolExecutor(max_workers=3) as executor:
        initial_lease_future = executor.submit(hold_initial_lease)
        assert is_initial_lease_started.wait(timeout=5.0) is True

        replacement_future = executor.submit(replace_instance)
        assert is_replacement_started.wait(timeout=5.0) is True
        assert is_replacement_finished.wait(timeout=0.1) is False

        new_lease_future = executor.submit(borrow_during_replacement)
        assert is_new_lease_started.wait(timeout=5.0) is True
        assert is_new_lease_entered.wait(timeout=0.1) is False

        can_finish_initial_lease.set()
        assert is_replacement_finished.wait(timeout=5.0) is True
        assert is_new_lease_entered.wait(timeout=5.0) is True
        initial_lease_future.result(timeout=10.0)
        replacement_future.result(timeout=10.0)
        new_lease_future.result(timeout=10.0)

    assert borrowed_instances == ["new"]


def test_unset_user_dict_keeps_global_jtalk_manager_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """辞書交換後も待機中の呼び出しが参照するマネージャーを維持する。"""

    module = cast(Any, pyopenjtalk)
    manager = module._ReplaceableInstanceManager(lambda: "old")

    def create_fake_openjtalk(**_kwargs: Any) -> str:
        """交換後の OpenJTalk を表す固定値を返す"""

        return "new"

    monkeypatch.setattr(module, "_global_jtalk", manager)
    monkeypatch.setattr(module, "OpenJTalk", create_fake_openjtalk)

    pyopenjtalk.unset_user_dict()

    assert module._global_jtalk is manager
    with manager() as instance:
        assert instance == "new"


def test_synthesize_serializes_htsengine_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTSEngine の設定変更から合成完了までを呼び出し単位で排他する。"""

    first_speed_is_set = Event()
    second_speed_is_set = Event()
    is_second_synthesis_started = Event()
    can_finish_first_synthesis = Event()
    original_synthesize = pyopenjtalk.synthesize

    def observed_synthesize(
        labels: list[str], speed: float = 1.0, half_tone: float = 0.0
    ) -> tuple[npt.NDArray[np.float64], int]:
        """2件目が実行開始したことを記録して実際の合成関数へ委譲する。"""

        if speed == 2.0:
            is_second_synthesis_started.set()
        return original_synthesize(labels, speed, half_tone)

    monkeypatch.setattr(pyopenjtalk, "synthesize", observed_synthesize)

    class FakeHTSEngine:
        """並行する話速設定の混在を検出するテスト用 HTSEngine。"""

        def __init__(self) -> None:
            self.speed = 0.0

        def get_sampling_frequency(self) -> int:
            """固定のサンプリング周波数を返す"""

            return 48000

        def set_speed(self, speed: float) -> None:
            """1件目の話速設定後に処理を止め、2件目が割り込めるかを観測する"""

            self.speed = speed
            if speed == 1.0:
                first_speed_is_set.set()
                assert can_finish_first_synthesis.wait(timeout=5.0) is True
            else:
                second_speed_is_set.set()

        def add_half_tone(self, _half_tone: float) -> None:
            """テスト対象外の半音設定を受け取る"""

        def synthesize(self, _labels: list[str]) -> npt.NDArray[np.float64]:
            """合成時点の話速を波形として返す"""

            return np.array([self.speed], dtype=np.float64)

    module = cast(Any, pyopenjtalk)
    fake_htsengine = FakeHTSEngine()
    monkeypatch.setattr(
        module,
        "_global_htsengine",
        module._ExclusiveInstanceManager(lambda: fake_htsengine),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(pyopenjtalk.synthesize, ["first"], 1.0)
        assert first_speed_is_set.wait(timeout=5.0) is True
        second_future = executor.submit(pyopenjtalk.synthesize, ["second"], 2.0)
        assert is_second_synthesis_started.wait(timeout=5.0) is True
        try:
            did_second_enter_early = second_speed_is_set.wait(timeout=0.1)
        finally:
            can_finish_first_synthesis.set()

        first_waveform, _ = first_future.result(timeout=10.0)
        second_waveform, _ = second_future.result(timeout=10.0)

    assert did_second_enter_early is False
    assert first_waveform.tolist() == [1.0]
    assert second_waveform.tolist() == [2.0]


def test_openjtalk_instances_have_independent_locks() -> None:
    """異なる OpenJTalk インスタンスが同じ排他ロックを共有しないことを確認。"""

    first_jtalk = pyopenjtalk.openjtalk.OpenJTalk(pyopenjtalk.OPEN_JTALK_DICT_DIR)
    second_jtalk = pyopenjtalk.openjtalk.OpenJTalk(pyopenjtalk.OPEN_JTALK_DICT_DIR)

    assert cast(Any, first_jtalk)._lock is not cast(Any, second_jtalk)._lock


def test_htsengine_instances_have_independent_locks() -> None:
    """異なる HTSEngine インスタンスが同じ排他ロックを共有しないことを確認。"""

    first_engine = pyopenjtalk.htsengine.HTSEngine(pyopenjtalk.DEFAULT_HTS_VOICE)
    second_engine = pyopenjtalk.htsengine.HTSEngine(pyopenjtalk.DEFAULT_HTS_VOICE)

    assert cast(Any, first_engine)._lock is not cast(Any, second_engine)._lock


def _concurrent_inference_test_metadata() -> tsqyomi.TsqyomiMetadata:
    """並行推論テスト用の最小 v2 メタデータ。"""

    return tsqyomi.TsqyomiMetadata.model_validate(
        {
            "schema_version": "modernbert_reading_class_v2",
            "target_boundary_contract": "mecab_target_segments_v1",
            "model_max_length": 512,
            "pad_token_id": 0,
            "model_scored_surfaces": ["人気"],
            "output_class_order": ["rc_1", "rc_2"],
            "reading_class_ids_by_surface_and_pronunciation": {
                "人気": {"ニンキ": ["rc_1"], "ヒトケ": ["rc_2"]},
            },
        }
    )


def _concurrent_inference_test_target() -> tsqyomi.ReadingTarget:
    """並行推論テスト用の単一対象。"""

    return tsqyomi.ReadingTarget(
        char_span=(0, 2),
        surface="人気",
        pronunciations=("ニンキ", "ヒトケ"),
    )


class _ConcurrentInferenceTestTokenizer:
    """`_predict_single_window()` 向けの最小トークナイザー。"""

    def encode(self, text: str, add_special_tokens: bool = True) -> SimpleNamespace:
        """空文字列と本文断片を固定トークン列へ符号化する。"""

        if text == "":
            return SimpleNamespace(ids=[1, 2], special_tokens_mask=[1, 1])
        return SimpleNamespace(ids=[10], offsets=[(0, len(text))])


def test_directml_model_serializes_concurrent_inference() -> None:
    """DirectML 用モデルは複数スレッドの ONNX Run() をモデル側で直列化する。"""

    class Session:
        """同時実行数を記録する DirectML 相当の ONNX セッション。"""

        def __init__(self) -> None:
            """同時実行数と排他制御を初期化する。"""

            self.active_count = 0
            self.maximum_active_count = 0
            self.lock = Lock()

        def run(self, _output_names: list[str], model_inputs: dict[str, Any]) -> list[Any]:
            """実行中の同時呼び出し数を記録して固定ロジットを返す。"""

            with self.lock:
                self.active_count += 1
                self.maximum_active_count = max(self.maximum_active_count, self.active_count)
            sleep(0.02)
            with self.lock:
                self.active_count -= 1
            target_count = len(model_inputs["target_mask"][0])
            return [np.zeros((1, target_count, 2), dtype=np.float32)]

        @staticmethod
        def get_providers() -> list[str]:
            """DirectML 利用中の ONNX セッション相当の EP 列を返す。"""

            return ["DmlExecutionProvider", "CPUExecutionProvider"]

    session = Session()
    model = tsqyomi.TsqyomiModel(
        _ConcurrentInferenceTestTokenizer(),
        session,
        _concurrent_inference_test_metadata(),
    )
    target = _concurrent_inference_test_target()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(model.predict, "人気", (target,)) for _ in range(2)]
        for future in futures:
            future.result(timeout=5.0)
    assert session.maximum_active_count == 1


def test_cpu_and_cuda_model_allow_concurrent_inference() -> None:
    """CPU と CUDA 用モデルは複数スレッドの ONNX Run() を並行実行できる。"""

    class Session:
        """同時実行数を記録する CPU・CUDA 相当の ONNX セッション。"""

        def __init__(self) -> None:
            """同時実行数と同期用バリアを初期化する。"""

            self.active_count = 0
            self.maximum_active_count = 0
            self.lock = Lock()
            self.barrier = Barrier(2)

        def run(self, _output_names: list[str], model_inputs: dict[str, Any]) -> list[Any]:
            """2スレッドを同期し、並行実行数を記録して固定ロジットを返す。"""

            with self.lock:
                self.active_count += 1
                self.maximum_active_count = max(self.maximum_active_count, self.active_count)
            self.barrier.wait(timeout=5.0)
            with self.lock:
                self.active_count -= 1
            target_count = len(model_inputs["target_mask"][0])
            return [np.zeros((1, target_count, 2), dtype=np.float32)]

        @staticmethod
        def get_providers() -> list[str]:
            """CPU 利用中の ONNX セッション相当の EP 列を返す。"""

            return ["CPUExecutionProvider"]

    session = Session()
    model = tsqyomi.TsqyomiModel(
        _ConcurrentInferenceTestTokenizer(),
        session,
        _concurrent_inference_test_metadata(),
    )
    target = _concurrent_inference_test_target()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(model.predict, "人気", (target,)) for _ in range(2)]
        for future in futures:
            future.result(timeout=5.0)
    assert session.maximum_active_count == 2
