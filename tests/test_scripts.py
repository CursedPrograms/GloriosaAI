import random

from PIL import Image

import common
import image_processor


def test_load_settings_fills_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"epochs": 5, "directories": {"video": "v"}}')
    settings = common.load_settings(path)
    assert settings["epochs"] == 5
    assert settings["batch_size"] == common.DEFAULT_SETTINGS["batch_size"]
    assert settings["directories"]["video"] == "v"
    assert settings["directories"]["training_data"] == common.DEFAULT_SETTINGS["directories"]["training_data"]


def test_list_images_is_recursive_and_filters(tmp_path):
    (tmp_path / "a").mkdir()
    Image.new("RGB", (4, 4)).save(tmp_path / "a" / "one.PNG")
    (tmp_path / "notes.txt").write_text("x")
    assert [p.name for p in common.list_images(tmp_path)] == ["one.PNG"]
    assert common.list_images(tmp_path / "missing") == []


def test_process_image_is_square_rgb_at_target_size():
    source = Image.new("RGBA", (300, 200), (255, 0, 0, 128))
    result = image_processor.process_image(source, size=64, rng=random.Random(1))
    assert result.size == (64, 64)
    assert result.mode == "RGB"


def test_main_writes_variations(tmp_path):
    src, dst = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    Image.new("RGB", (50, 80)).save(src / "pic.jpg")
    code = image_processor.main(["--input-dir", str(src), "--output-dir", str(dst),
                                 "--variations", "2", "--seed", "0"])
    assert code == 0
    assert sorted(p.name for p in dst.iterdir()) == ["processed_pic_0.png", "processed_pic_1.png"]


class _FakeLayer:
    def __init__(self, activation):
        self.activation = activation


class _FakeModel:
    def __init__(self, activation):
        self.layers = [_FakeLayer(activation)]


def test_modelout_rescales_by_output_activation():
    import numpy as np
    import modelout

    def tanh(x):
        return x

    def sigmoid(x):
        return x

    batch = np.array([[[[-1.0, 0.0, 1.0]]]])
    assert modelout.to_uint8(batch, _FakeModel(tanh)).ravel().tolist() == [0, 127, 255]
    assert modelout.to_uint8(np.array([[[[0.0, 0.5, 1.0]]]]), _FakeModel(sigmoid)).ravel().tolist() == [0, 127, 255]


def test_modelout_finds_new_and_legacy_generators(tmp_path):
    import modelout

    (tmp_path / "generator_10.keras").write_text("")
    (tmp_path / "generator_architecture_5.json").write_text("{}")
    (tmp_path / "generator_weights_5.h5").write_text("")
    (tmp_path / "generator_architecture_7.json").write_text("{}")  # no weights: skipped
    assert [label for label, _ in modelout.find_generators(tmp_path)] == ["generator_10", "generator_5"]


def test_encode_video_numbers_files_and_prunes_old_ones(tmp_path):
    import video_encoder

    frames, out = tmp_path / "frames", tmp_path / "video"
    frames.mkdir()
    for i in range(3):
        Image.new("RGB", (16, 16), (i * 50, 0, 0)).save(frames / f"f{i}.png")
    for _ in range(3):
        assert video_encoder.encode_video(frames, out, fps=5, keep=2) is not None
    assert sorted(p.name for p in out.glob("*.mp4")) == ["output_video_2.mp4", "output_video_3.mp4"]
    assert video_encoder.encode_video(tmp_path / "empty", out) is None
