"""Tests for pure utility functions (no GPU/models needed)."""

import numpy as np
import pytest

from app.models.corner import compute_output_size, order_corners
from app.models.vlm import LocalVLM, _make_few_shot_prompt, _parse_json_output


# --- _parse_json_output ---


class TestParseJsonOutput:
    def test_plain_json(self):
        assert _parse_json_output('{"name": "John"}') == {"name": "John"}

    def test_markdown_code_block(self):
        text = '```json\n{"a": "1"}\n```'
        assert _parse_json_output(text) == {"a": "1"}

    def test_markdown_block_no_lang(self):
        text = '```\n{"a": "1"}\n```'
        assert _parse_json_output(text) == {"a": "1"}

    def test_json_with_surrounding_text(self):
        text = 'Here is the result:\n{"x": "y"}\nDone.'
        assert _parse_json_output(text) == {"x": "y"}

    def test_empty_json(self):
        assert _parse_json_output("{}") == {}

    def test_no_json(self):
        assert _parse_json_output("no json here") == {}

    def test_invalid_json(self):
        assert _parse_json_output("{broken: json}") == {}

    def test_nested_values_to_str(self):
        text = '{"count": 42, "flag": true}'
        result = _parse_json_output(text)
        assert result == {"count": "42", "flag": "True"}

    def test_unicode(self):
        text = '{"имя": "Иван"}'
        assert _parse_json_output(text) == {"имя": "Иван"}


# --- _make_few_shot_prompt ---


class TestMakeFewShotPrompt:
    def test_without_examples(self):
        prompt = _make_few_shot_prompt()
        assert "Return ONLY a JSON object" in prompt
        assert "Example output" not in prompt

    def test_with_examples(self):
        example = {"name": "John", "birth_date": "01.01.1990"}
        prompt = _make_few_shot_prompt(example)
        assert "Example output" in prompt
        assert '"name": "John"' in prompt
        assert "Return ONLY a JSON object" in prompt


# --- order_corners ---


class TestOrderCorners:
    def test_already_ordered(self):
        pts = np.array([[0, 0], [100, 0], [100, 80], [0, 80]], dtype=np.float32)
        result = order_corners(pts)
        np.testing.assert_array_equal(result[0], [0, 0])    # TL
        np.testing.assert_array_equal(result[1], [100, 0])   # TR
        np.testing.assert_array_equal(result[2], [100, 80])  # BR
        np.testing.assert_array_equal(result[3], [0, 80])    # BL

    def test_shuffled(self):
        pts = np.array([[100, 80], [0, 0], [0, 80], [100, 0]], dtype=np.float32)
        result = order_corners(pts)
        assert result[0][0] < result[1][0]  # TL.x < TR.x
        assert result[0][1] < result[3][1]  # TL.y < BL.y

    def test_returns_four_points(self):
        pts = np.array([[10, 20], [200, 10], [210, 150], [5, 160]], dtype=np.float32)
        result = order_corners(pts)
        assert result.shape == (4, 2)


# --- compute_output_size ---


class TestComputeOutputSize:
    def test_rectangle(self):
        quad = np.array([[0, 0], [300, 0], [300, 200], [0, 200]], dtype=np.float32)
        w, h = compute_output_size(quad)
        assert w == 300
        assert h == 200

    def test_trapezoid_takes_max(self):
        quad = np.array([[10, 0], [290, 0], [300, 200], [0, 200]], dtype=np.float32)
        w, h = compute_output_size(quad)
        assert w == 300  # bottom edge is longer


# --- _resize_for_vlm ---


class TestResizeForVlm:
    def test_small_image_unchanged(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = LocalVLM._resize_for_vlm(img, 802816)
        assert result.shape == (100, 200, 3)

    def test_large_image_downscaled(self):
        img = np.zeros((2000, 3000, 3), dtype=np.uint8)
        result = LocalVLM._resize_for_vlm(img, 802816)
        h, w = result.shape[:2]
        assert h * w <= 802816
        assert h < 2000
        assert w < 3000

    def test_preserves_aspect_ratio(self):
        img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        result = LocalVLM._resize_for_vlm(img, 500000)
        h, w = result.shape[:2]
        assert abs(w / h - 2.0) < 0.05
