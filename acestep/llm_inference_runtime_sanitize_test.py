"""Unit tests for torch runtime sanitization in ``LLMHandler``."""

import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import torch

try:
    from acestep.llm_inference import LLMHandler
    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - dependency guard
    LLMHandler = None
    _IMPORT_ERROR = exc


@unittest.skipIf(LLMHandler is None, f"llm_inference import unavailable: {_IMPORT_ERROR}")
class LlmRuntimeSanitizeTests(unittest.TestCase):
    """Verify runtime-state sanitization on PyTorch fallback paths."""

    def test_load_pytorch_model_sanitizes_runtime_first(self):
        """PyTorch backend load should sanitize stale global state before model load."""
        handler = LLMHandler()
        handler.offload_to_cpu = True
        handler.dtype = None

        fake_model = MagicMock()
        fake_model.to.return_value = fake_model
        fake_model.eval.return_value = fake_model

        with patch.object(handler, "_sanitize_torch_runtime_state") as sanitize_mock, patch(
            "acestep.llm_inference.AutoModelForCausalLM.from_pretrained",
            return_value=fake_model,
        ):
            ok, _msg = handler._load_pytorch_model("dummy_model", "cuda")

        self.assertTrue(ok)
        sanitize_mock.assert_called_once()

    def test_generate_from_formatted_prompt_sanitizes_before_pt_run(self):
        """PT generation should sanitize runtime state before entering custom decode loop."""
        handler = LLMHandler()
        handler.llm_initialized = True
        handler.llm_backend = "pt"
        handler.llm = object()
        handler.llm_tokenizer = object()

        with patch.object(handler, "_sanitize_torch_runtime_state") as sanitize_mock, patch.object(
            handler,
            "_run_pt",
            return_value="ok",
        ) as run_pt_mock:
            text, status = handler.generate_from_formatted_prompt("prompt")

        self.assertEqual(text, "ok")
        self.assertIn("Generated successfully (pt)", status)
        sanitize_mock.assert_called_once()
        run_pt_mock.assert_called_once()

    def test_sample_tokens_falls_back_to_cpu_on_capture_offset_error(self):
        """Sampling should recover via CPU multinomial on residual graph-capture RNG errors."""
        handler = LLMHandler()
        logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)

        original_multinomial = torch.multinomial
        calls = {"n": 0}

        def _multinomial_side_effect(input_tensor, num_samples, replacement=False, *, generator=None, out=None):
            _ = (replacement, generator, out)
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("Offset increment outside graph capture encountered unexpectedly.")
            return original_multinomial(input_tensor, num_samples)

        with patch("torch.multinomial", side_effect=_multinomial_side_effect):
            sampled = handler._sample_tokens(logits, temperature=1.0)

        self.assertEqual(sampled.shape, torch.Size([1]))
        self.assertGreaterEqual(calls["n"], 2)
        self.assertTrue(getattr(handler, "_pt_cpu_sampling_fallback_logged", False))

    def test_release_gpu_for_dit_offloads_pt_model(self):
        """release_gpu_for_dit should move PT model to CPU and clear CUDA cache."""
        handler = LLMHandler()
        handler.llm_backend = "pt"

        fake_model = MagicMock()
        fake_model.parameters.return_value = iter(
            [SimpleNamespace(device=SimpleNamespace(type="cuda"))]
        )
        handler.llm = fake_model

        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.empty_cache") as empty_mock, patch(
            "torch.cuda.synchronize"
        ) as sync_mock:
            handler.release_gpu_for_dit(reason="test")

        fake_model.to.assert_called_once_with("cpu")
        empty_mock.assert_called_once_with()
        sync_mock.assert_called_once_with()

    def test_release_gpu_for_dit_noop_for_non_pt_backend(self):
        """release_gpu_for_dit should no-op when backend is not PyTorch."""
        handler = LLMHandler()
        handler.llm_backend = "vllm"
        handler.llm = MagicMock()

        handler.release_gpu_for_dit(reason="test")

        handler.llm.to.assert_not_called()


if __name__ == "__main__":
    unittest.main()
