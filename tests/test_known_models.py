"""
tests/test_known_models.py — Unit tests for analyzer/known_models.py.
Verifies structure and content of the known models registry.
"""

import unittest


class TestKnownModels(unittest.TestCase):
    """Tests for KNOWN_MODELS registry structure and content."""

    def setUp(self):
        from analyzer.known_models import KNOWN_MODELS
        self.models = KNOWN_MODELS

    def test_all_providers_present(self):
        for provider in ("anthropic", "openai", "gemini"):
            self.assertIn(provider, self.models, f"Missing provider: {provider}")

    def test_each_provider_has_models(self):
        for provider, models in self.models.items():
            self.assertGreater(len(models), 0, f"{provider} has no models")

    def test_each_model_has_id_and_label(self):
        for provider, models in self.models.items():
            for m in models:
                self.assertIn("id",    m, f"{provider} model missing 'id': {m}")
                self.assertIn("label", m, f"{provider} model missing 'label': {m}")

    def test_model_ids_are_non_empty_strings(self):
        for provider, models in self.models.items():
            for m in models:
                self.assertIsInstance(m["id"], str)
                self.assertTrue(m["id"].strip(), f"{provider} model has empty id")

    def test_labels_contain_cost_indicator(self):
        """Every label should mention cheapest, balanced, best, cheap, expensive, or fast."""
        keywords = {"cheapest", "balanced", "best", "cheap", "expensive", "fast", "powerful", "reasoning"}
        for provider, models in self.models.items():
            for m in models:
                label_words = set(m["label"].lower().split())
                has_indicator = bool(label_words & keywords)
                self.assertTrue(has_indicator,
                    f"{provider}/{m['id']} label has no cost indicator: {m['label']!r}")

    def test_no_duplicate_ids_per_provider(self):
        for provider, models in self.models.items():
            ids = [m["id"] for m in models]
            self.assertEqual(len(ids), len(set(ids)),
                f"{provider} has duplicate model ids: {ids}")

    def test_anthropic_includes_sonnet(self):
        from analyzer.known_models import KNOWN_MODELS
        ids = [m["id"] for m in KNOWN_MODELS["anthropic"]]
        self.assertTrue(any("sonnet" in i for i in ids))

    def test_openai_includes_gpt4o_mini(self):
        from analyzer.known_models import KNOWN_MODELS
        ids = [m["id"] for m in KNOWN_MODELS["openai"]]
        self.assertIn("gpt-4o-mini", ids)

    def test_gemini_includes_flash(self):
        from analyzer.known_models import KNOWN_MODELS
        ids = [m["id"] for m in KNOWN_MODELS["gemini"]]
        self.assertTrue(any("flash" in i for i in ids))


if __name__ == "__main__":
    unittest.main()
