import tempfile
import unittest
from pathlib import Path

from safe.config import apply_overrides, deep_merge, load_config


class ConfigTests(unittest.TestCase):
    def test_deep_merge_preserves_nested_defaults(self):
        merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}})
        self.assertEqual(merged, {"a": {"b": 3, "c": 2}})

    def test_yaml_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("inference:\n  max_steps: 4\n", encoding="utf-8")
            config = apply_overrides(
                load_config(path),
                ["generation.batch_size=2", "models.load_in_4bit=false"],
            )
        self.assertEqual(config["inference"]["max_steps"], 4)
        self.assertEqual(config["generation"]["batch_size"], 2)
        self.assertFalse(config["models"]["load_in_4bit"])


if __name__ == "__main__":
    unittest.main()

