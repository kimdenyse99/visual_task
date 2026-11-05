import json
import subprocess
import sys
import unittest


class ManifestTests(unittest.TestCase):
    def test_describe_lists_the_stages_and_their_levels(self):
        output = subprocess.run(
            [sys.executable, "main.py", "--describe"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout
        manifest = json.loads(output)

        self.assertEqual(manifest["name"], "visual_tasks")
        stages = manifest["stages"]
        self.assertEqual(
            [stage["name"] for stage in stages],
            [
                "size_reduction_3pt",
                "visual_stability_training",
                "visual3pt_calibration",
                "visual_tracing",
                "touchscreen_training",
                "target_position_generalization_training",
            ],
        )
        for stage in stages:
            with self.subTest(stage=stage["name"]):
                self.assertTrue(stage["levels"])


if __name__ == "__main__":
    unittest.main()
