from __future__ import annotations

import unittest

from biomol_surface_unsup.features.local_features import build_local_features


class LocalFeaturesTestCase(unittest.TestCase):
    def test_build_local_features(self) -> None:
        features = build_local_features({"values": [1.0, 2.0]})
        self.assertEqual(features["count"], 2)


if __name__ == "__main__":
    unittest.main()
