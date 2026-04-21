"""Tests for topaz_device normalization + argv behavior."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from relion.topaz_runner import (
    TopazDeviceValueError,
    build_topaz_extract_argv,
    normalize_topaz_device,
)


class TestNormalizeTopazDevice(unittest.TestCase):
    def test_int_passthrough(self) -> None:
        self.assertEqual(normalize_topaz_device(0), 0)
        self.assertEqual(normalize_topaz_device(1), 1)
        self.assertEqual(normalize_topaz_device(-1), -1)

    def test_numeric_string_is_coerced(self) -> None:
        self.assertEqual(normalize_topaz_device("0"), 0)
        self.assertEqual(normalize_topaz_device(" 2 "), 2)

    def test_none_and_empty_string_are_none(self) -> None:
        self.assertIsNone(normalize_topaz_device(None))
        self.assertIsNone(normalize_topaz_device(""))
        self.assertIsNone(normalize_topaz_device("   "))

    def test_cpu_variants_are_none(self) -> None:
        self.assertIsNone(normalize_topaz_device("cpu"))
        self.assertIsNone(normalize_topaz_device("CPU"))
        self.assertIsNone(normalize_topaz_device("Cpu"))

    def test_bool_is_rejected(self) -> None:
        with self.assertRaises(TopazDeviceValueError):
            normalize_topaz_device(True)
        with self.assertRaises(TopazDeviceValueError):
            normalize_topaz_device(False)

    def test_bogus_string_raises_clean_error(self) -> None:
        with self.assertRaises(TopazDeviceValueError) as ctx:
            normalize_topaz_device("gpu0")
        self.assertIn("topaz_device", str(ctx.exception))

        with self.assertRaises(TopazDeviceValueError):
            normalize_topaz_device("abc")

    def test_integer_float_allowed(self) -> None:
        self.assertEqual(normalize_topaz_device(1.0), 1)

    def test_nonintegral_float_raises(self) -> None:
        with self.assertRaises(TopazDeviceValueError):
            normalize_topaz_device(1.5)


def _star_with_one_micrograph(root: Path) -> Path:
    star = root / "micrographs.star"
    (root / "m1.mrc").write_text("img", encoding="utf-8")
    star.write_text(
        "data_\nloop_\n_rlnMicrographName #1\nm1.mrc\n",
        encoding="utf-8",
    )
    return star


class TestTopazArgvDeviceFlag(unittest.TestCase):
    def test_device_zero_emits_flag(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            star = _star_with_one_micrograph(root)
            argv, _out, _meta = build_topaz_extract_argv(
                {
                    "micrographs_star_path": str(star),
                    "output_dir": str(root / "out"),
                    "particle_diameter_angstrom": 200,
                    "topaz_device": 0,
                },
                {},
                {"executable_path": "/usr/bin/topaz"},
            )
            self.assertIn("-d", argv)
            d_idx = argv.index("-d")
            self.assertEqual(argv[d_idx + 1], "0")

    def test_device_none_omits_flag(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            star = _star_with_one_micrograph(root)
            argv, _out, _meta = build_topaz_extract_argv(
                {
                    "micrographs_star_path": str(star),
                    "output_dir": str(root / "out"),
                    "particle_diameter_angstrom": 200,
                    "topaz_device": None,
                },
                {},
                {"executable_path": "/usr/bin/topaz"},
            )
            self.assertNotIn("-d", argv)

    def test_config_default_cpu_does_not_leak_dash_d(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            star = _star_with_one_micrograph(root)
            argv, _out, _meta = build_topaz_extract_argv(
                {
                    "micrographs_star_path": str(star),
                    "output_dir": str(root / "out"),
                    "particle_diameter_angstrom": 200,
                },
                {"default_device": "CPU"},
                {"executable_path": "/usr/bin/topaz"},
            )
            # Previously "CPU" would be passed to -d and topaz argparse would
            # reject it with ``invalid int value: 'CPU'``.  Now it becomes
            # None and -d is omitted entirely.
            self.assertNotIn("-d", argv)

    def test_device_string_zero_is_coerced(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            star = _star_with_one_micrograph(root)
            argv, _out, _meta = build_topaz_extract_argv(
                {
                    "micrographs_star_path": str(star),
                    "output_dir": str(root / "out"),
                    "particle_diameter_angstrom": 200,
                    "topaz_device": "0",
                },
                {},
                {"executable_path": "/usr/bin/topaz"},
            )
            self.assertIn("-d", argv)
            d_idx = argv.index("-d")
            self.assertEqual(argv[d_idx + 1], "0")


if __name__ == "__main__":
    unittest.main()
