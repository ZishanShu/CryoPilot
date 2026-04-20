"""Tests for agent.experiment_protocol protocol loader/parser."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from agent.experiment_protocol import (
    ExperimentProtocol,
    ProtocolValidationError,
    default_protocols,
    load_protocol,
    load_protocols_from_dir,
    parse_protocol,
    resolve_protocol,
)


class TestParseProtocol(unittest.TestCase):
    def test_parse_fixed_workflow_minimal(self) -> None:
        proto = parse_protocol(
            {
                "protocol_name": "fixed_topaz",
                "protocol_kind": "fixed_workflow",
                "allowed_backends": ["topaz"],
            }
        )
        self.assertIsInstance(proto, ExperimentProtocol)
        self.assertEqual(proto.protocol_kind, "fixed_workflow")
        self.assertEqual(proto.routing_mode, "fixed")
        self.assertFalse(proto.llm_supervision_enabled)
        self.assertEqual(proto.allowed_backends, ["topaz"])

    def test_fixed_workflow_rejects_multiple_backends(self) -> None:
        with self.assertRaises(ProtocolValidationError):
            parse_protocol(
                {
                    "protocol_name": "bad",
                    "protocol_kind": "fixed_workflow",
                    "allowed_backends": ["topaz", "cryolo"],
                }
            )

    def test_llm_supervised_forces_supervision_flag(self) -> None:
        proto = parse_protocol(
            {
                "protocol_name": "llm_x",
                "protocol_kind": "llm_supervised_agent",
                "allowed_backends": ["topaz"],
                "llm_supervision_enabled": False,
            }
        )
        self.assertTrue(proto.llm_supervision_enabled)

    def test_fixed_workflow_rejects_llm_supervision(self) -> None:
        with self.assertRaises(ProtocolValidationError):
            parse_protocol(
                {
                    "protocol_name": "bad",
                    "protocol_kind": "fixed_workflow",
                    "allowed_backends": ["topaz"],
                    "llm_supervision_enabled": True,
                }
            )

    def test_invalid_kind_rejected(self) -> None:
        with self.assertRaises(ProtocolValidationError):
            parse_protocol(
                {
                    "protocol_name": "x",
                    "protocol_kind": "not_a_kind",
                    "allowed_backends": ["topaz"],
                }
            )

    def test_invalid_stop_point_rejected(self) -> None:
        with self.assertRaises(ProtocolValidationError):
            parse_protocol(
                {
                    "protocol_name": "x",
                    "protocol_kind": "rule_only_agent",
                    "allowed_backends": ["topaz"],
                    "stop_points": ["bogus"],
                }
            )

    def test_to_dict_is_json_friendly(self) -> None:
        proto = parse_protocol(
            {
                "protocol_name": "rule_x",
                "protocol_kind": "rule_only_agent",
                "allowed_backends": ["topaz", "cryodrgn"],
            }
        )
        json.dumps(proto.to_dict())  # must be serializable


class TestLoadProtocol(unittest.TestCase):
    def test_load_from_yaml_file(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "protocol_name": "rule_y",
                        "protocol_kind": "rule_only_agent",
                        "allowed_backends": ["topaz"],
                    }
                ),
                encoding="utf-8",
            )
            proto = load_protocol(path)
            self.assertEqual(proto.protocol_name, "rule_y")

    def test_load_from_json_file(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.json"
            path.write_text(
                json.dumps(
                    {
                        "protocol_name": "rule_z",
                        "protocol_kind": "rule_only_agent",
                        "allowed_backends": ["topaz"],
                    }
                ),
                encoding="utf-8",
            )
            proto = load_protocol(path)
            self.assertEqual(proto.protocol_name, "rule_z")

    def test_load_dir_builds_mapping_and_rejects_dups(self) -> None:
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "a.yaml").write_text(
                yaml.safe_dump(
                    {
                        "protocol_name": "a",
                        "protocol_kind": "fixed_workflow",
                        "allowed_backends": ["topaz"],
                    }
                ),
                encoding="utf-8",
            )
            (d / "b.yaml").write_text(
                yaml.safe_dump(
                    {
                        "protocol_name": "b",
                        "protocol_kind": "rule_only_agent",
                        "allowed_backends": ["topaz"],
                    }
                ),
                encoding="utf-8",
            )
            mapping = load_protocols_from_dir(d)
            self.assertEqual(set(mapping), {"a", "b"})


class TestDefaultsAndResolve(unittest.TestCase):
    def test_default_protocols_contain_three_kinds(self) -> None:
        d = default_protocols()
        self.assertIn("fixed_workflow_topaz", d)
        self.assertIn("rule_only_agent", d)
        self.assertIn("llm_supervised_agent", d)
        self.assertEqual(d["fixed_workflow_topaz"].protocol_kind, "fixed_workflow")
        self.assertTrue(d["llm_supervised_agent"].llm_supervision_enabled)

    def test_resolve_protocol_by_name_uses_defaults(self) -> None:
        proto = resolve_protocol("fixed_workflow_topaz")
        self.assertEqual(proto.protocol_kind, "fixed_workflow")

    def test_resolve_protocol_by_search_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "custom.yaml").write_text(
                yaml.safe_dump(
                    {
                        "protocol_name": "custom",
                        "protocol_kind": "rule_only_agent",
                        "allowed_backends": ["topaz"],
                    }
                ),
                encoding="utf-8",
            )
            proto = resolve_protocol("custom", search_dirs=[d])
            self.assertEqual(proto.protocol_name, "custom")

    def test_resolve_protocol_missing_raises(self) -> None:
        with self.assertRaises(ProtocolValidationError):
            resolve_protocol("no_such_protocol_xyz")


if __name__ == "__main__":
    unittest.main()
