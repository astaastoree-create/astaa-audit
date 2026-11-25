import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apps", "worker", "src"))

import mistralai.workflows as workflows

from worker.workflows.astaa_audit import AUDIT_INSTRUCTIONS, AstaaAuditWorkflow


def test_discovery_finds_audit_workflow():
    classes = workflows.discover_all_workflows_in_package("worker.workflows")
    names = {cls.__name__ for cls in classes}
    assert "AstaaAuditWorkflow" in names


def test_workflow_declares_connector_slot():
    assert AstaaAuditWorkflow.__module__ == "worker.workflows.astaa_audit"


def test_instructions_forbid_completing_checkout():
    lowered = AUDIT_INSTRUCTIONS.lower()
    assert "never call complete_checkout" in lowered
    assert "never" in lowered
