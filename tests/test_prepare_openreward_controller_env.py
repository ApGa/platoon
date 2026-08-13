from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "pbs-scripts" / "prepare_openreward_controller_env.sh"


class ControllerEnvironmentHelperTests(unittest.TestCase):
    def test_validate_failure_and_success_are_safe_under_nounset(self) -> None:
        # Execute the function definitions exactly as Bash sees them while
        # omitting only the helper's command dispatcher.  The missing-cache
        # branch caught a same-declaration local expansion that `set -u`
        # rejects before validation can return false.
        definitions, separator, _ = HELPER.read_text().partition('\ncase "${1:-}" in\n')
        self.assertTrue(separator, "controller helper command dispatcher not found")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = root / "environment"
            manifest = root / "manifest"
            script = (
                definitions
                + "\n"
                + f"environment={environment!s}\n"
                + f"manifest={manifest!s}\n"
                + "if validate_controller_environment \"${environment}\" \"${manifest}\"; then\n"
                + "    echo 'missing environment unexpectedly validated' >&2\n"
                + "    exit 10\n"
                + "fi\n"
                + "mkdir -p \"${environment}/bin\"\n"
                + "printf '%s\\n' '#!/bin/sh' 'exit 0' >\"${environment}/bin/python\"\n"
                + "chmod 0755 \"${environment}/bin/python\"\n"
                + "touch \"${environment}/.platoon-controller-ready\" \\\n"
                + "      \"${environment}/.build-complete\"\n"
                + "chmod 0400 \"${environment}/.platoon-controller-ready\" \\\n"
                + "           \"${environment}/.build-complete\"\n"
                + "printf '%s\\n' 'manifest=test' >\"${manifest}\"\n"
                + "cp \"${manifest}\" \"${environment}/.platoon-controller-manifest\"\n"
                + "printf '%s\\n' 'relocatable = true' >\"${environment}/pyvenv.cfg\"\n"
                + "validate_controller_environment \"${environment}\" \"${manifest}\"\n"
            )
            result = subprocess.run(
                ["/bin/bash"],
                input=script,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_builder_publishes_read_only_runtime_markers(self) -> None:
        source = HELPER.read_text()

        assert 'chmod 0400 "${stage_env}/.build-complete"' in source
        assert 'chmod 0400 "${final_env}/.platoon-controller-ready"' in source
        assert 'stat -c \'%a\' "${environment}/.platoon-controller-ready"' in source
        assert 'stat -c \'%a\' "${environment}/.build-complete"' in source


if __name__ == "__main__":
    unittest.main()
