"""Build hook to run isort and ruff before building the project."""

import subprocess
import sys

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Custom build hook to format and lint code before building."""

    PLUGIN_NAME = "custom"

    def initialize(self, version, build_data):
        """Run isort and ruff before building."""
        print("Running isort...")
        try:
            subprocess.run(
                [sys.executable, "-m", "isort", "."],
                cwd=self.root,
                check=True,
                capture_output=True,
                text=True,
            )
            print("✓ isort completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ isort failed: {e.stderr}")
            raise

        print("Running ruff format...")
        try:
            subprocess.run(
                [sys.executable, "-m", "ruff", "format", "."],
                cwd=self.root,
                check=True,
                capture_output=True,
                text=True,
            )
            print("✓ ruff format completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ ruff format failed: {e.stderr}")
            raise

        print("Running ruff check...")
        try:
            subprocess.run(
                [sys.executable, "-m", "ruff", "check", ".", "--fix"],
                cwd=self.root,
                check=True,
                capture_output=True,
                text=True,
            )
            print("✓ ruff check completed successfully")
        except subprocess.CalledProcessError as e:
            # Don't fail build on linting errors, just warn
            print(f"⚠ ruff check found issues: {e.stderr}")
