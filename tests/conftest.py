import pytest
import os
import sys
import shutil

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

@pytest.fixture(scope="session", autouse=True)
def cleanup_test_output():
    """Fixture to clean up the test_output directory at the start of a test session."""
    output_dir = os.path.join(project_root, "test_output")
    
    print(f"\nCleaning up directory for test session: {output_dir}")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print("Cleanup complete.")


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return tmp_path 