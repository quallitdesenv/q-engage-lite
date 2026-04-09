"""
Pytest configuration and shared fixtures for Q-Engage-Lite tests.
"""
import pytest
import warnings
import cv2


def pytest_configure(config):
    """Configure pytest."""
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="ultralytics")


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add markers
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session", autouse=True)
def setup_opencv():
    """Setup OpenCV for headless testing."""
    # Disable GUI for all tests
    cv2.setNumThreads(1)
    yield
    cv2.destroyAllWindows()


@pytest.fixture(scope="session")
def test_markers():
    """Return available test markers."""
    return {
        "integration": "Integration tests that test multiple components",
        "slow": "Tests that may take longer to execute",
        "unit": "Unit tests for individual components"
    }
