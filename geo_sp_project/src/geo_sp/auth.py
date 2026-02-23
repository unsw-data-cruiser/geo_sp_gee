"""
src/geo_sp/auth.py
Earth Engine authentication helper
"""
import ee


def init_ee(project: str) -> None:
    """Initialize (and authenticate if needed) Earth Engine."""
    try:
        ee.Initialize(project=project)
        print(f"✅ Earth Engine initialized  (project: {project})")
    except Exception:
        print("⚠️  Earth Engine auth required — browser window will open...")
        ee.Authenticate()
        ee.Initialize(project=project)
        print("✅ Earth Engine initialized")
