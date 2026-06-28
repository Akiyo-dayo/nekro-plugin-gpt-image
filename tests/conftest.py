import sys
from pathlib import Path

# Add the plugin root so 'client' and 'presets' can be imported directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))