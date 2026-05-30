import sys
from pathlib import Path

# Make the project root importable in all test files
sys.path.insert(0, str(Path(__file__).parent.parent))