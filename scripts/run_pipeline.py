from pathlib import Path
import sys

# Add the project root so this script can import src.models.train.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train import main as train_model_main


if __name__ == "__main__":
    train_model_main()