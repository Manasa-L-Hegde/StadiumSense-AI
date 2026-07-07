import os
import sys
import warnings

# Ensure the root project directory is added to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Suppress google-generativeai deprecation warnings during testing to keep output clean
warnings.filterwarnings("ignore", category=FutureWarning, module=".*generativeai.*")

