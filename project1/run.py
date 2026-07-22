import sys
import os

# Add src folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import run_pipeline

if __name__ == "__main__":
    print("==================================================================")
    # Highlight the internship project and AI & ML context
    print("Project-1: Summer Internship Program in AI&ML Machine Learning 2026")
    print("AI-Powered Fake News Detection Using Text Classification")
    print("==================================================================")
    
    run_pipeline()
