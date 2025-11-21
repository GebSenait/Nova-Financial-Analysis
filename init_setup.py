import os
from pathlib import Path

# Define the project structure based on Week 1 requirements
project_structure = [
    ".vscode/settings.json",
    ".github/workflows/unittests.yml",
    ".gitignore",
    "requirements.txt",
    "README.md",
    "src/__init__.py",
    "notebooks/__init__.py",
    "notebooks/README.md",
    "tests/__init__.py",
    "scripts/__init__.py",
    "scripts/README.md",
    "data/raw/.gitkeep",         # Added for storing the FNSPID dataset (ignored by git usually)
    "data/processed/.gitkeep",
]

def create_project_structure(file_list):
    print("🚀 Starting Project Structure Setup...")
    
    for file_path in file_list:
        path = Path(file_path)
        
        # Create directories if they don't exist
        if path.parent != Path('.'):
            path.parent.mkdir(parents=True, exist_ok=True)
            
        # Create file if it doesn't exist
        if not path.exists():
            path.touch()
            print(f"✅ Created: {path}")
        else:
            print(f"ℹ️  Already exists: {path}")

    # --- specialized content injection ---
    
    # 1. Populate .gitignore with best practices
    gitignore_content = """
# Python basics
__pycache__/
*.py[cod]
*$py.class
.env
.venv/
venv/

# Jupyter Notebooks
.ipynb_checkpoints

# Data (Never commit large datasets)
data/
!data/**/*.gitkeep

# IDE settings
.vscode/
.idea/
"""
    with open(".gitignore", "w") as f:
        f.write(gitignore_content.strip())
    print("✅ Populated: .gitignore")

    # 2. specific README for context
    readme_content = "# Nova Financial Solutions - Stock Prediction Project\n\n## Week 1 Challenge\n\nProject structure initialized."
    with open("README.md", "w") as f:
        f.write(readme_content)
    print("✅ Populated: Root README.md")

    print("\n🎉 Project setup complete! You are ready to initialize Git.")

if __name__ == "__main__":
    create_project_structure(project_structure)