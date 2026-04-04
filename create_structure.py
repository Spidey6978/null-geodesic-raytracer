import os

# The Architecture Tree
structure = {
    "core": ["__init__.py", "constants.py", "geodesics.py", "utils.py"],
    "api": ["__init__.py", "main.py", "routes.py"],
    "workers": ["__init__.py", "celery_app.py", "tasks.py"],
    "notebooks": ["prototyping.ipynb"],
    "tests": ["test_physics.py", "test_api.py"],
    "output": [], # Empty folder for renders
}

def create_project():
    base_path = os.getcwd()
    print(f"🔨 Building project structure in: {base_path}")

    for folder, files in structure.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"   📂 Created: {folder}/")
        
        for file in files:
            file_path = os.path.join(folder_path, file)
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    # Add a docstring so files aren't completely empty
                    f.write(f'"""\nModule: {folder}.{file.replace(".py", "")}\n"""\n')
                print(f"      📄 Created: {file}")
            else:
                print(f"      ⚠️  Exists: {file}")

    # Create root level files
    open(".env", "a").close()
    open(".gitignore", "a").close()
    print("✅ Project Structure Complete.")

if __name__ == "__main__":
    create_project()