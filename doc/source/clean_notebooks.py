import json
import sys
from pathlib import Path

def clean_notebook(file_path):
    """Removes the 'id' field from all cells in a Jupyter notebook."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        changes_made = False
        if 'cells' in notebook and isinstance(notebook['cells'], list):
            for cell in notebook['cells']:
                if isinstance(cell, dict) and 'id' in cell:
                    del cell['id']
                    changes_made = True

        if changes_made:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1, ensure_ascii=False)
                f.write('\n') # Add a newline at the end of the file
            print(f"Cleaned: {file_path}")
        else:
            print(f"No changes needed: {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python clean_notebooks.py <directory>")
        sys.exit(1)

    target_dir = Path(sys.argv[1])
    if not target_dir.is_dir():
        print(f"Error: {target_dir} is not a valid directory.")
        sys.exit(1)

    print(f"Searching for notebooks in {target_dir}...")
    notebook_files = list(target_dir.rglob('*.ipynb'))

    if not notebook_files:
        print("No notebook files found.")
        return

    for notebook_file in notebook_files:
        clean_notebook(notebook_file)

if __name__ == "__main__":
    main()
