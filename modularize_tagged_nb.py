import os
import nbformat
from pathlib import Path
from nbformat import read
import ast
import astunparse  # If using Python <3.9. For 3.9+, use ast.unparse()
from typing import List, Dict, Set
import argparse


def extract_symbols_from_import_line(import_line: str) -> list:
    """
    Extract all symbols introduced into the namespace by an import line.

    Examples:
        'import pandas as pd' → ['pd']
        'import os' → ['os']
        'from sklearn.model_selection import train_test_split' → ['train_test_split']
        'from sklearn.model_selection import GridSearchCV, KFold, cross_val_score'
            → ['GridSearchCV', 'KFold', 'cross_val_score']
    """
    if import_line.startswith("import"):
        parts = import_line.split(',')
        symbols = []
        for part in parts:
            sub_parts = part.strip().split()
            if "as" in sub_parts:
                symbols.append(sub_parts[-1])
            else:
                symbols.append(sub_parts[1].split('.')[0])
        return symbols

    elif import_line.startswith("from"):
        try:
            imports = import_line.split("import", 1)[1]
            symbols = [sym.strip().split(" as ")[-1] for sym in imports.split(',')]
            return symbols
        except IndexError:
            return []
    return []

def extract_imports_from_code(code):
    return [line.strip() for line in code.splitlines() if line.strip().startswith(('import', 'from'))]

def extract_required_packages(import_lines):
    packages = set()
    for line in import_lines:
        line = line.strip()
        if line.startswith("import"):
            packages.add(line.split()[1].split('.')[0])
        elif line.startswith("from"):
            packages.add(line.split()[1].split('.')[0])
    return sorted(packages)

def write_module(path, code):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(code + '\n\n')

def write_separation(tag):
    with open("ml_project/main.py", 'a', encoding='utf-8') as f:
        f.write("\n# " + tag + '\n')

def write_main_call(function_call):
    with open("ml_project/main.py", 'a', encoding='utf-8') as f:
        f.write(function_call + '\n')

def write_import(path, import_line):
    """Prepend imports to a given file."""
    existing = Path(path).read_text(encoding='utf-8') if Path(path).exists() else ""
    if import_line not in existing:
        with open(path, 'r+', encoding='utf-8') as f:
            content = f.read()
            f.seek(0)
            f.write(import_line + '\n' + content)

def most_prioritized_tag(tags, priority_order):
    for tag in priority_order:
        if tag in tags:
            return tag
    return None

def extract_code_parts(cell_code: str) -> Dict[str, List[str]]:
    tree = ast.parse(cell_code)
    
    functions = []
    classes = []
    other_code_nodes = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions.append(node)
        elif isinstance(node, ast.ClassDef):
            classes.append(node)
        else:
            other_code_nodes.append(node)

    def nodes_to_code(nodes):
        return [ast.unparse(n) if hasattr(ast, 'unparse') else astunparse.unparse(n).strip() for n in nodes]

    return {
        "functions": nodes_to_code(functions),
        "classes": nodes_to_code(classes),
        "other_code": nodes_to_code(other_code_nodes)
    }

def extract_function_names_from_code(code: str) -> Set[str]:
    """Extracts all function names used in code via ast.Call"""
    tree = ast.parse(code)
    return {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}

def extract_symbols_from_code(code: str) -> Set[str]:
    tree = ast.parse(code)
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

def modularize_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    tag_file_mapping = {
        'process_data': 'src/features.py',
        'train_model': 'src/train.py',
        'evaluate_model': 'src/evaluate.py',
        'visualize_data': 'src/visualize.py',
        'transfer_results': 'src/transfer.py', 
        'ingest_data': 'src/data_loader.py'
    }

    ignored_tags = {'setup_notebook', 'validate_data'}
    priority_order = ['train_model', 'process_data', 'ingest_data', 'evaluate_model', 'transfer_results', 'visualize_data']

    os.makedirs("ml_project/src", exist_ok=True)
    os.makedirs("ml_project/data", exist_ok=True)
    Path("ml_project/main.py").write_text("# Auto-generated pipeline entry point\n\n")

    # Collect setup imports
    requirements_imports = set()
    all_imports = set()
    function_defs = {}  # filename -> [func_name]
    function_uses = {}  # filename -> [func_name]
    module_code = {}
    written_files = set()
    written_files.add("ml_project/main.py")

    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue

        tags = cell.metadata.get('tags', [])
        code = cell.source.strip()

        if 'setup_notebook' in tags:
            imports = extract_imports_from_code(code)
            all_imports.update(imports)
            requirements_imports.update(imports)
            if tags == ['setup_notebook']:
                continue

        symbol_to_import = {}
        for imp in all_imports:
            symbols = extract_symbols_from_import_line(imp)
            for sym in symbols:
                symbol_to_import[sym] = imp

        primary_tag = most_prioritized_tag(tags, priority_order)
        if primary_tag is None or primary_tag in ignored_tags:
            continue

        if primary_tag in tag_file_mapping:
            file_path = f"ml_project/{tag_file_mapping[primary_tag]}"
            parsed = extract_code_parts(cell.source)

            defined_funcs = []

            for func in parsed['functions'] + parsed['classes']:
                name_tree = ast.parse(func)
                if isinstance(name_tree.body[0], (ast.FunctionDef, ast.ClassDef)):
                    defined_funcs.append(name_tree.body[0].name)
                    write_module(file_path, func)
                    written_files.add(file_path)

            function_defs[file_path] = function_defs.get(file_path, []) + defined_funcs
            module_code[file_path] = module_code.get(file_path, '') + '\n' + code

            if parsed['other_code']:
                write_separation(primary_tag)
            for call in parsed['other_code']:
                write_main_call(call)
                # Track usage for inter-module imports
                func_names = extract_function_names_from_code(call)
                function_uses['ml_project/main.py'] = function_uses.get('ml_project/main.py', []) + list(func_names)
                # Track the actual code content in main.py to detect used imports
                module_code['ml_project/main.py'] = module_code.get('ml_project/main.py', '') + '\n' + call

    # Add precise imports to only the files that use them
    for path in written_files:
        code = module_code.get(path, '')
        used_symbols = extract_symbols_from_code(code)
        for symbol in used_symbols:
            if symbol in symbol_to_import:
                write_import(path, symbol_to_import[symbol])

    # Add inter-module imports
    for target_file, used_funcs in function_uses.items():
        for used in used_funcs:
            for file, defined in function_defs.items():
                if used in defined and file != target_file:
                    module = Path(file).stem
                    rel_path = file.replace("ml_project/", "").replace("/", ".").replace(".py", "")
                    import_line = f"from {rel_path} import {used}"
                    write_import(target_file, import_line)

    # Requirements
    requirements = extract_required_packages(requirements_imports)
    with open('ml_project/requirements.txt', 'w') as f:
        f.write('\n'.join(requirements) + '\n')

    print("✅ Project modularization complete. Files saved in ml_project/")


# Constant base directory
home = os.getcwd()
BASE_DIR = Path(home) / "jupylab_cli/src/pipeline_analyzer/jn_analyzer/resources/outputs"

def build_notebook_path(notebook_name: str) -> Path:
    notebook_stem = Path(notebook_name).stem  # removes .ipynb if present
    labeled_name = f"{notebook_stem}._nlp_labeled.ipynb"
    return BASE_DIR / labeled_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modularize a labeled Jupyter notebook by name.")
    parser.add_argument(
        "notebook_name",
        type=str,
        help="Original notebook name"
    )
    args = parser.parse_args()

    notebook_path = build_notebook_path(args.notebook_name)

    if not notebook_path.exists():
        print(f"❌ Error: Notebook file '{notebook_path}' does not exist.")
    else:
        print(f"✅ Processing notebook at: {notebook_path}")
        modularize_notebook(str(notebook_path))