import os

def print_file_contents(file_path):
    with open(file_path, 'r') as file:
        print(f"\n### {file_path} ###\n")
        print(file.read())

def print_all_code(base_directory):
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print_file_contents(file_path)

def print_directory_structure(base_directory):
    for root, dirs, files in os.walk(base_directory):
        level = root.replace(base_directory, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        sub_indent = ' ' * 4 * (level + 1)
        for file in files:
            print(f'{sub_indent}{file}')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Utility script to print code and directory structure of the project.')
    parser.add_argument('--print-code', action='store_true', help='Print the code of all Python files in the project.')
    parser.add_argument('--print-structure', action='store_true', help='Print the directory structure of the project.')
    
    args = parser.parse_args()
    
    if args.print_code or args.print_structure:
        base_directory = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(base_directory, '..', '..'))
        
        if args.print_structure:
            print("\n### Project Directory Structure ###\n")
            print_directory_structure(project_root)
        
        if args.print_code:
            print("\n### Project Code ###\n")
            print_all_code(project_root)