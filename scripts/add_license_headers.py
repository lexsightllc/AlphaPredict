#!/usr/bin/env python3

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Add MPL-2.0 license headers to Python files.
"""
import os
import re
from pathlib import Path

LICENSE_HEADER = """# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

def should_skip_file(filepath):
    """Check if file should be skipped."""
    skip_dirs = ['venv', '.git', '__pycache__', 'build', 'dist']
    skip_files = ['__init__.py']
    
    if any(skip_dir in str(filepath) for skip_dir in skip_dirs):
        return True
    if filepath.name in skip_files:
        return True
    return False

def has_license_header(content):
    """Check if file already has an MPL-2.0 header."""
    return "Mozilla Public License" in content[:500]

def add_header_to_file(filepath):
    """Add license header to a single file."""
    try:
        with open(filepath, 'r+', encoding='utf-8') as f:
            content = f.read()
            
            if has_license_header(content):
                print(f"Skipping (has header): {filepath}")
                return
                
            # Preserve shebang if present
            lines = content.splitlines(keepends=True)
            if lines and lines[0].startswith('#!'):
                new_content = lines[0] + '\n' + LICENSE_HEADER + ''.join(lines[1:])
            else:
                new_content = LICENSE_HEADER + '\n' + content
                
            f.seek(0)
            f.write(new_content)
            f.truncate()
            print(f"Added header to: {filepath}")
            
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """Main function to process all Python files."""
    project_root = Path(__file__).parent.parent
    python_files = project_root.glob('**/*.py')
    
    for py_file in python_files:
        if not should_skip_file(py_file):
            add_header_to_file(py_file)

if __name__ == "__main__":
    main()
