#!/usr/bin/env python3
"""Check Dockerfiles for common issues."""

import os
import re
import sys
from pathlib import Path


def find_dockerfiles(repo_root):
    """Find all Dockerfiles."""
    files = []
    for root, dirs, filenames in os.walk(repo_root):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        if 'Dockerfile' in filenames:
            files.append(Path(root) / 'Dockerfile')
    return sorted(files)


def check_file(filepath, repo_root):
    """Validate one Dockerfile."""
    rel_path = filepath.relative_to(repo_root)
    
    try:
        with open(filepath) as f:
            lines = f.read().split('\n')
    except Exception as e:
        return None, [f"{rel_path}: Cannot read - {e}"], []
    
    errors = []
    warnings = []
    
    # Check: Has FROM statement
    has_from = any(re.match(r'^\s*FROM\s+', l, re.I) for l in lines)
    if not has_from:
        errors.append(f"{rel_path}: Missing FROM statement")
    
    # Check: Validate ARG names
    for i, line in enumerate(lines, 1):
        m = re.match(r'^\s*ARG\s+(\w+)', line, re.I)
        if m and not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', m.group(1)):
            errors.append(f"{rel_path}:{i}: Invalid ARG name '{m.group(1)}'")
    
    # Check: RUN with && should have set -e
    has_global_set = any('set -e' in l for l in lines[:10])
    for i, line in enumerate(lines, 1):
        if re.match(r'^\s*RUN\s+', line) and '&&' in line and not has_global_set:
            if 'set -e' not in line:
                warnings.append(f"{rel_path}:{i}: RUN with && should use 'set -e'")
                break
    
    # Check: :latest tags
    for i, line in enumerate(lines, 1):
        if ':latest' in line and not line.strip().startswith('#'):
            warnings.append(f"{rel_path}:{i}: Avoid ':latest' tag")
    
    # Extract info
    images = []
    args = {}
    
    for line in lines:
        m = re.match(r'^\s*FROM\s+(.*?)\s*(?:as\s+\w+)?$', line, re.I)
        if m:
            images.append(m.group(1).strip())
        
        m = re.match(r'^\s*ARG\s+(\w+)=(.+)', line)
        if m:
            name, val = m.group(1), m.group(2).strip().strip('"\'')
            if any(k in name.lower() for k in ['cuda', 'torch', 'python', 'ubuntu']):
                args[name] = val
    
    info = f"✓ {rel_path}: {', '.join(images) if images else 'no FROM'}"
    if args:
        info += f" | Args: {args}"
    
    return info, errors, warnings


def main():
    repo_root = Path(__file__).parent.parent
    dockerfiles = find_dockerfiles(repo_root)
    
    if not dockerfiles:
        print("❌ No Dockerfiles found")
        return 1
    
    print(f"📋 Found {len(dockerfiles)} Dockerfile(s)\n")
    
    all_info = []
    all_errors = []
    all_warnings = []
    
    for dockerfile in dockerfiles:
        info, errors, warnings = check_file(dockerfile, repo_root)
        if info:
            all_info.append(info)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
    
    print("=" * 70)
    
    if all_info:
        print("\n📊 Images:")
        for item in all_info:
            print(f"  {item}")
    
    if all_warnings:
        print(f"\n⚠️  Warnings ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"  {w}")
    
    if all_errors:
        print(f"\n❌ Errors ({len(all_errors)}):")
        for e in all_errors:
            print(f"  {e}")
    else:
        print("\n✅ All Dockerfiles valid!")
    
    print("=" * 70)
    
    return 1 if all_errors else 0


if __name__ == '__main__':
    sys.exit(main())
