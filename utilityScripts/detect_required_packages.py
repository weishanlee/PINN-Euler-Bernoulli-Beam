#!/usr/bin/env python3
"""
detect_required_packages.py

Scans all section .tex files for LaTeX commands that require specific packages,
compares with main.tex preamble, and reports missing packages.

Usage:
    python detect_required_packages.py
    
Output:
    - package_requirements_report.json: Detailed report of all findings
    - missing_packages.txt: List of packages that need to be added to main.tex
"""

import re
import json
import os
import glob
from collections import defaultdict
from typing import Dict, Set, List, Tuple

# Mapping of LaTeX commands/environments to required packages
PACKAGE_REQUIREMENTS = {
    # Graphics and Figures
    r'\\includegraphics': 'graphicx',
    r'\\graphicspath': 'graphicx',
    r'\\begin{subfigure}': 'subcaption',
    r'\\subfigure': 'subfig',  # Alternative to subcaption
    r'\\subfloat': 'subfig',
    r'\\begin{tikzpicture}': 'tikz',
    r'\\node': 'tikz',
    r'\\draw': 'tikz',
    r'\\includepdf': 'pdfpages',
    
    # Mathematics
    r'\\begin{align}': 'amsmath',
    r'\\begin{align\*}': 'amsmath',
    r'\\begin{equation\*}': 'amsmath',
    r'\\begin{gather}': 'amsmath',
    r'\\begin{multline}': 'amsmath',
    r'\\begin{split}': 'amsmath',
    r'\\begin{cases}': 'amsmath',
    r'\\begin{pmatrix}': 'amsmath',
    r'\\begin{bmatrix}': 'amsmath',
    r'\\begin{vmatrix}': 'amsmath',
    r'\\DeclareMathOperator': 'amsmath',
    r'\\binom': 'amsmath',
    r'\\overset': 'amsmath',
    r'\\underset': 'amsmath',
    r'\\mathbb': 'amssymb',
    r'\\mathcal': 'amssymb',
    r'\\mathfrak': 'amssymb',
    
    # Algorithms
    r'\\begin{algorithm}': 'algorithm',
    r'\\begin{algorithmic}': 'algorithmic',
    r'\\STATE': 'algorithmic',
    r'\\IF': 'algorithmic',
    r'\\FOR': 'algorithmic',
    r'\\WHILE': 'algorithmic',
    r'\\Require': 'algorithmic',
    r'\\Ensure': 'algorithmic',
    r'\\begin{lstlisting}': 'listings',
    r'\\lstinline': 'listings',
    r'\\mint': 'minted',
    r'\\begin{minted}': 'minted',
    
    # Tables
    r'\\toprule': 'booktabs',
    r'\\midrule': 'booktabs',
    r'\\bottomrule': 'booktabs',
    r'\\cmidrule': 'booktabs',
    r'\\multirow': 'multirow',
    r'\\begin{tabularx}': 'tabularx',
    r'\\begin{longtable}': 'longtable',
    
    # Lists and Formatting
    r'\\begin{enumerate}\[': 'enumitem',
    r'\\begin{itemize}\[': 'enumitem',
    r'\\textcolor': 'xcolor',
    r'\\color': 'xcolor',
    r'\\colorbox': 'xcolor',
    r'\\href': 'hyperref',
    r'\\url': 'hyperref',
    r'\\autoref': 'hyperref',
    
    # Citations (Advanced)
    r'\\citet': 'natbib',
    r'\\citep': 'natbib',
    r'\\citeauthor': 'natbib',
    r'\\citeyear': 'natbib',
    r'\\printbibliography': 'biblatex',
    
    # Other
    r'\\SI{': 'siunitx',
    r'\\si{': 'siunitx',
    r'\\num{': 'siunitx',
    r'\\cancel': 'cancel',
    r'\\sout': 'ulem',
    r'\\uline': 'ulem',
}

# Special package dependencies
PACKAGE_DEPENDENCIES = {
    'subcaption': ['caption'],  # subcaption requires caption
    'tikz': ['pgf'],           # tikz is built on pgf
}

# Packages that might conflict
PACKAGE_CONFLICTS = {
    'subfig': ['subcaption'],  # Use either subfig OR subcaption, not both
}


def scan_tex_file(filepath: str) -> Tuple[Set[str], Dict[str, List[str]], List[str]]:
    """
    Scan a .tex file for required packages.
    
    Returns:
        - Set of required packages
        - Dictionary mapping packages to the commands found
        - List of package comments found in the file
    """
    required_packages = set()
    command_usage = defaultdict(list)
    package_comments = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract package requirement comments
        comment_pattern = r'%\s*Package requirements.*?(?=\n(?!%))'
        comment_matches = re.findall(comment_pattern, content, re.DOTALL | re.IGNORECASE)
        for match in comment_matches:
            package_comments.extend(re.findall(r'-\s*(\w+)', match))
            
        # Check for each LaTeX command/environment
        for pattern, package in PACKAGE_REQUIREMENTS.items():
            if re.search(pattern, content):
                required_packages.add(package)
                # Find actual usage for reporting
                matches = re.findall(pattern + r'.*', content)
                if matches:
                    command_usage[package].append(f"{pattern} (found {len(matches)} times)")
                    
        # Add dependencies
        for package in list(required_packages):
            if package in PACKAGE_DEPENDENCIES:
                for dep in PACKAGE_DEPENDENCIES[package]:
                    required_packages.add(dep)
                    
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        
    return required_packages, command_usage, package_comments


def extract_packages_from_main(main_tex_path: str) -> Set[str]:
    """Extract packages already included in main.tex"""
    packages = set()
    
    try:
        with open(main_tex_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract \usepackage commands
        # Handle both \usepackage{package} and \usepackage[options]{package}
        package_pattern = r'\\usepackage(?:\[.*?\])?\{([^}]+)\}'
        matches = re.findall(package_pattern, content)
        
        for match in matches:
            # Handle multiple packages in one command: \usepackage{pack1,pack2}
            for pkg in match.split(','):
                packages.add(pkg.strip())
                
    except Exception as e:
        print(f"Error reading main.tex: {e}")
        
    return packages


def check_conflicts(required_packages: Set[str]) -> List[str]:
    """Check for package conflicts"""
    conflicts = []
    
    for package in required_packages:
        if package in PACKAGE_CONFLICTS:
            conflicting = [p for p in PACKAGE_CONFLICTS[package] if p in required_packages]
            if conflicting:
                conflicts.append(f"{package} conflicts with {', '.join(conflicting)}")
                
    return conflicts


def main():
    """Main function to detect required packages"""
    
    # Find all section .tex files
    section_files = []
    patterns = [
        'introduction*.tex', 'methods*.tex', 'results*.tex', 
        'resultsAndDiscussions*.tex', 'conclusions*.tex',
        'abstract*.tex', 'summary*.tex', 'letter*.tex',
        'appendix*.tex'
    ]
    
    for pattern in patterns:
        section_files.extend(glob.glob(pattern))
        
    print(f"Found {len(section_files)} section files to analyze")
    
    # Scan each file
    all_required = set()
    file_reports = {}
    documented_packages = set()
    
    for filepath in section_files:
        packages, usage, comments = scan_tex_file(filepath)
        all_required.update(packages)
        documented_packages.update(comments)
        
        file_reports[filepath] = {
            'required_packages': list(packages),
            'command_usage': dict(usage),
            'documented_packages': comments,
            'missing_documentation': list(packages - set(comments))
        }
        
    # Check main.tex
    main_packages = set()
    if os.path.exists('main.tex'):
        main_packages = extract_packages_from_main('main.tex')
    else:
        print("Warning: main.tex not found")
        
    # Find missing packages
    missing_packages = all_required - main_packages
    
    # Check for conflicts
    conflicts = check_conflicts(all_required)
    
    # Generate report
    report = {
        'total_packages_required': len(all_required),
        'packages_in_main': len(main_packages),
        'missing_packages': list(missing_packages),
        'all_required_packages': sorted(list(all_required)),
        'documented_packages': sorted(list(documented_packages)),
        'undocumented_packages': sorted(list(all_required - documented_packages)),
        'conflicts': conflicts,
        'file_analysis': file_reports,
        'recommendations': []
    }
    
    # Add recommendations
    if missing_packages:
        report['recommendations'].append(
            f"Add these packages to main.tex preamble: {', '.join(sorted(missing_packages))}"
        )
        
    if 'graphicx' in all_required and 'graphicx' not in main_packages:
        report['recommendations'].append(
            "Don't forget to add \\graphicspath{{figures/}} after loading graphicx"
        )
        
    if conflicts:
        report['recommendations'].append(
            f"Resolve package conflicts: {'; '.join(conflicts)}"
        )
        
    if all_required - documented_packages:
        report['recommendations'].append(
            "Some required packages are not documented in section files. "
            "Add package requirement comments to improve maintainability."
        )
    
    # Save report
    with open('package_requirements_report.json', 'w') as f:
        json.dump(report, f, indent=2)
        
    # Save missing packages list
    if missing_packages:
        with open('missing_packages.txt', 'w') as f:
            f.write("Missing packages that need to be added to main.tex:\n\n")
            for pkg in sorted(missing_packages):
                f.write(f"\\usepackage{{{pkg}}}\n")
            if 'graphicx' in missing_packages:
                f.write("\\graphicspath{{figures/}}  % Add after graphicx\n")
                
        print(f"\nWARNING: {len(missing_packages)} missing packages detected!")
        print("See missing_packages.txt for the list.")
    else:
        print("\n✓ All required packages are included in main.tex")
        if os.path.exists('missing_packages.txt'):
            os.remove('missing_packages.txt')
            
    # Print summary
    print(f"\nPackage Analysis Summary:")
    print(f"- Total packages required: {len(all_required)}")
    print(f"- Packages in main.tex: {len(main_packages)}")
    print(f"- Missing packages: {len(missing_packages)}")
    print(f"- Package conflicts: {len(conflicts)}")
    print(f"\nDetailed report saved to package_requirements_report.json")
    
    return len(missing_packages) == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)