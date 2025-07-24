#!/usr/bin/env python3
"""
Extract citations used in introduction.tex and create introduction_refs.bib
"""

import re
from pathlib import Path

def extract_citations(section_file, full_bib_file, output_bib_file):
    """Extract only citations used in a specific section"""
    
    # Read section content
    section_content = Path(section_file).read_text()
    
    # Find all citations
    citations = re.findall(r'\\cite\{([^}]+)\}', section_content)
    
    # Extract unique citation keys
    all_keys = []
    for cite_group in citations:
        keys = [k.strip() for k in cite_group.split(',')]
        all_keys.extend(keys)
    
    unique_keys = list(set(all_keys))
    print(f"Found {len(unique_keys)} unique citations in introduction.tex")
    
    # Read full bibliography
    full_bib = Path(full_bib_file).read_text()
    
    # Extract entries for these keys
    entries = []
    found_keys = []
    
    for key in unique_keys:
        # Find entry starting with @article{key, @book{key, etc.
        # Use a more robust pattern that handles multi-line entries
        pattern = rf'(@\w+\{{{key},(?:[^@])*?\n\}})'
        matches = re.findall(pattern, full_bib, re.DOTALL | re.MULTILINE)
        if matches:
            entries.extend(matches)
            found_keys.append(key)
        else:
            print(f"Warning: Citation '{key}' not found in ref.bib")
    
    # Write section-specific bibliography
    with open(output_bib_file, 'w') as f:
        f.write('% Section-specific bibliography for introduction\n')
        f.write(f'% Extracted {len(entries)} entries from {len(unique_keys)} citations\n\n')
        f.write('\n\n'.join(entries))
    
    print(f"Created {output_bib_file} with {len(entries)} bibliography entries")
    return len(unique_keys), len(entries)

if __name__ == "__main__":
    # Extract citations for introduction
    extract_citations('introduction.tex', 'ref.bib', 'introduction_refs.bib')