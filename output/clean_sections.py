#!/usr/bin/env python3
"""
Remove review checklists from section files for final compilation
"""
import re

def clean_section(input_file, output_file):
    """Remove review checklist from section file"""
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Find the start of the review checklist
    # Look for the pattern that starts the checklist
    patterns = [
        r'%% ========== SECTION REVIEW CHECKLIST ==========.*?%% ========== END SECTION REVIEW CHECKLIST ==========',
        r'\\clearpage\s*\\section\*{Review Checklist}.*?\\end{verbatim}\s*\\end{small}',
        r'\\clearpage.*?Review Checklist.*?(?=\\end{document}|$)'
    ]
    
    cleaned_content = content
    for pattern in patterns:
        cleaned_content = re.sub(pattern, '', cleaned_content, flags=re.DOTALL)
    
    # Also remove any trailing clearpage commands that were just for the checklist
    cleaned_content = re.sub(r'\\clearpage\s*$', '', cleaned_content)
    
    with open(output_file, 'w') as f:
        f.write(cleaned_content)
    
    print(f"Cleaned {input_file} -> {output_file}")

# Clean all section files
sections = [
    'introduction.tex',
    'methods.tex', 
    'resultsAndDiscussions.tex',
    'conclusions.tex',
    'abstract.tex',
    'appendixAIReport.tex'
]

for section in sections:
    output_name = section.replace('.tex', '_clean.tex')
    clean_section(section, output_name)

print("\nAll sections cleaned. Update main.tex to use _clean versions.")