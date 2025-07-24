import re

# Read the bib file
with open('/home/wslee/Desktop/claudeCode/paperAgent_Euler_Beam/output/ref.bib', 'r') as f:
    content = f.read()

# Find all entry keys
entries = re.findall(r'@\w+{([^,]+),', content)

# Find duplicates
seen = {}
duplicates = []

for i, entry in enumerate(entries):
    if entry in seen:
        duplicates.append((entry, seen[entry], i+1))
    else:
        seen[entry] = i+1

print(f"Found {len(duplicates)} duplicate entries:")
for dup, first_line, second_line in duplicates:
    print(f"  - '{dup}' appears at approximate positions {first_line} and {second_line}")

# Find line numbers for each duplicate
lines = content.split('\n')
for dup, _, _ in duplicates:
    print(f"\nSearching for '{dup}':")
    for i, line in enumerate(lines):
        if f'@article{{{dup}' in line or f'@inproceedings{{{dup}' in line:
            print(f"  Line {i+1}: {line}")