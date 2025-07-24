import re

# Read the original file
with open('/home/wslee/Desktop/claudeCode/paperAgent_Euler_Beam/output/ref.bib', 'r') as f:
    content = f.read()

# Define the duplicate entries and their line numbers
duplicates_to_remove = [
    ('penwarden2023unified', 447, 455),
    ('arzani2023theory', 457, 465),
    ('lin2022two', 477, 485),
    ('lee2024anti', 498, 506),
    ('haghighat2022physics', 508, 516),
    ('chen2021physics', 539, 548),
    ('zakian2023physics', 573, 581)
]

# Split content into lines
lines = content.split('\n')

# Mark lines to remove
lines_to_remove = set()
for entry_key, start_line, end_line in duplicates_to_remove:
    # Mark lines from start_line-1 to end_line-1 (0-indexed)
    for i in range(start_line-1, end_line):
        lines_to_remove.add(i)

# Create new content without duplicate entries
new_lines = []
for i, line in enumerate(lines):
    if i not in lines_to_remove:
        new_lines.append(line)

# Join lines back together
new_content = '\n'.join(new_lines)

# Write the cleaned content back
with open('/home/wslee/Desktop/claudeCode/paperAgent_Euler_Beam/output/ref.bib', 'w') as f:
    f.write(new_content)

print("Removed duplicate entries from ref.bib")
print(f"Original lines: {len(lines)}")
print(f"New lines: {len(new_lines)}")
print(f"Lines removed: {len(lines) - len(new_lines)}")