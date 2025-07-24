import os
import re

# Get all PNG files in figures directory
figures_dir = '/home/wslee/Desktop/claudeCode/paperAgent_Euler_Beam/output/figures'
all_figures = set([f for f in os.listdir(figures_dir) if f.endswith('.png')])

# Find all figures referenced in tex files
output_dir = '/home/wslee/Desktop/claudeCode/paperAgent_Euler_Beam/output'
referenced_figures = set()

for filename in os.listdir(output_dir):
    if filename.endswith('.tex'):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'r') as f:
            content = f.read()
            # Find all includegraphics commands
            matches = re.findall(r'\\includegraphics.*?{figures/(.*?\.png)}', content)
            referenced_figures.update(matches)

# Find missing figures
missing_figures = all_figures - referenced_figures

print(f"Total figures: {len(all_figures)}")
print(f"Referenced figures: {len(referenced_figures)}")
print(f"Missing figures: {len(missing_figures)}")
print("\nMissing figures:")
for fig in sorted(missing_figures):
    print(f"  - {fig}")