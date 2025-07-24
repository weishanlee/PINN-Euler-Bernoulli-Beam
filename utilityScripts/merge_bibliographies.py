#!/usr/bin/env python3
"""
Merge Section Bibliographies for Final PDF

This script merges all section-specific bibliography files into a single
unified bibliography for the final PDF compilation.

Usage:
    python merge_bibliographies.py --output ref_final.bib
"""

import argparse
import sys
from pathlib import Path
from collections import OrderedDict
import re


class BibliographyMerger:
    """Merge multiple bibliography files while removing duplicates."""
    
    def __init__(self, output_file):
        self.output_file = Path(output_file)
        self.entries = OrderedDict()
        self.section_files = []
        self.stats = {
            'total_files': 0,
            'total_entries_before': 0,
            'total_entries_after': 0,
            'duplicates_removed': 0
        }
    
    def find_section_bibliographies(self, directory='.'):
        """Find all section-specific bibliography files."""
        patterns = [
            '*_refs.bib',  # Standard section bibliography pattern
        ]
        
        found_files = []
        for pattern in patterns:
            found_files.extend(Path(directory).glob(pattern))
        
        # Sort files in logical order
        section_order = [
            'introduction', 'methods', 'results', 'conclusions',
            'summary', 'letter', 'abstract', 'appendix'
        ]
        
        def sort_key(file_path):
            name = file_path.stem.replace('_refs', '')
            for i, section in enumerate(section_order):
                if section in name:
                    return i
            return len(section_order)
        
        self.section_files = sorted(found_files, key=sort_key)
        self.stats['total_files'] = len(self.section_files)
        
        return self.section_files
    
    def parse_bib_entry(self, content, start_pos):
        """Parse a complete BibTeX entry starting from a given position."""
        # Find the entry type and key
        entry_match = re.match(r'@(\w+)\s*\{\s*([^,\s]+)', content[start_pos:])
        if not entry_match:
            return None, None, start_pos
            
        entry_type = entry_match.group(1)
        entry_key = entry_match.group(2)
        
        # Find the matching closing brace
        brace_count = 0
        pos = start_pos + entry_match.end()
        start = start_pos
        
        # Count braces to find the end of the entry
        i = pos
        while i < len(content):
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                if brace_count == 0:
                    # Found the closing brace for the entry
                    return entry_key, content[start:i+1], i+1
                brace_count -= 1
            i += 1
            
        return None, None, start_pos
    
    def process_file(self, bib_file):
        """Process a single bibliography file."""
        if not bib_file.exists():
            print(f"⚠️  Warning: File '{bib_file}' not found, skipping...")
            return 0
        
        content = bib_file.read_text(encoding='utf-8')
        entries_found = 0
        
        # Find all bibliography entries
        pos = 0
        while pos < len(content):
            # Look for start of entry
            match = re.search(r'@\w+\s*\{', content[pos:])
            if not match:
                break
                
            pos += match.start()
            entry_key, entry_content, new_pos = self.parse_bib_entry(content, pos)
            
            if entry_key and entry_content:
                if entry_key not in self.entries:
                    self.entries[entry_key] = entry_content
                    entries_found += 1
                else:
                    self.stats['duplicates_removed'] += 1
                pos = new_pos
            else:
                pos += 1
        
        return entries_found
    
    def merge_all(self):
        """Merge all found bibliography files."""
        print("\n🔍 Merging bibliography files...")
        
        if not self.section_files:
            print("❌ No section bibliography files found!")
            print("   Looking for files matching: *_refs.bib")
            return False
        
        print(f"📚 Found {len(self.section_files)} bibliography files:")
        
        for bib_file in self.section_files:
            entries_before = len(self.entries)
            entries_added = self.process_file(bib_file)
            self.stats['total_entries_before'] += entries_added
            
            print(f"   - {bib_file.name}: {entries_added} entries")
        
        self.stats['total_entries_after'] = len(self.entries)
        
        return True
    
    def write_merged_bibliography(self):
        """Write the merged bibliography to output file."""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write("% Merged bibliography from all sections\n")
            f.write(f"% Generated from {self.stats['total_files']} section bibliographies\n")
            f.write(f"% Total unique entries: {len(self.entries)}\n")
            f.write(f"% Duplicates removed: {self.stats['duplicates_removed']}\n\n")
            
            # Write entries
            for entry_key, entry_content in self.entries.items():
                f.write(entry_content)
                f.write("\n\n")
        
        print(f"\n✅ Created {self.output_file} with {len(self.entries)} unique entries")
    
    def generate_report(self):
        """Generate a detailed merge report."""
        report = []
        report.append("\n" + "=" * 60)
        report.append("BIBLIOGRAPHY MERGE REPORT")
        report.append("=" * 60)
        report.append(f"Files processed: {self.stats['total_files']}")
        report.append(f"Total entries before merge: {self.stats['total_entries_before']}")
        report.append(f"Duplicates removed: {self.stats['duplicates_removed']}")
        report.append(f"Final unique entries: {self.stats['total_entries_after']}")
        
        if self.stats['total_entries_before'] > 0:
            dedup_rate = (self.stats['duplicates_removed'] / self.stats['total_entries_before']) * 100
            report.append(f"Deduplication rate: {dedup_rate:.1f}%")
        
        report.append("\nSource files:")
        for f in self.section_files:
            report.append(f"  - {f.name}")
        
        report.append(f"\nOutput file: {self.output_file}")
        report.append("=" * 60)
        
        return '\n'.join(report)
    
    def run(self, directory='.'):
        """Run the complete merge process."""
        # Find bibliography files
        self.find_section_bibliographies(directory)
        
        # Merge all files
        if not self.merge_all():
            return False
        
        # Write output
        self.write_merged_bibliography()
        
        # Generate report
        print(self.generate_report())
        
        return True


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Merge section-specific bibliography files for final PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_bibliographies.py --output ref_final.bib
  python merge_bibliographies.py -o ref_final.bib --dir output/
  python merge_bibliographies.py -o ref_final.bib --check
        """
    )
    
    parser.add_argument(
        '--output', '-o',
        default='ref_final.bib',
        help='Output file for merged bibliography (default: ref_final.bib)'
    )
    
    parser.add_argument(
        '--dir', '-d',
        default='.',
        help='Directory to search for bibliography files (default: current directory)'
    )
    
    parser.add_argument(
        '--check', '-c',
        action='store_true',
        help='Check for missing section bibliographies'
    )
    
    args = parser.parse_args()
    
    # Create merger and run
    merger = BibliographyMerger(args.output)
    
    if args.check:
        # Check mode: look for expected files
        expected_sections = [
            'introduction', 'methods', 'results', 'conclusions'
        ]
        missing = []
        
        for section in expected_sections:
            if not Path(args.dir).joinpath(f"{section}_refs.bib").exists():
                missing.append(f"{section}_refs.bib")
        
        if missing:
            print("⚠️  Missing expected bibliography files:")
            for m in missing:
                print(f"   - {m}")
            print("\nMake sure all sections have been compiled with bibliography extraction.")
            sys.exit(1)
    
    if not merger.run(args.dir):
        sys.exit(1)
    
    print("\n✨ Bibliography merge completed successfully!")
    print(f"   Next step: Update main.tex to use \\bibliography{{{args.output.replace('.bib', '')}}}")


if __name__ == "__main__":
    main()