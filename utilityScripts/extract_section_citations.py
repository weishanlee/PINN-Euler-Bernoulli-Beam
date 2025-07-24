#!/usr/bin/env python3
"""
Extract Section-Specific Citations from LaTeX Files

This script extracts all citations used in a specific LaTeX section and creates
a section-specific bibliography file containing only the referenced entries.

Usage:
    python extract_section_citations.py --section introduction.tex --master ref.bib --output introduction_refs.bib
"""

import re
import argparse
import sys
from pathlib import Path
from collections import OrderedDict


class CitationExtractor:
    """Extract citations from LaTeX files and create section-specific bibliographies."""
    
    def __init__(self, section_file, master_bib, output_bib):
        self.section_file = Path(section_file)
        self.master_bib = Path(master_bib)
        self.output_bib = Path(output_bib)
        self.citations = []
        self.bib_entries = OrderedDict()
        
    def extract_citations(self):
        """Extract all \cite{} commands from the section file."""
        if not self.section_file.exists():
            print(f"❌ Error: Section file '{self.section_file}' not found")
            return False
            
        content = self.section_file.read_text(encoding='utf-8')
        
        # Find all citation commands
        # Matches: \cite{key}, \cite{key1,key2}, \citep{key}, \citet{key}, etc.
        cite_pattern = r'\\cite[pt]?\{([^}]+)\}'
        matches = re.findall(cite_pattern, content, re.IGNORECASE)
        
        # Extract individual citation keys
        for match in matches:
            # Split by comma and strip whitespace
            keys = [k.strip() for k in match.split(',')]
            self.citations.extend(keys)
        
        # Remove duplicates while preserving order
        self.citations = list(dict.fromkeys(self.citations))
        
        print(f"📚 Found {len(self.citations)} unique citations in {self.section_file.name}")
        
        if len(self.citations) == 0:
            print("⚠️  Warning: No citations found in the section")
            return True
            
        # Print first few citations for verification
        preview = self.citations[:5]
        if len(self.citations) > 5:
            preview_str = ', '.join(preview) + f", ... ({len(self.citations)-5} more)"
        else:
            preview_str = ', '.join(preview)
        print(f"   Citations: {preview_str}")
        
        return True
    
    def parse_bib_entry(self, content, start_pos):
        """Parse a complete BibTeX entry starting from a given position."""
        # Find the entry type and key
        entry_match = re.match(r'@(\w+)\s*\{\s*([^,\s]+)', content[start_pos:])
        if not entry_match:
            return None, start_pos
            
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
                    return content[start:i+1], i+1
                brace_count -= 1
            i += 1
            
        return None, start_pos
    
    def extract_bib_entries(self):
        """Extract bibliography entries for all found citations."""
        if not self.master_bib.exists():
            print(f"❌ Error: Master bibliography '{self.master_bib}' not found")
            return False
            
        content = self.master_bib.read_text(encoding='utf-8')
        
        # Find all bibliography entries
        for citation in self.citations:
            # Look for entries that match this citation key
            # Pattern matches @article{citation, @book{citation, etc.
            pattern = rf'@\w+\s*\{{\s*{re.escape(citation)}\s*,'
            
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                # Extract the complete entry
                entry, _ = self.parse_bib_entry(content, match.start())
                if entry:
                    self.bib_entries[citation] = entry
                else:
                    print(f"⚠️  Warning: Could not parse entry for '{citation}'")
            else:
                print(f"⚠️  Warning: Citation '{citation}' not found in master bibliography")
        
        print(f"📖 Extracted {len(self.bib_entries)} bibliography entries")
        
        return True
    
    def write_section_bibliography(self):
        """Write the section-specific bibliography file."""
        # Create output directory if it doesn't exist
        self.output_bib.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_bib, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"% Section-specific bibliography for {self.section_file.name}\n")
            f.write(f"% Generated from {self.master_bib.name}\n")
            f.write(f"% Total entries: {len(self.bib_entries)}\n\n")
            
            # Write entries in the order they were cited
            for citation, entry in self.bib_entries.items():
                f.write(entry)
                f.write("\n\n")
        
        print(f"✅ Created {self.output_bib} with {len(self.bib_entries)} entries")
        
        # Report missing citations
        missing = set(self.citations) - set(self.bib_entries.keys())
        if missing:
            print(f"\n⚠️  Missing entries for {len(missing)} citations:")
            for m in sorted(missing):
                print(f"   - {m}")
            print("\n   These citations will cause LaTeX errors during compilation.")
            print("   Please add them to the master bibliography or remove from the section.")
        
        return True
    
    def generate_report(self):
        """Generate a detailed report of the extraction process."""
        report = []
        report.append("=" * 60)
        report.append("CITATION EXTRACTION REPORT")
        report.append("=" * 60)
        report.append(f"Section file: {self.section_file}")
        report.append(f"Master bibliography: {self.master_bib}")
        report.append(f"Output bibliography: {self.output_bib}")
        report.append(f"\nTotal citations found: {len(self.citations)}")
        report.append(f"Total entries extracted: {len(self.bib_entries)}")
        report.append(f"Missing entries: {len(self.citations) - len(self.bib_entries)}")
        
        if len(self.citations) > 0:
            report.append("\nExtraction rate: {:.1f}%".format(
                100 * len(self.bib_entries) / len(self.citations)
            ))
        
        return '\n'.join(report)
    
    def run(self):
        """Run the complete extraction process."""
        print("\n🔍 Starting citation extraction...")
        
        # Step 1: Extract citations from section
        if not self.extract_citations():
            return False
        
        # Step 2: Extract bibliography entries
        if not self.extract_bib_entries():
            return False
        
        # Step 3: Write section bibliography
        if not self.write_section_bibliography():
            return False
        
        # Step 4: Generate report
        print("\n" + self.generate_report())
        
        return True


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Extract section-specific citations from LaTeX files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_section_citations.py --section introduction.tex --master ref.bib --output introduction_refs.bib
  python extract_section_citations.py -s methods.tex -m ref.bib -o methods_refs.bib
        """
    )
    
    parser.add_argument(
        '--section', '-s',
        required=True,
        help='Path to the section LaTeX file (e.g., introduction.tex)'
    )
    
    parser.add_argument(
        '--master', '-m',
        required=True,
        help='Path to the master bibliography file (e.g., ref.bib)'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Path for the output section-specific bibliography (e.g., introduction_refs.bib)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Create extractor and run
    extractor = CitationExtractor(args.section, args.master, args.output)
    
    if not extractor.run():
        sys.exit(1)
    
    print("\n✨ Citation extraction completed successfully!")


if __name__ == "__main__":
    main()