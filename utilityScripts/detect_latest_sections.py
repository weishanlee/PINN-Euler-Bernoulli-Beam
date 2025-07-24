#!/usr/bin/env python3
"""
Enhanced Section Version Detection Script with Content Analysis
Detects the BEST version of each section based on content completeness, not just modification time.

This script supports two modes:
1. Simple mode (default): Selects by modification time only
2. Content analysis mode (--analyze-content): Selects by content completeness score

The content analysis mode will:
- Count citations, figures, tables, equations in each version
- Check file sizes for potential content loss
- Score each version based on completeness
- Generate detailed decision logs
- Warn about potential content regressions
"""

import os
import re
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

class SectionVersionAnalyzer:
    """Analyzes multiple versions of sections to find the most content-complete version"""
    
    def __init__(self, output_dir: str = ".", paper_type: str = "competition", analyze_content: bool = False):
        self.output_dir = Path(output_dir)
        self.paper_type = paper_type
        self.analyze_content = analyze_content
        
        # Define sections based on paper type
        if paper_type == "competition":
            self.sections = [
                "summary", "letter", "introduction", "methods", 
                "resultsAndDiscussions", "conclusions", 
                "appendixCodes", "appendixAIReport"
            ]
        else:  # journal
            self.sections = [
                "abstract", "introduction", "methods", 
                "resultsAndDiscussions", "conclusions", 
                "appendixAIReport"
            ]
        
        self.warnings = []
        self.decisions = []
        
    def analyze_tex_file(self, filepath: Path) -> Dict:
        """Analyze a single .tex file for content metrics"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return {}
        
        # Remove comments for accurate counting
        content_no_comments = re.sub(r'%.*$', '', content, flags=re.MULTILINE)
        
        metrics = {
            'file': filepath.name,
            'path': str(filepath),
            'size': os.path.getsize(filepath),
            'modified': datetime.fromtimestamp(os.path.getmtime(filepath)),
            'citations': len(re.findall(r'\\cite\{[^}]+\}', content_no_comments)),
            'unique_citations': len(set(re.findall(r'\\cite\{([^}]+)\}', content_no_comments))),
            'figures': len(re.findall(r'\\includegraphics', content_no_comments)),
            'figure_refs': len(re.findall(r'\\ref\{fig:', content_no_comments)),
            'tables': len(re.findall(r'\\begin\{table\}|\\begin\{tabular\}', content_no_comments)),
            'table_refs': len(re.findall(r'\\ref\{tab:', content_no_comments)),
            'equations': len(re.findall(r'\\begin\{equation\}|\\begin\{align\}|\\\\\[|\$\$', content_no_comments)),
            'subsections': len(re.findall(r'\\(?:sub)*section\{', content_no_comments)),
            'labels': len(re.findall(r'\\label\{[^}]+\}', content_no_comments)),
            'references': len(re.findall(r'\\ref\{[^}]+\}', content_no_comments)),
        }
        
        # Extract subsection titles for comparison
        metrics['subsection_titles'] = re.findall(r'\\(?:sub)*section\{([^}]+)\}', content_no_comments)
        
        # Check for approval indicators
        metrics['has_approval'] = self._check_approval_status(filepath)
        
        # Calculate content hash (for detecting identical content)
        metrics['content_hash'] = hash(content_no_comments.strip())
        
        return metrics
    
    def _check_approval_status(self, filepath: Path) -> bool:
        """Check if this version has been approved"""
        filename_lower = filepath.name.lower()
        
        # Check filename indicators
        if any(indicator in filename_lower for indicator in ['approved', 'final']):
            return True
        
        # Check version_log.txt if exists
        version_log = self.output_dir / 'version_log.txt'
        if version_log.exists():
            try:
                with open(version_log, 'r') as f:
                    log_content = f.read()
                    if filepath.name in log_content and 'approved' in log_content:
                        return True
            except:
                pass
        
        return False
    
    def find_section_versions(self, section_base: str) -> List[Path]:
        """Find all versions of a section"""
        versions = []
        
        # Common version patterns
        patterns = [
            f"{section_base}.tex",
            f"{section_base}_*.tex",
            f"{section_base}*.tex"
        ]
        
        for pattern in patterns:
            versions.extend(self.output_dir.glob(pattern))
        
        # Filter out wrapper files and clean versions
        versions = [v for v in versions if not any(
            excluded in v.name for excluded in ['_wrapper', '_clean', '_refs']
        )]
        
        # Remove duplicates and sort
        versions = list(set(versions))
        versions.sort(key=lambda x: (x.stem, x.suffix))
        
        return versions
    
    def calculate_content_score(self, metrics: Dict, all_metrics: List[Dict]) -> float:
        """Calculate content completeness score for a version"""
        score = 0.0
        
        if not all_metrics:
            return 0.0
        
        # Find maximum values for normalization
        max_citations = max((m['citations'] for m in all_metrics), default=1)
        max_figures = max((m['figures'] for m in all_metrics), default=1)
        max_tables = max((m['tables'] for m in all_metrics), default=1)
        max_equations = max((m['equations'] for m in all_metrics), default=1)
        max_subsections = max((m['subsections'] for m in all_metrics), default=1)
        max_size = max((m['size'] for m in all_metrics), default=1)
        
        # Content completeness (40% weight)
        if max_citations > 0:
            citation_score = (metrics['citations'] / max_citations) * 20
            score += citation_score
        
        # Figures and tables (15% weight)
        if max_figures > 0:
            score += (metrics['figures'] / max_figures) * 7.5
        if max_tables > 0:
            score += (metrics['tables'] / max_tables) * 7.5
        
        # Mathematical content (10% weight)
        if max_equations > 0:
            score += (metrics['equations'] / max_equations) * 10
        
        # Structure preservation (10% weight)
        if max_subsections > 0:
            score += (metrics['subsections'] / max_subsections) * 10
        
        # Approval status (20% weight)
        if metrics['has_approval']:
            score += 20
        
        # Version naming conventions (5% weight)
        filename = metrics['file'].lower()
        if 'final' in filename:
            score += 5
        elif 'approved' in filename:
            score += 4
        elif re.search(r'_v\d+', filename):
            score += 2
        elif 'revised' in filename or 'modified' in filename:
            score += 1
        
        # Penalties
        # Significant size reduction penalty
        if metrics['size'] < 0.7 * max_size:
            penalty = 20
            score -= penalty
            self.warnings.append(f"WARNING: {metrics['file']} is significantly smaller ({penalty}% penalty)")
        
        # Lost citations penalty
        if max_citations > 0 and metrics['citations'] < 0.8 * max_citations:
            penalty = 15
            score -= penalty
            self.warnings.append(f"WARNING: {metrics['file']} has fewer citations ({penalty}% penalty)")
        
        return max(0, min(100, score))  # Clamp between 0-100
    
    def compare_versions(self, section: str) -> Optional[Dict]:
        """Compare all versions of a section and select the best"""
        versions = self.find_section_versions(section)
        
        if not versions:
            return None
        
        # Analyze each version
        analyses = []
        for version in versions:
            if self.analyze_content:
                metrics = self.analyze_tex_file(version)
                if metrics:
                    analyses.append(metrics)
            else:
                # Simple mode - just use modification time
                analyses.append({
                    'file': version.name,
                    'path': str(version),
                    'modified': datetime.fromtimestamp(os.path.getmtime(version)),
                    'size': os.path.getsize(version)
                })
        
        if not analyses:
            return None
        
        # Calculate scores if content analysis is enabled
        if self.analyze_content:
            for analysis in analyses:
                analysis['score'] = self.calculate_content_score(analysis, analyses)
            
            # Sort by score (highest first)
            analyses.sort(key=lambda x: x['score'], reverse=True)
            selected = analyses[0]
            
            # Generate decision explanation
            decision = f"\n{section.upper()} SECTION:"
            decision += f"\nSelected: {selected['file']} (Score: {selected['score']:.1f}/100)"
            
            if len(analyses) > 1:
                decision += "\nComparison:"
                for a in analyses:
                    decision += f"\n  - {a['file']}: "
                    decision += f"Score={a['score']:.1f}, "
                    decision += f"{a['citations']} citations, "
                    decision += f"{a['size']/1024:.1f}KB"
                    if a['has_approval']:
                        decision += " [APPROVED]"
            
            self.decisions.append(decision)
            
            # Check for potential issues
            if len(analyses) > 1:
                latest = max(analyses, key=lambda x: x['modified'])
                if latest['file'] != selected['file']:
                    self.warnings.append(
                        f"NOTE: {section} - Selected {selected['file']} over "
                        f"newer {latest['file']} due to better content completeness"
                    )
        else:
            # Simple mode - select by modification time
            analyses.sort(key=lambda x: x['modified'], reverse=True)
            selected = analyses[0]
        
        return {
            'section': section,
            'selected': selected,
            'all_versions': analyses
        }
    
    def generate_report(self) -> Dict:
        """Generate complete analysis report"""
        report = {
            'generated': datetime.now().isoformat(),
            'mode': 'content_analysis' if self.analyze_content else 'modification_time',
            'paper_type': self.paper_type,
            'detected_sections': {},
            'latex_inputs': {},
            'warnings': self.warnings,
            'decisions': self.decisions
        }
        
        print(f"\n{'='*60}")
        print(f"Section Version Detection Report")
        print(f"Mode: {'Content Analysis' if self.analyze_content else 'Modification Time'}")
        print(f"Paper Type: {self.paper_type}")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        for section in self.sections:
            result = self.compare_versions(section)
            
            if result:
                selected = result['selected']
                report['detected_sections'][section] = {
                    'latest_file': selected['file'],
                    'path': selected['path'],
                    'modified': selected['modified'].isoformat() if isinstance(selected['modified'], datetime) else str(selected['modified']),
                    'size': selected['size'],
                    'score': selected.get('score', 'N/A'),
                    'version_count': len(result['all_versions'])
                }
                
                # Generate LaTeX input command
                filename_without_ext = Path(selected['file']).stem
                report['latex_inputs'][section] = f"\\input{{{filename_without_ext}}}"
                
                # Print summary
                print(f"{section}:")
                print(f"  Selected: {selected['file']}")
                if self.analyze_content:
                    print(f"  Score: {selected.get('score', 0):.1f}/100")
                    if 'citations' in selected:
                        print(f"  Citations: {selected['citations']}")
                print(f"  Modified: {selected['modified']}")
                print(f"  Versions found: {len(result['all_versions'])}")
                print()
            else:
                print(f"{section}: No files found")
                print()
        
        # Save reports
        report_path = self.output_dir / 'section_detection_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save decision log if content analysis was used
        if self.analyze_content and self.decisions:
            log_path = self.output_dir / 'assembly_decisions.log'
            with open(log_path, 'w') as f:
                f.write("=== Version Selection Decisions ===\n")
                f.write(f"Generated: {datetime.now()}\n")
                f.write(f"Mode: Content Analysis\n")
                f.write("\n".join(self.decisions))
        
        # Print warnings
        if self.warnings:
            print(f"\n{'='*60}")
            print("WARNINGS:")
            print(f"{'='*60}")
            for warning in self.warnings:
                print(f"⚠️  {warning}")
        
        print(f"\n✓ Report saved to: {report_path}")
        if self.analyze_content:
            print(f"✓ Decision log saved to: {self.output_dir / 'assembly_decisions.log'}")
        
        return report
    
    def print_latex_commands(self, report: Dict):
        """Print LaTeX input commands for main.tex"""
        print(f"\n{'='*60}")
        print("LaTeX \\input{} commands for main.tex:")
        print(f"{'='*60}")
        
        for section, command in report['latex_inputs'].items():
            print(command)


# Keep the old functions for backward compatibility
def get_section_base_names():
    """Define standard section names for both competition and journal papers"""
    return {
        'competition': [
            'summary', 'letter', 'introduction', 'methods', 
            'resultsAndDiscussions', 'conclusions', 
            'appendixCodes', 'appendixAIReport'
        ],
        'journal': [
            'abstract', 'introduction', 'methods', 
            'resultsAndDiscussions', 'conclusions', 
            'appendixAIReport'
        ]
    }

def extract_version_info(filename, base_name):
    """Extract version information from filename"""
    # Remove .tex extension
    name_without_ext = filename[:-4] if filename.endswith('.tex') else filename
    
    # Pattern matching for different version formats
    patterns = [
        (r'^' + base_name + r'$', 0),  # Original file
        (r'^' + base_name + r'_v(\d+)$', 1),  # _v2, _v3, etc.
        (r'^' + base_name + r'_modified$', 1.5),  # _modified
        (r'^' + base_name + r'_revised$', 1.7),  # _revised
        (r'^' + base_name + r'_final$', 2),  # _final
        (r'^' + base_name + r'_approved$', 3),  # _approved
    ]
    
    for pattern, default_version in patterns:
        match = re.match(pattern, name_without_ext)
        if match:
            if len(match.groups()) > 0:
                return float(match.group(1))
            return default_version
    
    return -1  # Unknown pattern

def find_latest_section_files(output_dir, paper_type='competition'):
    """Find the latest version of each section file based on modification time"""
    latest_files = {}
    base_names = get_section_base_names()[paper_type]
    
    # Get all .tex files in output directory
    tex_files = [f for f in os.listdir(output_dir) if f.endswith('.tex')]
    
    for base_name in base_names:
        candidates = []
        
        for tex_file in tex_files:
            # Check if this file is a version of the current base section
            if tex_file.startswith(base_name):
                # Get file modification time
                file_path = os.path.join(output_dir, tex_file)
                mod_time = os.path.getmtime(file_path)
                
                # Get version info (for reporting purposes only)
                version = extract_version_info(tex_file, base_name)
                
                if version >= 0:
                    candidates.append({
                        'filename': tex_file,
                        'path': file_path,
                        'version': version,
                        'modified': mod_time,
                        'modified_str': datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                    })
        
        # Select the best candidate
        if candidates:
            # Sort by modification time ONLY (most recent first)
            candidates.sort(key=lambda x: x['modified'], reverse=True)
            latest_files[base_name] = candidates[0]
            
            # Also store alternatives for reporting
            if len(candidates) > 1:
                latest_files[base_name]['alternatives'] = candidates[1:]
    
    return latest_files

def generate_main_tex_inputs(latest_files):
    """Generate the \input commands for main.tex"""
    inputs = {}
    
    for base_name, file_info in latest_files.items():
        # Remove .tex extension for \input command
        tex_name = file_info['filename'][:-4]
        inputs[base_name] = f"\\input{{{tex_name}}}"
    
    return inputs

def create_detection_report(output_dir, paper_type='competition'):
    """Create a comprehensive report of detected files"""
    latest_files = find_latest_section_files(output_dir, paper_type)
    
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'paper_type': paper_type,
        'output_directory': output_dir,
        'detected_sections': {}
    }
    
    print("\n" + "="*60)
    print("LATEST SECTION FILE DETECTION REPORT")
    print("(Based on FILE MODIFICATION TIME)")
    print("="*60)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Paper Type: {paper_type}")
    print(f"Output Directory: {output_dir}")
    print("\nSelection Criteria: Most Recently Modified File")
    print("-"*60)
    
    for base_name, file_info in latest_files.items():
        report['detected_sections'][base_name] = {
            'latest_file': file_info['filename'],
            'version': file_info['version'],
            'modified': file_info['modified_str']
        }
        
        print(f"\n{base_name}:")
        print(f"  Latest (by modification time): {file_info['filename']}")
        print(f"  Last modified: {file_info['modified_str']}")
        
        if 'alternatives' in file_info:
            report['detected_sections'][base_name]['alternatives'] = []
            print("  Other versions (sorted by modification time, newest first):")
            for alt in file_info['alternatives']:
                alt_info = {
                    'file': alt['filename'],
                    'version': alt['version'],
                    'modified': alt['modified_str']
                }
                report['detected_sections'][base_name]['alternatives'].append(alt_info)
                print(f"    - {alt['filename']} - Modified: {alt['modified_str']}")
    
    # Generate \input commands
    inputs = generate_main_tex_inputs(latest_files)
    report['latex_inputs'] = inputs
    
    print("\n" + "-"*60)
    print("RECOMMENDED \\input COMMANDS FOR main.tex:")
    print("-"*60)
    
    for base_name in get_section_base_names()[paper_type]:
        if base_name in inputs:
            print(inputs[base_name])
    
    print("\n" + "="*60)
    
    # Save report to file
    report_path = os.path.join(output_dir, 'section_detection_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {report_path}")
    
    return report

def main():
    parser = argparse.ArgumentParser(
        description='Detect the best version of each section for final PDF assembly',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple mode (by modification time only):
  python detect_latest_sections.py
  
  # Content analysis mode (RECOMMENDED):
  python detect_latest_sections.py --analyze-content
  
  # For journal papers with content analysis:
  python detect_latest_sections.py --paper-type journal --analyze-content
  
  # Specify custom output directory:
  python detect_latest_sections.py --output-dir ../output --analyze-content
        """
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Directory containing section .tex files (default: current directory)'
    )
    
    parser.add_argument(
        '--paper-type',
        type=str,
        choices=['competition', 'journal'],
        default='competition',
        help='Type of paper (affects section list)'
    )
    
    parser.add_argument(
        '--analyze-content',
        action='store_true',
        help='Enable content analysis mode (RECOMMENDED) - selects based on content completeness, not just modification time'
    )
    
    parser.add_argument(
        '--show-commands',
        action='store_true',
        help='Show LaTeX input commands for main.tex'
    )
    
    parser.add_argument(
        '--json-only',
        action='store_true',
        help='Only output JSON report without printing (for backward compatibility)'
    )
    
    args = parser.parse_args()
    
    # Verify output directory exists
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory '{args.output_dir}' does not exist")
        sys.exit(1)
    
    # Handle backward compatibility
    if args.json_only:
        # Use old behavior for backward compatibility
        latest_files = find_latest_section_files(args.output_dir, args.paper_type)
        inputs = generate_main_tex_inputs(latest_files)
        
        report = {
            'detected_sections': latest_files,
            'latex_inputs': inputs
        }
        
        print(json.dumps(report, indent=2))
        return 0
    
    # Use new analyzer
    analyzer = SectionVersionAnalyzer(
        output_dir=args.output_dir,
        paper_type=args.paper_type,
        analyze_content=args.analyze_content
    )
    
    report = analyzer.generate_report()
    
    if args.show_commands:
        analyzer.print_latex_commands(report)
    
    # Exit with warning status if issues found
    if analyzer.warnings:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()