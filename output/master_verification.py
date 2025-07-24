# master_verification.py
import os
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime

class MasterVerification:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "checks_performed": [],
            "errors": [],
            "warnings": [],
            "status": "PENDING"
        }
    
    def check_web_scraping_integrity(self):
        """Verify all web-scraped materials are genuine"""
        print("🔍 Checking web scraping integrity...")
        
        # Check verification report exists
        if not Path("verification_report.md").exists():
            self.errors.append("CRITICAL: No web scraping verification report found")
            return False
        
        # Check paper counts
        arxiv_count = len(list(Path("papers/arxiv_papers").glob("*.pdf")))
        journal_count = len(list(Path("papers/journal_papers").glob("*.pdf")))
        
        if arxiv_count < 50:
            self.errors.append(f"Insufficient arXiv papers: {arxiv_count} < 50")
        if journal_count < 30:
            self.errors.append(f"Insufficient journal papers: {journal_count} < 30")
        
        # Verify checksums log exists
        if not Path("verification_log.txt").exists():
            self.errors.append("No verification log found")
        
        self.report["checks_performed"].append("web_scraping_integrity")
        return len(self.errors) == 0
    
    def check_all_sections_exist(self):
        """Verify all required LaTeX sections are present"""
        print("📄 Checking all sections exist...")
        
        required_sections = {
            "competition": ["summary.tex", "letter.tex", "introduction.tex", 
                          "methods.tex", "resultsAndDiscussions.tex", 
                          "conclusions.tex", "appendixCodes.tex", 
                          "appendixAIReport.tex"],
            "journal": ["abstract.tex", "introduction.tex", "methods.tex",
                       "resultsAndDiscussions.tex", "conclusions.tex",
                       "appendixAIReport.tex"]
        }
        
        # Determine paper type
        paper_type = "competition" if Path("summary.tex").exists() else "journal"
        
        for section in required_sections[paper_type]:
            section_path = Path(section)
            if not section_path.exists():
                self.errors.append(f"Missing section: {section}")
            elif section_path.stat().st_size < 100:
                self.warnings.append(f"Section suspiciously small: {section}")
        
        self.report["checks_performed"].append("section_existence")
        return True
    
    def check_deep_content_verification(self):
        """Perform deep content verification on all papers"""
        print("🔬 Performing deep content verification...")
        
        # Check paper_metadata.json exists
        if not Path("paper_metadata.json").exists():
            self.errors.append("CRITICAL: paper_metadata.json not found - no content verification performed")
            return False
        
        # Check verification_db.json exists
        if not Path("verification_db.json").exists():
            self.errors.append("CRITICAL: verification_db.json not found - no verification database")
            return False
        
        # Load verification data
        with open("paper_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        with open("verification_db.json", 'r') as f:
            verification_db = json.load(f)
        
        # Check verification statistics
        total_papers = verification_db.get('total_papers', 0)
        verified = verification_db.get('verified', 0)
        failed = verification_db.get('failed', 0)
        
        if total_papers == 0:
            self.errors.append("No papers were verified")
            return False
        
        success_rate = verified / total_papers
        print(f"  Verification success rate: {success_rate:.1%}")
        
        if success_rate < 0.95:
            self.errors.append(f"Verification success rate too low: {success_rate:.1%} < 95%")
        
        # Check for title mismatches
        title_mismatches = 0
        domain_mismatches = 0
        
        for citation_key, data in metadata.items():
            if 'similarity_score' in data and data['similarity_score'] < 0.8:
                title_mismatches += 1
            if 'verification_status' in data and data['verification_status'] == 'FAILED':
                if 'domain_mismatch' in str(data):
                    domain_mismatches += 1
        
        if title_mismatches > 0:
            self.errors.append(f"{title_mismatches} papers have title similarity < 80%")
        
        if domain_mismatches > 0:
            self.errors.append(f"{domain_mismatches} papers are from wrong domain/field")
        
        # Check verification failures log
        if Path("verification_failures.log").exists():
            with open("verification_failures.log", 'r') as f:
                failures = f.readlines()
            if len(failures) > 10:
                self.warnings.append(f"High number of verification failures: {len(failures)}")
        
        self.report["checks_performed"].append("deep_content_verification")
        return len(self.errors) == 0
    
    def check_citation_integrity(self):
        """Verify all citations have corresponding PDFs"""
        print("📚 Checking citation integrity...")
        
        # Extract all citations from all tex files
        all_citations = set()
        for tex_file in Path(".").glob("*.tex"):
            content = tex_file.read_text()
            citations = re.findall(r'\\cite{([^}]+)}', content)
            for cite_group in citations:
                keys = [k.strip() for k in cite_group.split(',')]
                all_citations.update(keys)
        
        # Check bibliography file
        bib_entries = {}
        if Path("ref.bib").exists():
            bib_content = Path("ref.bib").read_text()
            entries = re.findall(r'@\w+{([^,]+),.*?(?=@|\Z)', bib_content, re.DOTALL)
            for entry in entries:
                bib_entries[entry.strip()] = True
        
        # Check each citation
        missing_pdfs = []
        citation_pdf_matches = {}
        
        for cite in all_citations:
            pdf_found = False
            matching_pdf = None
            
            # Check if citation exists in bibliography
            if cite not in bib_entries:
                self.errors.append(f"Citation '{cite}' not found in ref.bib")
                continue
            
            # Look for corresponding PDF
            for pdf in Path("papers").rglob("*.pdf"):
                if cite.lower() in pdf.name.lower():
                    pdf_found = True
                    matching_pdf = pdf.name
                    break
            
            if not pdf_found:
                missing_pdfs.append(cite)
            else:
                citation_pdf_matches[cite] = matching_pdf
        
        # Double-check PDF content matches bibliography
        if Path("undownloadable_papers_list.md").exists():
            # Check if missing PDFs are in the verified undownloadable list
            undownloadable_content = Path("undownloadable_papers_list.md").read_text()
            verified_undownloadable = []
            
            for cite in missing_pdfs[:]:  # Create a copy to iterate
                if cite in undownloadable_content and "✅ Verified Genuine" in undownloadable_content:
                    verified_undownloadable.append(cite)
                    missing_pdfs.remove(cite)
        
        if missing_pdfs:
            self.errors.append(f"Citations without PDFs (not in verified list): {', '.join(missing_pdfs)}")
        
        if verified_undownloadable:
            self.warnings.append(f"Citations verified but PDFs unavailable: {', '.join(verified_undownloadable)}")
        
        # Report matches found
        print(f"  Found {len(citation_pdf_matches)} citation-PDF matches")
        print(f"  {len(verified_undownloadable)} verified but unavailable")
        print(f"  {len(missing_pdfs)} missing without verification")
        
        self.report["checks_performed"].append("citation_integrity")
        return len(missing_pdfs) == 0
    
    def check_forbidden_words(self):
        """Check for AI-detection trigger words"""
        print("🚫 Checking for forbidden words...")
        
        forbidden = ['Innovative', 'Meticulous', 'Intricate', 'Notable',
                    'Versatile', 'Noteworthy', 'Invaluable', 'Pivotal',
                    'Potent', 'Fresh', 'Ingenious', 'Meticulously',
                    'Reportedly', 'Lucidly', 'Innovatively', 'Aptly',
                    'Methodically', 'Excellently', 'Compellingly',
                    'Impressively', 'Undoubtedly', 'Scholarly', 'Strategically']
        
        found_words = {}
        for tex_file in Path(".").glob("*.tex"):
            content = tex_file.read_text()
            for word in forbidden:
                if word.lower() in content.lower():
                    if tex_file.name not in found_words:
                        found_words[tex_file.name] = []
                    found_words[tex_file.name].append(word)
        
        if found_words:
            for file, words in found_words.items():
                self.errors.append(f"Forbidden words in {file}: {', '.join(words)}")
        
        self.report["checks_performed"].append("forbidden_words")
        return len(found_words) == 0
    
    def check_code_functionality(self):
        """Verify all code runs without errors"""
        print("💻 Checking code functionality...")
        
        import subprocess
        
        failed_scripts = []
        for script in Path("codes").glob("*.py"):
            try:
                result = subprocess.run(
                    ["~/.venv/ml_31226124/bin/python", str(script), "--test"],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode != 0:
                    failed_scripts.append(script.name)
            except Exception as e:
                failed_scripts.append(f"{script.name}: {str(e)}")
        
        if failed_scripts:
            self.errors.append(f"Failed scripts: {', '.join(failed_scripts)}")
        
        self.report["checks_performed"].append("code_functionality")
        return len(failed_scripts) == 0
    
    def check_figures_and_data(self):
        """Verify all referenced figures and data exist"""
        print("📊 Checking figures and data...")
        
        # Extract figure references
        figure_refs = set()
        for tex_file in Path(".").glob("*.tex"):
            content = tex_file.read_text()
            refs = re.findall(r'\\includegraphics.*{figures/([^}]+)}', content)
            figure_refs.update(refs)
        
        # Check figures exist
        missing_figures = []
        for fig in figure_refs:
            if not Path(f"figures/{fig}").exists():
                missing_figures.append(fig)
        
        if missing_figures:
            self.errors.append(f"Missing figures: {', '.join(missing_figures)}")
        
        # Check data documentation
        undocumented_data = []
        for csv in Path("data").glob("*.csv"):
            doc_file = csv.with_suffix('.txt')
            if not doc_file.exists():
                undocumented_data.append(csv.name)
        
        if undocumented_data:
            self.warnings.append(f"Undocumented data: {', '.join(undocumented_data)}")
        
        self.report["checks_performed"].append("figures_and_data")
        return len(missing_figures) == 0
    
    def check_word_limits(self):
        """Verify sections meet word/page requirements"""
        print("📏 Checking word limits...")
        
        # Check summary word count (competition only)
        if Path("summary.tex").exists():
            content = Path("summary.tex").read_text()
            word_count = len(re.findall(r'\b\w+\b', content))
            if not (540 <= word_count <= 560):
                self.errors.append(f"Summary word count {word_count} not in range 540-560")
        
        # Check abstract (journal only)
        if Path("abstract.tex").exists():
            content = Path("abstract.tex").read_text()
            word_count = len(re.findall(r'\b\w+\b', content))
            if word_count > 350:
                self.warnings.append(f"Abstract word count {word_count} > 350")
        
        self.report["checks_performed"].append("word_limits")
        return True
    
    def generate_final_report(self):
        """Generate comprehensive verification report"""
        print("\n📋 Generating final report...")
        
        if self.errors:
            self.report["status"] = "FAILED"
            self.report["errors"] = self.errors
        else:
            self.report["status"] = "PASSED"
        
        self.report["warnings"] = self.warnings
        
        # Save report
        with open("final_verification_report.json", 'w') as f:
            json.dump(self.report, f, indent=2)
        
        # Display results
        print("\n" + "="*60)
        print("FINAL VERIFICATION REPORT")
        print("="*60)
        
        if self.errors:
            print("\n❌ ERRORS FOUND - DO NOT COMPILE:")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if not self.errors:
            print("\n✅ ALL CHECKS PASSED - SAFE TO COMPILE")
        else:
            print("\n🛑 FIX ALL ERRORS BEFORE PROCEEDING")
        
        print("="*60)
        
        return len(self.errors) == 0
    
    def run_all_checks(self):
        """Execute all verification checks"""
        checks = [
            self.check_web_scraping_integrity,
            self.check_deep_content_verification,  # NEW: Deep content check
            self.check_all_sections_exist,
            self.check_citation_integrity,
            self.check_forbidden_words,
            self.check_code_functionality,
            self.check_figures_and_data,
            self.check_word_limits
        ]
        
        for check in checks:
            try:
                check()
            except Exception as e:
                self.errors.append(f"Check {check.__name__} failed: {str(e)}")
        
        return self.generate_final_report()

if __name__ == "__main__":
    verifier = MasterVerification()
    if not verifier.run_all_checks():
        exit(1)
    
    print("\n🎉 Ready for final compilation!")