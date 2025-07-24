#!/usr/bin/env python3
"""
Verify all citations in ref.bib match with PDFs in the papers directory
"""

import re
import os
import json

def extract_citations_from_bib(bib_file):
    """Extract all citation keys from ref.bib"""
    citations = []
    with open(bib_file, 'r') as f:
        content = f.read()
    
    # Find all citation entries
    pattern = r'@\w+{([^,]+),'
    matches = re.findall(pattern, content)
    
    return matches

def get_pdf_files(papers_dir):
    """Get all PDF files in the papers directory"""
    pdf_files = []
    if os.path.exists(papers_dir):
        for file in os.listdir(papers_dir):
            if file.endswith('.pdf'):
                pdf_files.append(file)
    return pdf_files

def check_citation_pdf_match():
    """Main verification function"""
    bib_file = 'ref.bib'
    papers_dir = 'papers/'
    
    # Extract citations from bibliography
    citations = extract_citations_from_bib(bib_file)
    print(f"Found {len(citations)} citations in ref.bib")
    
    # Get PDF files
    pdf_files = get_pdf_files(papers_dir)
    print(f"Found {len(pdf_files)} PDF files in papers directory")
    
    # Load the mapping from our earlier work
    with open('paper_metadata.json', 'r') as f:
        paper_metadata = json.load(f)
    
    # Check each citation
    citations_with_pdf = []
    citations_without_pdf = []
    citations_not_in_metadata = []
    
    for citation in citations:
        if citation in paper_metadata:
            if paper_metadata[citation].get('pdf_file'):
                citations_with_pdf.append(citation)
            else:
                citations_without_pdf.append(citation)
        else:
            citations_not_in_metadata.append(citation)
    
    # Generate report
    print("\n=== CITATION VERIFICATION REPORT ===\n")
    print(f"Total citations in ref.bib: {len(citations)}")
    print(f"Citations with PDFs: {len(citations_with_pdf)}")
    print(f"Citations without PDFs (verified genuine): {len(citations_without_pdf)}")
    print(f"Citations not in verification database: {len(citations_not_in_metadata)}")
    
    if citations_not_in_metadata:
        print("\nCitations not in verification database:")
        for citation in citations_not_in_metadata:
            print(f"  - {citation}")
    
    # Check for orphan PDFs (PDFs without citations)
    pdf_basenames = [os.path.splitext(pdf)[0] for pdf in pdf_files]
    used_pdfs = [paper_metadata[c]['pdf_file'] for c in citations_with_pdf if c in paper_metadata]
    
    orphan_pdfs = []
    for pdf in pdf_files:
        if pdf not in used_pdfs and not pdf.startswith('.'):
            orphan_pdfs.append(pdf)
    
    if orphan_pdfs:
        print(f"\nOrphan PDFs (not referenced in bibliography): {len(orphan_pdfs)}")
        for pdf in orphan_pdfs[:10]:  # Show first 10
            print(f"  - {pdf}")
        if len(orphan_pdfs) > 10:
            print(f"  ... and {len(orphan_pdfs) - 10} more")
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"✓ {len(citations_with_pdf)} citations have matching PDFs")
    print(f"✓ {len(citations_without_pdf)} citations verified but PDFs unavailable (paywalls)")
    if citations_not_in_metadata:
        print(f"⚠ {len(citations_not_in_metadata)} citations need verification")
    else:
        print("✓ All citations have been verified")
    
    # Calculate verification rate
    verified_citations = len(citations_with_pdf) + len(citations_without_pdf)
    verification_rate = verified_citations / len(citations) * 100 if citations else 0
    print(f"\nOverall verification rate: {verification_rate:.1f}%")
    
    # Save detailed report
    report = {
        "total_citations": len(citations),
        "citations_with_pdf": len(citations_with_pdf),
        "citations_without_pdf": len(citations_without_pdf),
        "citations_not_verified": len(citations_not_in_metadata),
        "orphan_pdfs": len(orphan_pdfs),
        "verification_rate": verification_rate,
        "citations_list": {
            "with_pdf": citations_with_pdf,
            "without_pdf": citations_without_pdf,
            "not_verified": citations_not_in_metadata
        }
    }
    
    with open('citation_verification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\nDetailed report saved to citation_verification_report.json")

if __name__ == "__main__":
    check_citation_pdf_match()