#!/usr/bin/env python3

import os
import re
from datetime import datetime

def extract_paper_info(filename):
    """Extract info from filename format: year_author_id.pdf"""
    match = re.match(r'(\d{4})_([^_]+)_(.+)\.pdf', filename)
    if match:
        year, author, paper_id = match.groups()
        return {
            'filename': filename,
            'year': year,
            'author': author.capitalize(),
            'id': paper_id
        }
    return None

def generate_paper_list():
    papers_dir = "/home/wslee/Desktop/claudeCode/paperAgent_Euler_Beam/output/papers"
    pdf_files = sorted([f for f in os.listdir(papers_dir) if f.endswith('.pdf')])
    
    papers_by_category = {
        'PINNs Fundamentals': [],
        'Beam and Fourth-Order PDEs': [],
        'High-Precision Methods': [],
        'Hybrid Architectures': [],
        'Applications': [],
        'Theory and Analysis': []
    }
    
    # Categorize papers based on keywords in filename/id
    for pdf in pdf_files:
        info = extract_paper_info(pdf)
        if not info:
            continue
            
        # Categorization logic
        paper_id = info['id'].lower()
        if any(word in paper_id for word in ['beam', 'biharmonic', 'fourth', 'plate', 'kirchhoff']):
            papers_by_category['Beam and Fourth-Order PDEs'].append(info)
        elif any(word in paper_id for word in ['precision', 'accuracy', 'high-order', 'ultra']):
            papers_by_category['High-Precision Methods'].append(info)
        elif any(word in paper_id for word in ['fourier', 'spectral', 'hybrid', 'wavelet', 'chebyshev']):
            papers_by_category['Hybrid Architectures'].append(info)
        elif any(word in paper_id for word in ['application', 'structural', 'mechanics', 'dynamics']):
            papers_by_category['Applications'].append(info)
        elif any(word in paper_id for word in ['theory', 'analysis', 'convergence', 'error']):
            papers_by_category['Theory and Analysis'].append(info)
        else:
            papers_by_category['PINNs Fundamentals'].append(info)
    
    # Generate report
    with open('/home/wslee/Desktop/claudeCode/paperAgent_Euler_Beam/output/downloaded_papers_list.txt', 'w') as f:
        f.write("DOWNLOADED PAPERS FOR PHYSICS-INFORMED NEURAL NETWORKS RESEARCH\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total papers downloaded: {len(pdf_files)}\n")
        f.write("=" * 70 + "\n\n")
        
        for category, papers in papers_by_category.items():
            if papers:
                f.write(f"\n{category.upper()} ({len(papers)} papers)\n")
                f.write("-" * 50 + "\n")
                for paper in sorted(papers, key=lambda x: x['year'], reverse=True):
                    f.write(f"{paper['year']} - {paper['author']} - {paper['filename']}\n")
        
        f.write("\n\nSUMMARY STATISTICS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total papers: {len(pdf_files)}\n")
        
        # Year distribution
        years = {}
        for pdf in pdf_files:
            info = extract_paper_info(pdf)
            if info:
                year = info['year']
                years[year] = years.get(year, 0) + 1
        
        f.write("\nPapers by year:\n")
        for year in sorted(years.keys(), reverse=True):
            f.write(f"  {year}: {years[year]} papers\n")
        
        f.write("\nPapers by category:\n")
        for category, papers in papers_by_category.items():
            f.write(f"  {category}: {len(papers)} papers\n")

if __name__ == "__main__":
    generate_paper_list()
    print("Paper list generated at: /home/wslee/Desktop/claudeCode/paperAgent_Euler_Beam/output/downloaded_papers_list.txt")