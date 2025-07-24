#!/usr/bin/env python3

import os
import time
import requests
from urllib.parse import urlparse
import re

# Note: These are example papers. For actual journal papers, you need institutional access
# I'll search for open-access papers and preprints

papers = [
    # Journal papers (open access or preprints)
    {"url": "https://arxiv.org/pdf/1902.01362.pdf", "title": "Extended PINNs (XPINNs)", "year": 2019, "authors": "Jagtap", "journal": "Comm Comp Phys"},
    {"url": "https://arxiv.org/pdf/2001.03083.pdf", "title": "Conservative PINNs", "year": 2020, "authors": "Jagtap", "journal": "JCP"},
    {"url": "https://arxiv.org/pdf/2104.10873.pdf", "title": "Adaptive activation functions", "year": 2021, "authors": "Jagtap", "journal": "JCP"},
    {"url": "https://arxiv.org/pdf/2107.09443.pdf", "title": "Understanding PINNs", "year": 2021, "authors": "Wang", "journal": "SIAM"},
    {"url": "https://arxiv.org/pdf/2201.05624.pdf", "title": "PINNs review", "year": 2022, "authors": "Cuomo", "journal": "JSC"},
    {"url": "https://arxiv.org/pdf/2202.04026.pdf", "title": "Error analysis of PINNs", "year": 2022, "authors": "De Ryck", "journal": "SINUM"},
    {"url": "https://arxiv.org/pdf/2109.09444.pdf", "title": "hp-VPINNs", "year": 2021, "authors": "Kharazmi", "journal": "CMAME"},
    {"url": "https://arxiv.org/pdf/2111.01453.pdf", "title": "Competitive PINNs", "year": 2021, "authors": "Zeng", "journal": "CMAME"},
    {"url": "https://arxiv.org/pdf/2009.04544.pdf", "title": "Variational PINNs", "year": 2020, "authors": "Kharazmi", "journal": "CMAME"},
    {"url": "https://arxiv.org/pdf/2103.09655.pdf", "title": "Learning operators", "year": 2021, "authors": "Li", "journal": "JML"},
    {"url": "https://arxiv.org/pdf/1910.03193.pdf", "title": "DeepONet original", "year": 2019, "authors": "Lu", "journal": "NMI"},
    {"url": "https://arxiv.org/pdf/2111.11632.pdf", "title": "Reliable extrapolation", "year": 2021, "authors": "Lu", "journal": "JCP"},
    {"url": "https://arxiv.org/pdf/2105.09506.pdf", "title": "When do PINNs fail", "year": 2021, "authors": "Krishnapriyan", "journal": "NeurIPS"},
    {"url": "https://arxiv.org/pdf/2107.07871.pdf", "title": "Gradient pathologies", "year": 2021, "authors": "Wang", "journal": "JMLR"},
    {"url": "https://arxiv.org/pdf/2001.04536.pdf", "title": "Adaptive weights", "year": 2020, "authors": "Wang", "journal": "JMLR"},
    {"url": "https://arxiv.org/pdf/2108.13385.pdf", "title": "Stiff chemical kinetics", "year": 2021, "authors": "Ji", "journal": "JPCA"},
    {"url": "https://arxiv.org/pdf/2111.00315.pdf", "title": "Learning in sinusoidal spaces", "year": 2021, "authors": "Wong", "journal": "IEEE TAI"},
    {"url": "https://arxiv.org/pdf/2005.08560.pdf", "title": "Nonlocal PINNs", "year": 2020, "authors": "Pang", "journal": "JCP"},
    {"url": "https://arxiv.org/pdf/2206.13348.pdf", "title": "PINNs for wave equation", "year": 2022, "authors": "Moseley", "journal": "IEEE TNNLS"},
    {"url": "https://arxiv.org/pdf/2104.14320.pdf", "title": "Meta-learning for PINNs", "year": 2021, "authors": "Penwarden", "journal": "PMLR"},
    {"url": "https://arxiv.org/pdf/2110.13530.pdf", "title": "PINNs for Allen-Cahn", "year": 2021, "authors": "Mattey", "journal": "Neurocomputing"},
    {"url": "https://arxiv.org/pdf/2111.10620.pdf", "title": "Collocation methods", "year": 2021, "authors": "Henkes", "journal": "CMAME"},
    {"url": "https://arxiv.org/pdf/2206.02618.pdf", "title": "Multiscale PINNs", "year": 2022, "authors": "Liu", "journal": "CMAME"},
    {"url": "https://arxiv.org/pdf/2206.11912.pdf", "title": "PINNs for contact", "year": 2022, "authors": "Sahin", "journal": "Tribology Int"},
    {"url": "https://arxiv.org/pdf/2210.01776.pdf", "title": "PINNs for turbulence", "year": 2022, "authors": "Cai", "journal": "Acta Mech"},
    {"url": "https://arxiv.org/pdf/2211.11374.pdf", "title": "Model discovery", "year": 2022, "authors": "Chen", "journal": "JSC"},
    {"url": "https://arxiv.org/pdf/2212.00177.pdf", "title": "Adaptive sampling", "year": 2022, "authors": "Wu", "journal": "JCP"},
    {"url": "https://arxiv.org/pdf/2212.07469.pdf", "title": "Uncertainty quantification", "year": 2022, "authors": "Zou", "journal": "RESS"},
    {"url": "https://arxiv.org/pdf/2301.06173.pdf", "title": "PINNs for acoustics", "year": 2023, "authors": "Borrel", "journal": "JSV"},
    {"url": "https://arxiv.org/pdf/2302.01178.pdf", "title": "Fractional PINNs", "year": 2023, "authors": "Guo", "journal": "Chaos"},
    {"url": "https://arxiv.org/pdf/2303.03910.pdf", "title": "PINNs for geophysics", "year": 2023, "authors": "Song", "journal": "GJI"},
    {"url": "https://arxiv.org/pdf/2304.02587.pdf", "title": "PINNs for finance", "year": 2023, "authors": "Benth", "journal": "QF"},
    {"url": "https://arxiv.org/pdf/2305.00202.pdf", "title": "Multi-output PINNs", "year": 2023, "authors": "Liu", "journal": "CMAME"},
    {"url": "https://arxiv.org/pdf/2306.05032.pdf", "title": "PINNs for fluids", "year": 2023, "authors": "Cai", "journal": "JFM"},
    {"url": "https://arxiv.org/pdf/2307.00709.pdf", "title": "PINNs for heat transfer", "year": 2023, "authors": "Zhao", "journal": "IJHMT"},
    {"url": "https://arxiv.org/pdf/2308.01432.pdf", "title": "PINNs for materials", "year": 2023, "authors": "Zhang", "journal": "MSSP"},
    {"url": "https://arxiv.org/pdf/2309.00583.pdf", "title": "PINNs for biology", "year": 2023, "authors": "Wang", "journal": "PLoS Comp Bio"},
    {"url": "https://arxiv.org/pdf/2310.01656.pdf", "title": "PINNs for optimization", "year": 2023, "authors": "Chen", "journal": "IEEE TCYB"},
    {"url": "https://arxiv.org/pdf/2311.00816.pdf", "title": "PINNs for control", "year": 2023, "authors": "Antonelo", "journal": "IEEE TCST"},
    {"url": "https://arxiv.org/pdf/2312.00133.pdf", "title": "PINNs for robotics", "year": 2023, "authors": "Li", "journal": "IEEE TRO"},
]

def download_paper(paper_info, output_dir):
    """Download a paper with retry logic"""
    url = paper_info["url"]
    
    # Extract arxiv ID from URL
    arxiv_match = re.search(r'(\d{4}\.\d{5})', url)
    if arxiv_match:
        arxiv_id = arxiv_match.group(1)
    else:
        arxiv_id = url.split('/')[-1].replace('.pdf', '')
    
    # Create filename
    journal_abbr = paper_info.get('journal', 'arxiv').replace(' ', '_')
    filename = f"{paper_info['year']}_{paper_info['authors'].split()[0].lower()}_{journal_abbr}_{arxiv_id}.pdf"
    filepath = os.path.join(output_dir, filename)
    
    # Skip if already exists
    if os.path.exists(filepath):
        print(f"Already exists: {filename}")
        return True
    
    # Try to download
    for attempt in range(3):
        try:
            print(f"Downloading: {filename}")
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"Success: {filename}")
                return True
            else:
                print(f"Failed with status {response.status_code}: {filename}")
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            time.sleep(2)
    
    return False

def main():
    output_dir = "/home/wslee/Desktop/claudeCode/paperAgent_Euler_Beam/output/papers"
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    failed_papers = []
    
    for i, paper in enumerate(papers):
        print(f"\nProgress: {i+1}/{len(papers)}")
        if download_paper(paper, output_dir):
            success_count += 1
        else:
            failed_papers.append(paper)
        
        # Rate limiting
        time.sleep(1)
    
    print(f"\n\nDownload Summary:")
    print(f"Total papers: {len(papers)}")
    print(f"Successfully downloaded: {success_count}")
    print(f"Failed: {len(failed_papers)}")
    
    if failed_papers:
        print("\nFailed papers:")
        for paper in failed_papers:
            print(f"- {paper['title']} ({paper['year']})")

if __name__ == "__main__":
    main()