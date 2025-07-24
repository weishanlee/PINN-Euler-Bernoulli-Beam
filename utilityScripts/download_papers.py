#!/usr/bin/env python3
"""
Web scraping script to download research papers from arXiv for the paper agent project.
This script downloads papers related to prompt engineering, Model Context Protocol, and AI agents.
"""

import requests
import os
import time
import json
from typing import List, Dict

# Create output directory for papers
os.makedirs('output/papers', exist_ok=True)

# List of paper URLs to download (from web search results)
papers = [
    # Prompt Engineering Papers
    {"url": "https://arxiv.org/pdf/2501.11613", "title": "Conversation Routines: A Prompt Engineering Framework"},
    {"url": "https://arxiv.org/pdf/2307.12980", "title": "A Systematic Survey of Prompt Engineering on Vision"},
    {"url": "https://arxiv.org/pdf/2503.22978", "title": "Prompt Engineering for Large Language Model-assisted"},
    {"url": "https://arxiv.org/pdf/2310.14735", "title": "Unleashing the potential of prompt engineering in Large"},
    {"url": "https://arxiv.org/pdf/2303.13534", "title": "An Investigation into the Creative Skill of Prompt Engineering"},
    {"url": "https://arxiv.org/pdf/2412.05127", "title": "The Prompt Canvas: A Literature-Based Practitioner Guide"},
    {"url": "https://arxiv.org/pdf/2310.14201", "title": "Prompt Engineering Through the Lens of Optimal Control"},
    {"url": "https://arxiv.org/pdf/2210.15157", "title": "Exploring Prompt Engineering for Solving CS1 Problems"},
    {"url": "https://arxiv.org/pdf/2502.06039", "title": "Benchmarking Prompt Engineering Techniques for Secure"},
    
    # Model Context Protocol Papers
    {"url": "https://arxiv.org/pdf/2503.23278", "title": "Model Context Protocol (MCP): Landscape, Security"},
    {"url": "https://arxiv.org/pdf/2504.08623", "title": "Enterprise-Grade Security for the Model Context Protocol"},
    
    # AI Agents Papers
    {"url": "https://arxiv.org/pdf/2503.12687", "title": "AI Agents: Evolution, Architecture, and Real-World"},
    {"url": "https://arxiv.org/pdf/2407.01502", "title": "AI Agents That Matter"},
    {"url": "https://arxiv.org/pdf/2405.06643", "title": "Levels of AI Agents: from Rules to Large Language Models"},
    {"url": "https://arxiv.org/pdf/2406.08689", "title": "Security of AI Agents"},
    {"url": "https://arxiv.org/pdf/2407.12165", "title": "Building AI Agents for Autonomous Clouds"},
    {"url": "https://arxiv.org/pdf/2506.02055", "title": "Will Agents Replace Us? Perceptions of Autonomous Multi"},
    {"url": "https://arxiv.org/pdf/2505.20273", "title": "Ten Principles of AI Agent Economics"},
    {"url": "https://arxiv.org/pdf/2404.11584", "title": "AI Agents Architecture April 2024"},
    {"url": "https://arxiv.org/pdf/2402.17553", "title": "AI Agents Architecture February 2024"},
    {"url": "https://arxiv.org/pdf/2408.08435", "title": "AI Agents Architecture August 2024"},
]

def download_paper(url: str, title: str) -> Dict:
    """Download a single paper from the given URL."""
    # Extract arXiv ID from URL
    arxiv_id = url.split('/')[-1].replace('.pdf', '')
    filename = f"output/papers/{arxiv_id}_{title[:50].replace(' ', '_').replace('/', '_')}.pdf"
    
    try:
        print(f"Downloading: {title}")
        response = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"✓ Downloaded: {filename}")
            return {"status": "success", "filename": filename, "url": url, "title": title}
        else:
            print(f"✗ Failed to download {title}: HTTP {response.status_code}")
            return {"status": "failed", "url": url, "title": title, "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        print(f"✗ Error downloading {title}: {str(e)}")
        return {"status": "failed", "url": url, "title": title, "error": str(e)}
    
    # Be respectful to the server
    time.sleep(1)

# Download all papers
download_results = []
for paper in papers:
    result = download_paper(paper["url"], paper["title"])
    download_results.append(result)

# Save download log
with open('output/papers/download_log.json', 'w') as f:
    json.dump(download_results, f, indent=2)

# Count successful downloads
successful = len([r for r in download_results if r["status"] == "success"])
print(f"\n✓ Successfully downloaded {successful}/{len(papers)} papers")
print("Download log saved to: output/papers/download_log.json")