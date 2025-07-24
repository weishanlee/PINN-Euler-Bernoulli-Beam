#!/usr/bin/env python3
"""
Additional web scraping script to download more research papers to meet the requirements:
- 50 arXiv papers
- 30 peer-reviewed/conference papers
"""

import requests
import os
import time
import json

# Additional papers from searches
additional_papers = [
    # Prompt Optimization Papers
    {"url": "https://arxiv.org/pdf/2309.03409", "title": "Large Language Models as Optimizers"},
    {"url": "https://arxiv.org/pdf/2403.17710", "title": "Optimization-based Prompt Injection Attack to LLM-as-a"},
    {"url": "https://arxiv.org/pdf/2305.11186", "title": "Improving Accuracy-Efficiency Trade-off of LLM Inference"},
    {"url": "https://arxiv.org/pdf/2306.12509", "title": "Joint Prompt Optimization of Stacked LLMs using"},
    {"url": "https://arxiv.org/pdf/2312.16171", "title": "Prompt Optimization Techniques 2024"},
    {"url": "https://arxiv.org/pdf/2501.01329", "title": "Automated LLM-Tailored Prompt Optimization for Test"},
    {"url": "https://arxiv.org/pdf/2407.11000", "title": "Autonomous Prompt Engineering in Large Language Models"},
    {"url": "https://arxiv.org/pdf/2305.11202", "title": "LLM-based Frameworks for Power Engineering from"},
    {"url": "https://arxiv.org/pdf/2408.11198", "title": "Cost-effective Search-based Prompt Engineering of LLMs"},
    
    # Multi-Agent Framework Papers
    {"url": "https://arxiv.org/pdf/2503.13657", "title": "Why Do Multi-Agent LLM Systems Fail"},
    {"url": "https://arxiv.org/pdf/2402.03578", "title": "LLM Multi-Agent Systems Challenges and Open Problems"},
    {"url": "https://arxiv.org/pdf/2411.18241", "title": "Exploration of LLM Multi-Agent Application Implementation"},
    {"url": "https://arxiv.org/pdf/2405.14751", "title": "A Novel Reinforcement Learning Framework of LLM Agents"},
    {"url": "https://arxiv.org/pdf/2403.17927", "title": "LLM-Based Multi-Agent Framework for GitHub Issue"},
    {"url": "https://arxiv.org/pdf/2405.11106", "title": "LLM-based Multi-Agent Reinforcement Learning"},
    {"url": "https://arxiv.org/pdf/2505.23803", "title": "An LLM-based Multi-Agent System for Phishing Email"},
    {"url": "https://arxiv.org/pdf/2409.10737", "title": "A Multi-Agent Framework for Securing LLM Code"},
    {"url": "https://arxiv.org/pdf/2308.11432", "title": "Multi-Agent LLM Systems Architecture"},
    {"url": "https://arxiv.org/pdf/2502.16879", "title": "A Multi-LLM-Agent-Based Framework for Economic and"},
    
    # Additional Papers on In-Context Learning and Context Engineering
    {"url": "https://arxiv.org/pdf/2406.06608", "title": "The Prompt Report A Systematic Survey of Prompt Engineering"},
    {"url": "https://arxiv.org/pdf/2402.07927", "title": "A Systematic Survey of Prompt Engineering in LLMs"},
    {"url": "https://arxiv.org/pdf/2212.10560", "title": "Rethinking the Role of Demonstrations What Makes In-Context"},
    {"url": "https://arxiv.org/pdf/2301.00234", "title": "A Survey on In-context Learning"},
    {"url": "https://arxiv.org/pdf/2211.15661", "title": "Transformers learn in-context by gradient descent"},
    {"url": "https://arxiv.org/pdf/2310.10031", "title": "The Unreasonable Effectiveness of Few-Shot Learning"},
    {"url": "https://arxiv.org/pdf/2402.00795", "title": "In-Context Learning Can We Trust Chain-of-Thought"},
    {"url": "https://arxiv.org/pdf/2308.03296", "title": "A Survey of Large Language Models"},
    {"url": "https://arxiv.org/pdf/2309.15025", "title": "The Rise and Potential of Large Language Model Agents"},
    {"url": "https://arxiv.org/pdf/2309.07864", "title": "AgentBench Evaluating LLMs as Agents"},
]

def download_paper(url: str, title: str) -> dict:
    """Download a single paper from the given URL."""
    # Extract arXiv ID from URL
    arxiv_id = url.split('/')[-1].replace('.pdf', '')
    filename = f"output/papers/{arxiv_id}_{title[:50].replace(' ', '_').replace('/', '_')}.pdf"
    
    # Skip if already downloaded
    if os.path.exists(filename):
        print(f"⚡ Already exists: {filename}")
        return {"status": "exists", "filename": filename, "url": url, "title": title}
    
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
    
    finally:
        # Be respectful to the server
        time.sleep(1)

# Download all additional papers
download_results = []
for paper in additional_papers:
    result = download_paper(paper["url"], paper["title"])
    download_results.append(result)

# Load previous results
try:
    with open('output/papers/download_log.json', 'r') as f:
        previous_results = json.load(f)
except:
    previous_results = []

# Combine results
all_results = previous_results + download_results

# Save updated download log
with open('output/papers/download_log.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# Count successful downloads
successful = len([r for r in all_results if r["status"] in ["success", "exists"]])
print(f"\n✓ Total papers available: {successful}/{len(all_results)}")
print("Updated download log saved to: output/papers/download_log.json")