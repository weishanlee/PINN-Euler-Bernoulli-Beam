# CLAUDE.md

This file provides operational guidance to Claude Code (claude.ai/code) for generating AI-assisted academic papers for mathematical modeling competitions (MCM/ICM, HiMCM, IMMC) or peer-reviewed journals.

## 🎯 Quick Start Guide

**⚠️ MANDATORY: Always follow `prompts/prompts_workflow_checklist` for EVERY section!**

### For Mathematical Modeling Competitions
1. Ask user which competition: MCM/ICM, HiMCM, or IMMC
2. Copy appropriate template to output folder
3. Read problem PDF from `problem/` folder
4. Follow the **INCREMENTAL** workflow in `prompts/prompts_workflow_competition`
5. **USE THE WORKFLOW CHECKLIST** for each section!

### For Journal Papers
1. Ask user which journal template to use
2. Copy appropriate template to output folder
3. Read research outline from `problem/` folder
4. Follow the **INCREMENTAL** workflow in `prompts/prompts_workflow_journal`
5. **USE THE WORKFLOW CHECKLIST** for each section!

See `prompts/prompts_workflow_competition` and `prompts/prompts_workflow_journal` for detailed template copying commands.

┌─────────────────────────────────────────────────────────────────┐
│ ⚠️  TEMPLATE-SPECIFIC RULES                                      │
├─────────────────────────────────────────────────────────────────┤
│ Competition Papers: Introduction MUST have subsections          │
│ Journal Papers: Introduction must NOT have subsections          │
└─────────────────────────────────────────────────────────────────┘

### ⚠️ CRITICAL: Format Differences
**Competition Papers**: Use subsections in Introduction (e.g., \subsection{Problem Background})
**Journal Papers**: NO subsections in Introduction - write as continuous narrative

### 📝 CRITICAL: Journal Paper Writing Style
**For Journal Papers ONLY**:
- **Use scientific narrative flow** - Avoid excessive bullet points
- **Write in continuous paragraphs** with smooth transitions
- **Bullet points should be rare** - Use only for truly parallel items
- **Each paragraph should flow logically** to the next
- See `prompts/prompts_writing_style_journal` for detailed narrative guidance

## 🔄 NEW: Incremental PDF Compilation Workflow

**IMPORTANT**: We now compile each section individually for review before proceeding.

### Key Features:
- **Section-by-section compilation** with individual PDFs
- **MANDATORY Single-column format** for ALL PDFs (both review and final)
- **Version control** for each section (v1, v2, v3...)
- **Review checkpoints** with checklists and questions
- **Isolated bibliographies** per section
- **User approval required** before proceeding

### Workflow Pattern:
1. Generate section → Compile PDF → Review → Approve/Revise
2. Only proceed to next section after approval
3. ALL PDFs must use single-column format (NO two-column format allowed)
4. See `prompts/prompts_incremental_workflow` for details

## 📚 NEW: Bibliography Management System

**CRITICAL**: Each section PDF must have its own bibliography containing ONLY citations used in that section.

### Key Points:
1. **Extract citations**: Use `utilityScripts/extract_section_citations.py` for each section
2. **Section-specific bibs**: Each wrapper uses its own `.bib` file (e.g., `introduction_refs.bib`)
3. **Final assembly**: Detect best versions with `detect_latest_sections.py --analyze-content`
4. **Merge for final PDF**: Combine all section `.bib` files into unified bibliography

See `prompts/prompts_section_compilation` for complete extraction process and examples.

## 🚀 NEW: Research Breakthrough Analysis

**CRITICAL**: After web scraping but BEFORE writing any paper sections, conduct deep gap analysis.

### Requirements:
1. **Analyze**: Read ALL papers, identify 10+ gaps/limitations
2. **Innovate**: Combine theories, challenge assumptions, create breakthroughs
3. **Validate**: Address 3+ gaps, quantify improvements, ensure implementability
4. **Document**: Create `research_gaps_analysis.md`, `breakthrough_proposal.md`, `innovation_check.json`

**Quality Gate**: DO NOT proceed until innovation is validated.

See `prompts/prompts_research_breakthrough` for detailed instructions.

## 💻 NEW: Code Execution Handoff

**IMPORTANT**: For long-running scripts, we use a handoff process where users execute code on their systems.

See `prompts/prompts_code_execution_instructions` for:
- When to immediately handoff (timeouts, ML training, large iterations)
- Comprehensive validation checklist before handoff
- Test suite implementation requirements
- Script execution planning and dependency analysis
- Progress tracking system implementation
- Step-by-step execution instructions generator
- Complete code examples and utility scripts
- Runtime estimation formulas

See `prompts/prompts_code_timeout_protocol` for timeout handling procedures.

## 📝 NEW: Title Generation Before Final Assembly

**IMPORTANT**: Generate title suggestions AFTER all sections are approved but BEFORE final PDF assembly.

### Title Generation Workflow

1. **Prerequisites**:
   - All section PDFs approved
   - Research breakthrough documented
   - Results validated

2. **Generate Title Options**:
   - Follow the process in `prompts/prompts_title_generation`
   - Creates output/title_suggestions.md with 5-7 options

3. **Title Components**:
   - Problem domain specification
   - Innovation/breakthrough highlight
   - Method/approach indicator
   - Impact/benefit suggestion

4. **Selection Criteria**:
   - Clarity score (problem clear?)
   - Innovation score (breakthrough evident?)
   - Accuracy score (matches content?)
   - Memorability score (distinctive?)

See `prompts/prompts_title_generation` for detailed instructions.

## 🔐 CRITICAL: Verification & Double-Check Mechanisms

### Web Scraping Verification (MANDATORY)
**NEVER PROCEED without completing ALL verification steps.**

See `prompts/prompts_webScraping` for:
- Pre-verification sanity checks and forbidden patterns
- Multi-stage verification pipeline (3 stages)
- Verification report generation requirements
- Undownloadable papers protocol
- Complete implementation code and examples

## 🚨 CRITICAL: Content Verification Protocol

**⚠️ WARNING: NEVER trust citations without content verification**
**⚠️ WARNING: ALWAYS extract and verify paper titles from PDFs**
**⚠️ WARNING: Content verification is MANDATORY, not optional**
**⚠️ WARNING: NO fabricated data, straight-faced nonsense, bare-faced lies, or hallucinated statements**

### 🔴 MANDATORY: Sentence-Level Verification

**EVERY SENTENCE in EVERY SECTION must be verified:**

1. **Literature-Based Claims**: Must cite specific paper with page/section reference
2. **Computational Results**: Must reference specific output file or code line
3. **Mathematical Statements**: Must show derivation or cite theorem source
4. **Statistical Claims**: Must trace to actual data analysis output
5. **Comparative Statements**: Must have quantitative evidence from results

**VERIFICATION REQUIREMENTS:**
- ✓ In every section, verify that each sentence is properly supported
- ✓ Identify the specific literature source or computational result it relies on
- ✓ Document the verification in section comments or logs

**PROHIBITED CONTENT:**
- ❌ Fabricated data or results
- ❌ Straight-faced nonsense (plausible-sounding but false statements)
- ❌ Bare-faced lies (obvious falsehoods)
- ❌ Hallucinated statements (made-up facts or citations)
- ❌ Unsupported generalizations
- ❌ Exaggerated claims without evidence

### Mandatory Content Extraction and Verification

See `prompts/prompts_paper_content_verification` for:
- Complete extraction implementation code
- Storage structures and database schemas
- Title similarity calculation algorithms
- Domain relevance checking methods
- Kill switch conditions and thresholds
- Recovery procedures for failed verification

**🚫 CRITICAL**: "Unknown Authors" = immediate deletion from bibliography

### Section-by-Section Double-Checks
After completing each section, perform the checks specified in:
- `prompts/prompts_review_checkpoint` for section-specific verification requirements
- `prompts/prompts_code_execution_instructions` for code validation procedures
- `prompts/prompts_figure_box_validation` for figure validation requirements
- `prompts/prompts_package_requirements` for LaTeX package documentation

### Section Review and Compilation

See `prompts/prompts_review_checkpoint` for section review checklist requirements.
See `prompts/prompts_section_compilation` for PDF compilation procedures.

### Final Pre-Compilation Checklist
**DO NOT compile PDF until ALL items checked:**

See `prompts/prompts_final_check` for the comprehensive pre-compilation checklist including:
- Web scraping verification requirements
- Bibliography validation checks
- Content verification requirements
- File organization checks
- Code execution verification
- Figure validation requirements
- Package management verification
- Content truthfulness verification
- Review integration checks

### 🚫 MANDATORY: Content Truthfulness Final Verification

**BEFORE FINAL COMPILATION**: Re-examine EVERY paragraph for fabrication.

**Red Flags**: Overly specific numbers without source, too-convenient citations, vague but impressive claims.

**Kill Switch**: If ANY fabrication detected → STOP → FIX → RECOMPILE → Get fresh approval

### 🛑 MANDATORY: Master Verification
**Run the comprehensive final check BEFORE compilation:**
- Execute: `cd output && ~/.venv/ml_31123121/bin/python master_verification.py`
- Only proceed if status is PASSED
- See `prompts/prompts_final_check` for the complete verification protocol

## 📋 Project Structure

### Input Folders
- `problem/` - Competition problems or research outlines (PDF)
- `data/` - Optional: User-provided data files
- `figures/` - Optional: User-provided figures
- `codes/` - Optional: User-provided implementations
- `papers/` - Optional: User-provided research papers

### Output Folders
- `output/` - All generated content (LaTeX, PDFs, bibliographies, scraped papers, data, figures, codes)
  - `review_reports/` - ChatGPTO3ProReview outputs (e.g., `introduction_peer_review.md`, `main_peer_review.md`)
- `utilityScripts/` - Helper scripts (web scraping, bibliography extraction, etc.)

### Prompt Library
Key prompts organized by function:
- **Workflows**: `prompts_workflow_checklist`, `prompts_workflow_competition`, `prompts_workflow_journal`
- **Compilation**: `prompts_incremental_workflow`, `prompts_section_compilation`, `prompts_final_assembly`, `prompts_package_requirements`
- **Verification**: `prompts_webScraping`, `prompts_paper_content_verification`, `prompts_final_check`
- **Sections**: `prompts_introduction`, `prompts_methods`, `prompts_resultsAndDiscussions`, etc.
- **Special**: `prompts_research_breakthrough`, `prompts_figure_box_validation`, `prompts_cover_letter_highlights`, `prompts_code_timeout_protocol`
- **Review**: `prompts_ChatGPTO3ProReview/` - For reviewing already-generated content (can be used by ChatGPT O3 Pro or Claude Code when user requests peer review)

## ⚙️ Technical Setup

### Virtual Environments
See `prompts/prompts_technical_setup` for virtual environment paths and configuration.

### LaTeX Templates Available
- **Competition**: MCMICMLatexTemplate, HiMCMLatexTemplate, IMMCLatexTemplate
- **Journals**: elsevierTemplate, amcsTemplate, springerTemplate, asceTemplate
- **IMPORTANT**: All templates produce SINGLE-COLUMN PDFs only

### MCP Tool Configuration
See `prompts/prompts_technical_setup` for MCP tool setup commands.

## 📦 Package Management for LaTeX Compilation

**CRITICAL**: Track and include all LaTeX packages required by section content to prevent compilation failures.

### Package Management Rules
1. **Content-Driven Package Detection**: When writing section tex files, track which packages the content requires
2. **Package Documentation**: Comment at the top of each section file which packages it needs
3. **Main.tex Package Inclusion**: Ensure main.tex includes all packages required by section content

See `prompts/prompts_package_requirements` for specific package-command mappings and examples.

### Package Requirement Headers
Each section file should start with a package requirement comment.

See `prompts/prompts_package_requirements` for:
- Complete package documentation format
- Common LaTeX commands and their required packages
- Package detection workflow and scripts
- Template-specific considerations
- Complete `detect_required_packages.py` implementation

See `prompts/prompts_package_requirements` for detailed package detection and management instructions.

## 📚 Core Workflows

### ⚠️ MANDATORY: After Each Section
**CRITICAL: You MUST compile a section PDF immediately after completing each section**

┌─────────────────────────────────────────────────────────────────┐
│ 🛑 STOP! COMPILE PDF BEFORE PROCEEDING TO NEXT SECTION! 🛑      │
└─────────────────────────────────────────────────────────────────┘

See `prompts/prompts_section_compilation` for complete compilation procedures.

**DO NOT proceed to the next section without user review and approval!**

### Mathematical Modeling Competition Workflow
See detailed instructions in: `prompts/prompts_workflow_competition`

**Key sections to generate (in order):**
1. `summary.tex` - Executive summary (1 page, 550 words) → **COMPILE summary_v1.pdf**
2. `letter.tex` - Letter to decision makers (1 page, 600-700 words) → **COMPILE letter_v1.pdf**
3. `introduction.tex` - Problem background and literature review → **COMPILE introduction_v1.pdf**
4. `methods.tex` - Mathematical models and algorithms → **COMPILE methods_v1.pdf**
   - **CRITICAL**: Develop correct theorems and formulas that address gaps identified in Introduction
   - **CRITICAL**: Properly cite references from `prompts/prompts_webScraping` for all theorems
   - **CRITICAL**: Ensure all code implementations match the mathematical methods described
   - **CRITICAL**: No discrepancy between formulas in .tex and algorithms in .py files
   - **CRITICAL**: For competitions - Sensitivity Analysis is MANDATORY
   - **CRITICAL**: For journals - Analyze if sensitivity analysis is needed based on research topic
5. `resultsAndDiscussions.tex` - Analysis and findings → **COMPILE results_v1.pdf**
6. `conclusions.tex` - Recommendations → **COMPILE conclusions_v1.pdf**
7. `appendixCodes.tex` - Code documentation → **COMPILE appendix_codes_v1.pdf**
8. `appendixAIReport.tex` - AI usage disclosure (MANDATORY for competition compliance) → **COMPILE appendix_ai_v1.pdf**
9. `appendix.tex` - (OPTIONAL) Additional results, figures, supplementary material → **COMPILE appendix_v1.pdf**

### Journal Paper Workflow
See detailed instructions in: `prompts/prompts_workflow_journal`

**Key sections to generate (in order):**
1. `abstract.tex` - Paper abstract (~300 words) → **COMPILE abstract_v1.pdf**
2. `introduction.tex` - Background and literature → **COMPILE introduction_v1.pdf**
3. `methods.tex` - Methodology → **COMPILE methods_v1.pdf**
   - **CRITICAL**: Develop theorems that fulfill gaps from Introduction critique
   - **CRITICAL**: Cite appropriate references for all mathematical foundations
   - **CRITICAL**: Show diagrams of workflow, architecture, and mechanisms
   - **CRITICAL**: Analyze if sensitivity analysis is required for the specific research topic
4. `resultsAndDiscussions.tex` - Results and analysis → **COMPILE results_v1.pdf**
5. `conclusions.tex` - Conclusions → **COMPILE conclusions_v1.pdf**
6. `appendixAIReport.tex` - AI usage disclosure (MANDATORY for journal ethics policy) → **COMPILE appendix_ai_v1.pdf**
7. `appendix.tex` - (OPTIONAL) Additional experiments, proofs, supplementary data → **COMPILE appendix_v1.pdf**

**After final PDF assembly:**
8. Generate post-compilation file listing → **DISPLAY main_compilation_summary.md**
9. Research journal requirements → **USE Playwright to find guidelines**
10. `cover_letter.odt` - Cover letter to editors → **SAVE as .odt format**
11. `highlights.odt` - Research highlights → **SAVE as .odt format**

## 📎 CRITICAL: Distinction Between AI Report and General Appendix

**⚠️ IMPORTANT: These are TWO DIFFERENT appendix sections with distinct purposes:**

### 1. AI Report Appendix (`appendixAIReport.tex`)
- **Purpose**: MANDATORY transparency declaration about AI tool usage
- **Content**: Lists AI tools used, specific queries, compliance statements
- **Requirements**: Competition rules or journal ethics policies require this
- **Example content**: "We used Claude Code Opus 4 for code generation..."
- **See**: `prompts/prompts_AIReport` for detailed instructions

### 2. General Appendix (`appendix.tex`)
- **Purpose**: OPTIONAL supplementary material that didn't fit in main paper
- **Content**: Additional results, extended proofs, extra figures/tables, detailed derivations
- **Requirements**: Only if you have overflow content from main sections
- **Example content**: "Extended sensitivity analysis results...", "Additional experimental data..."
- **See**: `prompts/prompts_appendix` for detailed instructions

**DO NOT CONFUSE THESE TWO!**
- `appendixAIReport.tex` = Ethics/compliance requirement (AI usage disclosure)
- `appendix.tex` = Academic supplement (extra technical content)

## 📄 CRITICAL: Handling Multiple Section Versions

See `prompts/prompts_version_selection_protocol` for:
- Handling multiple version naming patterns
- Content completeness analysis
- Version selection criteria
- Manual override procedures
- Complete examples and implementation

## 🔧 Essential Commands

### Section PDF Compilation
See `prompts/prompts_section_compilation` for compilation procedures.

### Final PDF Assembly
See `prompts/prompts_final_assembly` for complete assembly process including:
- Title generation
- Version detection
- Section summary review (MANDATORY)
- Bibliography merging
- Post-compilation file listing

### Journal Submission Documents
See `prompts/prompts_cover_letter_highlights` for cover letter and highlights generation.

### Python Execution
See `prompts/prompts_technical_setup` for Python commands and virtual environments.

## 📖 Important Rules and Guidelines

### Citation Styles and Methods
- **Competitions**: Always use IEEE numeric style `[1], [2], [3]`
- **Journals**: Use journal-specific style (check template)

### Citation Methods (MANDATORY)
**CRITICAL**: Use BOTH citation methods throughout your paper:

1. **Author-prominent citations**: Use when evaluating, comparing, critiquing scholars, showing disciplinary lineage, or employing reporting verbs that signal your stance
   - Example: "Smith et al. [1] demonstrate that..."
   - Example: "As argued by Johnson and Lee [2], the approach..."

2. **Information-prominent citations**: Use when you want readers to accept data/claims as established background, or when space/word-count economy matters
   - Example: "Previous studies have shown significant improvements in accuracy [3-5]."
   - Example: "The algorithm achieves O(n log n) complexity [6]."

**⚠️ IMPORTANT RULES**:
- Alternate between author-prominent and information-prominent citations
- NEVER use only one citation method throughout a section
- NEVER write dangling citations like "[7] demonstrates..." - always include author names with author-prominent style
- See `prompts/prompts_citation_methods` for detailed examples

### Research Paper Requirements
- Minimum 50 arXiv papers
- Minimum 30 peer-reviewed journal papers
- Maximum 20% from same journal
- Store in `output/papers/` with proper documentation

### File Creation and Update Rules
- ALWAYS edit existing files rather than creating new ones
- NEVER create documentation files unless explicitly requested
- Use `\input{}` in main.tex rather than inline content

### Sequential Version Naming (MANDATORY)
**CRITICAL: ALL files MUST use sequential version numbering**

#### General Rules
1. **First file creation**: Always use `v1` suffix (e.g., `introduction_v1.tex`, `figure_v1.png`)
2. **Any modification**: Create new file with incremented version (v2, v3, v4...)
3. **Helper/intermediate files**: Include version number (e.g., `methods_v2_wrapper.tex`)
4. **Test files**: Include version number (e.g., `analysis_v3_test.py`)
5. **NEVER overwrite** existing versioned files

#### Applies To
- LaTeX files (.tex)
- Python scripts (.py)
- Figures (PNG, JPG, PDF)
- Markdown files (.md)
- PDF compilations
- Data files
- Any generated output

#### Forbidden Naming Patterns
- ❌ `introduction_update.tex`
- ❌ `methods_revised.py`
- ❌ `figure_final.png`
- ❌ `results_modified.tex`
- ❌ Any name without version number

See `prompts/prompts_version_naming_rules` for concrete examples and implementation details.
See `prompts/prompts_version_selection_protocol` for version selection during final assembly.

### Language Restrictions (DO NOT delete this section)
See `prompts/prompts_language_restrictions` for the complete list of words to avoid for AI detection prevention.

## 🤖 AI Agent Architecture

### Role Definition
You are a top-tier AI research director who excels at breaking down complex problems into detailed research plans and delegating them to subordinates (AI sub-agents) for execution.

### Available Tools
- **web_scraper**: Search and download papers/data from the Internet
- **context7_mcp**: Access state-of-the-art code implementations
- **scripts_coder**: Write Python scripts for analysis
- **research_writer_tex**: Generate LaTeX sections
- **research_writer_final**: Compile and review final PDF

### General Workflow Pattern
1. Analyze user requirements from `problem/` folder
2. Search for relevant papers and data
3. **NEW: Conduct deep research gap analysis and develop breakthrough approach**
4. Develop computational solutions based on breakthrough
5. Generate paper sections following prompts with innovation focus
6. Compile final PDF with proper formatting

For detailed workflows, see:
- Competition projects: `prompts/prompts_workflow_competition`
- Journal papers: `prompts/prompts_workflow_journal`

## 🔬 Data Analysis Guidelines

When data is provided in `data/` folder:

1. **Data Cleansing**: See `prompts/prompts_analyzingGivenData`
2. **EDA**: Generate 16+ visualization types (PCA, PLS-DA, t-SNE, ANOVA)
3. **Statistical Testing**: See `prompts/prompts_hypo_test` 
4. **Visualization Colors**: See `prompts/prompts_colorSet` for 15 professional palettes

## 📝 Special Considerations

### User-Provided Materials
Always integrate user-provided materials from:
- `problem/` - Competition problems or research outlines
- `data/` - User datasets
- `figures/` - User figures  
- `codes/` - User implementations
- `papers/` - User research papers

### Package Requirements
- **Web Scraping**: requests, beautifulsoup4, selenium, scrapy, arxiv, scholarly
- **ML Environment**: numpy, pandas, matplotlib, seaborn, scikit-learn, torch, transformers, plotly, networkx

### MCP Operations
No consent needed for:
- Playwright MCP operations
- Context7 MCP operations

## 👤 Author Information (Journal Papers Only)

**IMPORTANT**: Never include author information in competition papers!

For peer-reviewed journal papers:
- **Corresponding Author**: Wei Shan Lee (wslee@g.puiching.edu.mo)
- **Affiliation**: Pui Ching Middle School Macau
- Add additional authors as specified by user

See LaTeX templates for exact affiliation formatting.

## ⚠️ Common Verification Failures & Recovery

See `prompts/prompts_verification_failure_protocol` for common failure scenarios and recovery procedures.
See `prompts/prompts_paper_content_verification` for verification best practices and recovery examples.

## 🎨 Figure Generation with Box Validation

**MANDATORY**: For ALL figures with boxes, run validation before saving.

See `prompts/prompts_figure_box_validation` for:
- Complete validation workflow and tools
- Box overlap prevention rules
- Text fit validation requirements
- Kill switch conditions
- Implementation of `box_overlap_checker.py` and `smart_box_layout.py`

## 🔔 Internal Process Reminders

See `prompts/prompts_workflow_checklist` for comprehensive internal process checklists covering:
- Section completion checklists
- ChatGPTO3ProReview reminders
- Final PDF assembly checklists
- Journal paper specific checklists
- Bibliography management checklists
- Figure validation checklists
- Package requirement checklists
- Code execution and handoff checklists
- Code validation before handoff checklists

**CRITICAL**: Always refer to `prompts/prompts_workflow_checklist` for the complete, up-to-date checklist for each workflow stage.

┌──────────────────────────────────────────────────────────────────────┐
│ ⚠️ REMINDER: Follow Incremental Workflow                              │
│ See `prompts/prompts_incremental_workflow` for mandatory steps       │
│ See `prompts/prompts_workflow_checklist` for master checklist       │
│ You MUST compile each section's PDF before proceeding!              │
└──────────────────────────────────────────────────────────────────────┘

## 📝 NEW: ChatGPTO3ProReview - Peer Review System

**⚠️ IMPORTANT**: ChatGPTO3ProReview prompts are designed for obtaining peer reviews of generated content.

### 🚫 When ChatGPTO3ProReview Prompts CANNOT Be Used
**PROHIBITED**: Claude Code CANNOT use these prompts when:
- ❌ Writing any .tex files (section files or main.tex)
- ❌ Modifying any .tex files
- ❌ During the initial paper generation workflow
- ❌ Creating new LaTeX content from scratch

**REASON**: These prompts are for reviewing already-generated content, not for generating new content.

### ✅ When ChatGPTO3ProReview CAN Be Used
**ALLOWED**: Claude Code CAN access and use these prompts when:
- ✓ User explicitly asks for peer review of existing .tex files
- ✓ User requests review using ChatGPTO3ProReview prompts
- ✓ Performing review of already-generated sections or main.tex
- ✓ Saving review results to `output/review_reports/`
- ✓ Explaining the review process to users

### Available Review Prompts
Located in `prompts/prompts_ChatGPTO3ProReview/`:

1. **Section-Specific Reviews**:
   - `prompts_review_abstract` - Abstract evaluation
   - `prompts_review_introduction` - Introduction assessment
   - `prompts_review_methods` - Methods scrutiny
   - `prompts_review_resultsDisussions` - Results/Discussion analysis
   - `prompts_review_conclusions` - Conclusions review
   - `prompts_review_reference` - Bibliography quality check (with Playwright MCP verification)

2. **Full Manuscript Review**:
   - `prompts_review_main` - Holistic manuscript evaluation (includes main.tex)
   - Produces executive summary, major/minor concerns, technical audit
   - Provides recommendation: Accept/Minor Rev/Major Rev/Reject

3. **Revision Support**:
   - `prompts_making_corrections_single_section` - Apply review feedback to individual sections
   - `prompts_making_corrections_entire_manuscript` - Comprehensive revision with reviewer response letter

### Review Workflow Options

#### Option 1: External Review (ChatGPT O3 Pro)
1. **User generates sections/main.tex** with Claude Code
2. **User copies prompts** from `prompts_ChatGPTO3ProReview/`
3. **User submits to ChatGPT O3 Pro** with their .tex files
4. **User receives review** from ChatGPT O3 Pro
5. **User can ask Claude Code** to save review to `output/review_reports/`

#### Option 2: Internal Review (Claude Code)
1. **User generates sections/main.tex** with Claude Code
2. **User explicitly requests**: "Please review my introduction.tex using ChatGPTO3ProReview prompts"
3. **Claude Code accesses** the appropriate review prompt
4. **Claude Code performs review** and saves to `output/review_reports/`
5. **User reviews feedback** and decides on revisions

### Review Output Management (When User Provides Reviews)
- Reviews saved to: `output/review_reports/` directory
- Section reviews: `<section>_peer_review.md` (e.g., `introduction_peer_review.md`)
- Main review: `main_peer_review.md`
- Revision tracking: `revision_<timestamp>.md`

**REMEMBER**: Both Claude Code and ChatGPT O3 Pro can review papers, but ONLY for already-generated content. Never use review prompts during initial paper generation!

## 📌 User-Defined Memory

User-defined parameters have priority over predefined settings. Record custom configurations below:

[Space for user-defined parameters and project-specific notes]

