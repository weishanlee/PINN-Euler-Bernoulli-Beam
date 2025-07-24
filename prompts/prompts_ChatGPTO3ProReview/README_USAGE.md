# ChatGPTO3ProReview Usage Guidelines

## Purpose
These prompts are designed for peer review of already-generated academic content, NOT for initial content generation.

## Usage Scenarios

### ✅ ALLOWED Uses

1. **External Review with ChatGPT O3 Pro**
   - User generates content with Claude Code
   - User copies review prompts to ChatGPT O3 Pro
   - ChatGPT O3 Pro performs independent review
   - User brings review back to Claude Code for revisions

2. **Internal Review with Claude Code**
   - User generates content with Claude Code
   - User explicitly requests: "Please review my [section].tex using ChatGPTO3ProReview prompts"
   - Claude Code accesses the review prompt and performs review
   - Review is saved to `output/review_reports/`

### ❌ PROHIBITED Uses

1. **During Initial Content Generation**
   - DO NOT use these prompts when writing new .tex files
   - DO NOT use these prompts to guide initial content creation
   - DO NOT use these prompts as templates for generating sections

## Example Requests

### Correct Usage Examples:
- "Please review my introduction.tex using the ChatGPTO3ProReview prompt"
- "Can you perform a peer review of my methods section?"
- "Review my main.tex file using the review prompts"
- "I received this review from ChatGPT O3 Pro, can you save it?"

### Incorrect Usage Examples:
- "Write an introduction following the review criteria"
- "Generate a methods section using the review prompt guidelines"
- "Create content that would pass the review criteria"

## Review Output Structure

All reviews are saved to `output/review_reports/` with standardized naming:
- Section reviews: `[section]_peer_review.md`
- Main manuscript: `main_peer_review.md`
- Revisions: `revision_[timestamp].md`

## Key Principle
**Review prompts evaluate existing content; they do not generate new content.**