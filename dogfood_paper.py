"""
Dogfood test: Run get_paper() on Project2 through Max Router proxy.
Tests: proxy routing, resilient sections, figure generation, LaTeX packages.

Usage:
  cd /Users/speed/Denario
  OPENAI_API_KEY=dummy OPENAI_BASE_URL=http://localhost:3002/v1 python dogfood_paper.py
"""

from denario import Denario, Journal

project_dir = "examples/Project2"
d = Denario(project_dir=project_dir)
d.set_data_description(f"{project_dir}/input.md")

# Run paper generation using claude-sonnet-4-5 through Max Router (OpenAI-compatible proxy).
# Use "sonnet-4-5" as model name — doesn't match gemini/gpt/claude prefixes,
# so it hits the OpenAI-compatible fallback route with OPENAI_BASE_URL.
# Max Router translates the model name to the correct Anthropic model.
# This tests:
# 1. The fallback route (unknown model name -> OpenAI-compatible via ChatOpenAI)
# 2. OPENAI_BASE_URL proxy routing to Max Router
# 3. llm_parser accepting arbitrary model names
# 4. booktabs/float/caption LaTeX packages
# 5. Resilient section_node (try/except)
# 6. generate_figures_node (matplotlib + Nano Banana if GOOGLE_API_KEY set)
d.get_paper(
    journal=Journal.AAS,
    llm="claude-sonnet-4-5-20250929",
    add_citations=False,
)
