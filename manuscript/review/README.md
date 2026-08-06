# Review manuscript build

This directory contains a review-oriented assembly of the main manuscript.
It preserves the current text and the three generated main tables, resolves
citations from `manuscript/references.bib`, and appends the five main figures
with their validated captions in numerical order.

Unresolved `TODO[...]` markers are deliberately retained so that missing
author, acquisition, repository, license, DOI, funding, and declaration
information remains visible during review.

## Build

From the repository root:

```bash
./.venv/bin/python manuscript/review/assemble_review_markdown.py
```

Then run the local Pandoc installation with cite processing:

```text
C:\Users\rlewa\AppData\Local\Pandoc\pandoc.exe
  manuscript\review\article_review.md
  --from=markdown
  --citeproc
  --bibliography=manuscript\references.bib
  --resource-path=.
  --toc
  --toc-depth=2
  -o manuscript\review\trackleed_manuscript_review.docx
```

`source_manifest.json` records hashes of the manuscript, bibliography, figure
images, caption sources, assembly script, and generated review Markdown.
