# alimiraftab.github.io

Personal website of **Ali Miraftab** — Lead AI/ML Engineer / Scientist.
Built with Jekyll and hosted on GitHub Pages.

🔗 Live: https://alimiraftab.github.io

---

## Tech stack

- **Jekyll** via the [`github-pages`](https://github.com/github/pages-gem) gem (keeps local builds in lockstep with what GitHub Pages runs)
- **kramdown** (GFM input) for Markdown, **Rouge** for syntax highlighting
- **Mermaid** (diagrams) and **MathJax** (LaTeX) loaded on demand from a CDN
- Custom CSS in `assets/css/main.css` — dark-first with a light theme toggle

---

## Local development

Requires Ruby (see `.ruby-version`, currently `3.3.5`) and Bundler.

```bash
# one-time, or after editing the Gemfile
bundle install

# run the dev server
bundle exec jekyll serve --livereload --future
```

Then open http://localhost:4000.

- `--future` is **required** to preview posts dated in the future (the blog series uses forward dates so it sorts in reading order).
- `--livereload` refreshes the browser on save (note: `_config.yml` changes still need a manual restart).
- The stylesheet link is cache-busted with `?v=<build-time>`, so after deploying you don't need to hard-refresh to see CSS changes.

---

## Project structure

```
.
├── _config.yml                      # Site config (title, permalinks, kramdown, excludes)
├── _layouts/
│   ├── default.html                 # Base layout: nav, footer, theme toggle, Mermaid + MathJax loaders
│   └── post.html                    # Blog post layout
├── index.md                         # Home page
├── about/index.md                   # About page
├── cv/index.md                      # CV page
├── blogs/index.md                   # "Writing" index (post list + topic filters)
├── retrieval-ranking-recommendation/
│   └── index.md                     # Hub page for the RecSys series
├── _posts/                          # All blog posts (Markdown)
├── assets/css/main.css              # All styles
├── Gemfile                          # github-pages gem
└── old/                             # Archived previous site (excluded from build)
```

---

## Writing a post

Create a file in `_posts/` named `YYYY-MM-DD-title.md` with front matter:

```yaml
---
layout: post
title: "My Post Title"
date: 2026-01-15 09:00:00 -0500
topic: LLMs            # drives the filter chips on the Writing page
description: "One-line summary used for SEO and previews."
---
```

The post body is Markdown. The page title comes from `title`, so don't repeat it as an `# H1` in the body.

### Authoring features

| Feature | How to use |
|---|---|
| **Diagrams** | A fenced ` ```mermaid ` block. Rendered client-side by Mermaid. |
| **Math** | Inline `$ ... $` and display `$$ ... $$`. Rendered by MathJax. Avoid stray/literal `$` in prose (use "dollars" or `\$`) so it isn't read as math. |
| **Code** | Fenced blocks with a language, e.g. ` ```python `. Highlighted by Rouge using the theme in `main.css`. |
| **Tables** | Standard Markdown tables. Styled automatically inside posts. |

---

## The "Retrieval, Ranking & Recommendation" series

A 24-part series lives in `_posts/` (filenames prefixed `rrr-NN-…`) with a dedicated hub at
`/retrieval-ranking-recommendation/`. Series posts carry extra front matter so the hub can list
them in order:

```yaml
---
layout: post
title: "18 — The Cold Start Problem"
date: 2026-04-13 09:00:00 -0500
topic: RecSys
series: rrr            # marks membership in the series
order: 18              # reading-order position (1–24)
theme: "New users/items"   # short theme label shown on the hub
description: "..."
---
```

The hub page (`retrieval-ranking-recommendation/index.md`) pulls every post where
`series == "rrr"`, sorts by `order`, and renders the card list. To add a part, just create a post
with these fields — no edits to the hub are needed.

---

## Deployment

Push to the default branch; GitHub Pages builds and deploys automatically.

```bash
git add -A
git commit -m "..."
git push
```

Allow a minute for the rebuild. If something looks stale, the cache-busted CSS link usually handles
it; otherwise hard-refresh once (`Cmd/Ctrl+Shift+R`).
