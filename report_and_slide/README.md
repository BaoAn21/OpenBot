# Thesis report and slides

## Report (Typst) — `report/`

```bash
cd report
typst watch main.typ      # rebuilds main.pdf on every save; open it in a PDF viewer
typst compile main.typ    # one-off build
```

- `main.typ` — title-page info, abstract, list of chapters
- `template.typ` — all styling (fonts, margins, headings, numbering); edit once, applies everywhere
- `chapters/*.typ` — one file per chapter
- `figures/` — images; `references.bib` — bibliography

Syntax: `= Chapter`, `== Section`, `*bold*`, `_italic_`, `$x^2$` math, `@fig:car` to cross-reference
a `<fig:car>` label, `@muller2021openbot` to cite. Editor preview: the Tinymist extension.

## Slides (Slidev) — `slides/`

```bash
cd slides
npm install               # once
npm run dev               # opens the deck at http://localhost:3030, live reload
npm run export            # PDF backup (first run asks to install playwright-chromium)
```

- `slides.md` — the whole deck; slides are separated by `---`
- `public/` — videos, GIFs, images; reference them as `/name.mp4`
- `styles/` — fonts and CSS (fonts are bundled, so no internet needed)

While presenting: arrow keys / space to move, `P` or `/presenter` for speaker notes and timer,
`D` toggles dark mode, `O` shows an overview of all slides.

Demo videos: convert to H.264 MP4 before adding, e.g.
`ffmpeg -i clip.mov -c:v libx264 -crf 23 -an -movflags +faststart public/clip.mp4`.
Before the defense, test once with Wi-Fi off.
