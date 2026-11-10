// HCMUT thesis layout (Faculty of Computer Science and Engineering format): title page,
// front matter, list pages and chapter styling. Every chapter inherits these rules.

#let rule() = line(length: 100%, stroke: 0.5pt)

// Centred title of a front-matter text page (Commitment, Abstract, ...).
#let fm-heading(title) = {
  v(2.2cm)
  align(center, text(size: 13pt, upper(title)))
  v(0.8cm)
}

// Right-aligned title between two rules (Table of contents, list pages, unnumbered chapters).
#let rule-heading(title) = {
  v(2cm)
  rule()
  v(0.2em)
  align(right, text(size: 16pt, weight: "bold", title))
  v(0.2em)
  rule()
  v(1.4cm)
}

// Front-matter list page; the metadata lets the header repeat the title on later pages.
#let list-heading(title) = {
  [#metadata(title) <fm-title>]
  rule-heading(title)
}

// Front matter: empty rule on text pages, list title on continuation pages of a list.
#let fm-header = context {
  let pg = here().page()
  let marks = query(<fm-title>)
  if marks.any(m => m.location().page() == pg) { return }
  let before = marks.filter(m => m.location().page() < pg)
  set text(size: 10pt, weight: "bold")
  if before.len() > 0 {
    align(right, before.last().value)
    v(-0.6em)
  }
  rule()
}

// Main matter: no header on a chapter's first page, otherwise the current section.
#let main-header = context {
  let pg = here().page()
  let chapters = query(heading.where(level: 1))
  if chapters.any(c => c.location().page() == pg) { return }
  let before = chapters.filter(c => c.location().page() < pg)
  if before.len() == 0 { return }
  let ch = before.last()
  let sections = query(heading.where(level: 2)).filter(s => {
    let p = s.location().page()
    p >= ch.location().page() and p <= pg
  })
  let label = if sections.len() > 0 {
    let s = sections.last()
    [#numbering("1.1", ..counter(heading).at(s.location())). #h(0.2em) #s.body]
  } else {
    upper(ch.body)
  }
  set text(size: 10pt, weight: "bold")
  align(right, label)
  v(-0.6em)
  rule()
}

#let thesis(
  title: "",
  thesis-type: "GRADUATION THESIS",
  kind: "THESIS",
  major: "",
  supervisors: (),
  students: (),            // ((name, id), ...)
  place-date: "",
  logo: "figures/hcmut_logo.png",
  commitment: [],
  acknowledgement: [],
  abstract: [],
  keywords: (),
  abbreviations: (),       // (("AI", "Artificial Intelligence"), ...)
  body,
) = {
  set document(title: title, author: students.map(s => s.at(0)))
  set page(
    paper: "a4",
    margin: (left: 3cm, right: 2cm, top: 2.5cm, bottom: 2.5cm),
    header-ascent: 0.6cm,
  )
  set text(font: "New Computer Modern", size: 12pt, lang: "en")
  set par(justify: true, leading: 0.75em, spacing: 0.75em, first-line-indent: (amount: 1.2em, all: true))
  show raw: set text(font: "DejaVu Sans Mono", size: 9.5pt)

  // Code blocks: light grey box.
  show raw.where(block: true): it => block(
    fill: luma(245), inset: 10pt, radius: 4pt, width: 100%, align(left, it),
  )

  // Tables: shaded header row, thin horizontal rules only.
  set table(
    stroke: (_, y) => (bottom: 0.4pt + luma(190)),
    fill: (_, y) => if y == 0 { luma(232) },
    inset: (x: 8pt, y: 5pt),
  )

  // ---- Headings ----
  set heading(numbering: "1.1")
  show heading.where(level: 1): set heading(supplement: [Chapter])
  show heading.where(level: 1): it => {
    // Figures, tables, listings and equations restart per chapter: Figure 3.2, (4.1).
    counter(figure.where(kind: image)).update(0)
    counter(figure.where(kind: table)).update(0)
    counter(figure.where(kind: raw)).update(0)
    counter(math.equation).update(0)
    pagebreak(weak: true)
    set par(first-line-indent: 0pt)
    if it.numbering == none {
      rule-heading(upper(it.body))
    } else {
      v(1.6cm)
      text(size: 11pt, weight: "bold", smallcaps[Chapter])
      h(0.4em)
      text(size: 22pt, weight: "bold", counter(heading).display("1"))
      v(0.3em)
      rule()
      v(0.2em)
      align(right, text(size: 15pt, weight: "bold", upper(it.body)))
      v(0.2em)
      rule()
      v(1.2cm)
    }
  }
  show heading.where(level: 2): it => block(above: 1.8em, below: 1em, text(size: 16pt, weight: "bold")[
    #if it.numbering != none [#counter(heading).display() #h(0.8em)]#it.body
  ])
  show heading.where(level: 3): it => block(above: 1.5em, below: 0.9em, text(size: 13pt, weight: "bold")[
    #if it.numbering != none [#counter(heading).display() #h(0.8em)]#it.body
  ])

  set figure(numbering: n => numbering("1.1", counter(heading).get().first(), n))
  set math.equation(numbering: n => numbering("(1.1)", counter(heading).get().first(), n))
  show figure.where(kind: table): set figure.caption(position: top)
  show figure.caption: set text(size: 10.5pt)

  // ---- Table of contents and lists ----
  show outline.entry: it => {
    let el = it.element
    if el.func() == figure {
      // "3.1  Caption ........ 42", like the LaTeX list of figures.
      let loc = el.location()
      let num = [#counter(heading).at(loc).first().#el.counter.at(loc).first()]
      link(loc, grid(
        columns: (2.4em, 1fr),
        num, [#el.caption.body #box(width: 1fr, repeat[.#h(4pt)]) #it.page()],
      ))
    } else if el.func() == heading and el.level == 1 {
      // Chapters: bold, no dot leaders.
      v(1.1em, weak: true)
      let prefix = if el.numbering != none [Chapter #counter(heading).at(el.location()).first(). ]
      link(el.location(), strong[#prefix#upper(el.body) #h(1fr) #it.page()])
    } else {
      it
    }
  }

  // ---- Title page ----
  page(margin: (left: 2.5cm, right: 1.5cm, top: 2cm, bottom: 1.5cm), header: none, numbering: none)[
    #rect(width: 100%, height: 100%, stroke: 1.8pt, inset: 2.5pt, rect(
      width: 100%, height: 100%, stroke: 0.6pt, inset: (x: 1cm, y: 1.2cm),
    )[
      #set par(first-line-indent: 0pt, justify: false)
      #set align(center)
      #text(weight: "bold", size: 11.5pt)[
        VIET NAM NATIONAL UNIVERSITY HO CHI MINH CITY \
        HO CHI MINH CITY UNIVERSITY OF TECHNOLOGY \
        FACULTY OF COMPUTER SCIENCE AND ENGINEERING
      ]
      #v(0.8cm)
      #image(logo, width: 2.6cm)
      #v(0.8cm)
      #text(size: 14pt, thesis-type)
      #v(0.9cm)
      #text(size: 11pt, weight: "bold", kind)
      #v(1.2cm)
      #text(size: 15pt, upper(title))
      #v(1fr)
      Major: #major
      #v(1.2cm)
      #grid(
        columns: (auto, auto), column-gutter: 1em, row-gutter: 0.65em, align: (left, left),
        text(weight: "bold")[SUPERVISOR(s):], text(weight: "bold", supervisors.join(linebreak())),
        grid.cell(colspan: 2, align: center)[—o0o—],
        text(weight: "bold")[STUDENT(s):], text(weight: "bold", students.map(s => s.at(0)).join(linebreak())),
        text(weight: "bold")[STUDENT'S ID:], text(weight: "bold", students.map(s => s.at(1)).join(linebreak())),
      )
      #v(1.6cm)
      #upper(place-date)
    ]
    )
  ]

  // ---- Front matter (roman page numbers) ----
  set page(numbering: "i", header: fm-header)
  counter(page).update(1)

  fm-heading("Verification of instructor")
  v(1fr)
  align(right)[Signature confirmation]
  pagebreak()

  fm-heading("Commitment")
  {
    set par(spacing: 1.5em)
    commitment
  }
  v(0.8em)
  align(right)[Sincerely,]
  v(0.5em)
  align(right, strong(students.map(s => s.at(0)).join(", ")))
  pagebreak()

  fm-heading("Acknowledgement")
  {
    set par(spacing: 1.5em)
    acknowledgement
  }
  pagebreak()

  fm-heading("Abstract")
  {
    set par(spacing: 1.5em)
    abstract
  }
  if keywords.len() > 0 {
    v(1em)
    par(first-line-indent: 0pt)[#underline[Keywords:] #keywords.join(", ")]
  }
  pagebreak()

  list-heading("Table of contents")
  outline(title: none, indent: auto, depth: 3)
  pagebreak()
  list-heading("List of figures")
  outline(title: none, target: figure.where(kind: image))
  pagebreak()
  list-heading("Tables")
  outline(title: none, target: figure.where(kind: table))

  // ---- Main matter (arabic page numbers, starting at the abbreviations page) ----
  set page(numbering: "1", header: main-header)
  counter(page).update(1)

  if abbreviations.len() > 0 {
    heading(numbering: none)[List of abbreviations]
    let items = abbreviations.sorted(key: a => a.at(0))
    let half = calc.ceil(items.len() / 2)
    let column(part) = grid(
      columns: (4.6em, 1fr), row-gutter: 1em,
      ..part.map(((a, m)) => (a, m)).flatten(),
    )
    set text(size: 10pt)
    set par(justify: false, first-line-indent: 0pt)
    grid(columns: (1fr, 1fr), column-gutter: 1.5em, column(items.slice(0, half)), column(items.slice(half)))
  }

  body
}
