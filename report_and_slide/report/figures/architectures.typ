// Block diagrams of the two driving models (Chapter 7), drawn with fletcher.
// Colours: image path blue, temporal (LSTM) green, command orange, head grey.
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#let c-img = rgb("#dbe8f6")
#let c-time = rgb("#dcefdc")
#let c-cmd = rgb("#fbe3c8")
#let c-head = rgb("#ececec")

#let blk(pos, body, fill, name: none, w: auto) = node(
  pos, align(center, body), fill: fill, stroke: 0.6pt + luma(90), corner-radius: 2pt,
  inset: 5pt, width: w, name: name, shape: fletcher.shapes.rect,
)
#let dot(pos) = node(pos, none, shape: circle, radius: 1.3pt, fill: black, stroke: none)
#let arr = "-|>"

// CIL-Mobile-Fast: one frame, the command MLP joins the image features at the first
// FC 64, and the raw command is concatenated again before each later layer.
#let cil-arch = {
  set text(size: 8pt)
  diagram(
    spacing: (7mm, 6mm),
    blk((0, 0), [Image\ $96 times 256 times 3$], c-img),
    blk((1, 0), [5 conv layers\ 32, 32, 64, 64, 128\ BN, dropout], c-img),
    blk((2, 0), [FC 512\ FC 512], c-img),
    blk((3, 0), [FC\ 64], c-head),
    blk((4, 0), [FC\ 64], c-head),
    blk((5, 0), [FC 2\ linear], c-head),
    node((6, 0), [$hat(s), hat(t)$]),
    blk((0, 1), [Command\ $c$], c-cmd),
    blk((2, 1), [MLP\ 16, 16], c-cmd),

    edge((0, 0), (1, 0), arr),
    edge((1, 0), (2, 0), arr),
    edge((2, 0), (3, 0), arr),
    edge((3, 0), (4, 0), arr),
    edge((4, 0), (5, 0), arr),
    edge((5, 0), (6, 0), arr),
    // command branch into the first FC 64
    edge((0, 1), (2, 1), arr),
    edge((2, 1), (3, 1), (3, 0), arr),
    // raw command into each later layer
    edge((0, 1), (0, 1.75), (5, 1.75), (5, 0), arr),
    dot((4, 1.75)),
    edge((4, 1.75), (4, 0), arr),
    node((2.5, 1.75), text(size: 7pt)[raw $c$], fill: white, inset: 1.5pt),
  )
}

// Recurrent model: a shared CNN on each of the three frames, two LSTM layers over time,
// then the same command head as CIL-Mobile-Fast.
#let rnn-arch = {
  set text(size: 8pt)
  diagram(
    spacing: (5.5mm, 6mm),
    blk((0, 0), [$bold(I)_(t-6)$], c-img),
    blk((0, 1), [$bold(I)_(t-3)$], c-img),
    blk((0, 2), [$bold(I)_t$], c-img),
    ..range(3).map(r => blk((1, r), [4 conv, pool\ FC 100], c-img)),
    ..range(3).map(r => blk((2, r), [LSTM\ 128], c-time)),
    ..range(3).map(r => blk((3, r), [LSTM\ 128], c-time)),
    node(enclose: ((1, 0), (1, 2)), stroke: (dash: "dashed", paint: luma(120), thickness: 0.5pt),
      inset: 3pt, corner-radius: 3pt),
    node((1, -0.75), text(size: 7pt)[shared weights]),
    node((2.5, -0.75), text(size: 7pt)[time $arrow.b$]),
    blk((4, 2), [FC\ 128], c-head),
    blk((5, 2), [FC\ 64], c-head),
    blk((6, 2), [FC\ 10], c-head),
    blk((7, 2), [FC 2\ linear], c-head),
    node((8, 2), [$hat(s), hat(t)$]),
    blk((0, 3.2), [Command\ $c$], c-cmd),
    blk((3, 3.2), [MLP\ 16, 16], c-cmd),

    ..range(3).map(r => edge((0, r), (1, r), arr)),
    ..range(3).map(r => edge((1, r), (2, r), arr)),
    ..range(3).map(r => edge((2, r), (3, r), arr)),
    ..range(2).map(r => edge((2, r), (2, r + 1), arr)),
    ..range(2).map(r => edge((3, r), (3, r + 1), arr)),
    edge((3, 2), (4, 2), arr),
    edge((4, 2), (5, 2), arr),
    edge((5, 2), (6, 2), arr),
    edge((6, 2), (7, 2), arr),
    edge((7, 2), (8, 2), arr),
    // command branch into FC 128
    edge((0, 3.2), (3, 3.2), arr),
    edge((3, 3.2), (4, 3.2), (4, 2), arr),
    // raw command into each later layer
    edge((0, 3.2), (0, 4), (7, 4), (7, 2), arr),
    dot((5, 4)),
    dot((6, 4)),
    edge((5, 4), (5, 2), arr),
    edge((6, 4), (6, 2), arr),
    node((2, 4), text(size: 7pt)[raw $c$], fill: white, inset: 1.5pt),
  )
}
