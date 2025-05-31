#import "@preview/typslides:1.2.5": *

// #let blue = rgb("#048BA8")
// #let purple = rgb("#662E9B")
// #let yellow = rgb("#DCAB6B")
// #let cream = rgb("#F9F5FF")
#let red = rgb("#AD160B")
#let green = rgb("#03B5AA")
// #let purple = rgb("#DCAB6B")
#let blplot = rgb("2A67A8")
#let orplot = rgb("CE6A07")


#show: typslides.with(
  ratio: "16-9",
  theme: "dusky",
)

#show raw.where(block: false): box.with(
  fill: luma(230),
  inset: (x: 5pt, y: 0pt),
  outset: (y: 6pt),
  radius: 4pt,
)

#show raw.where(block: true): block.with(
  fill: luma(240),
  inset: 10pt,
  radius: 4pt,
)

#front-slide(
  title: "Michelle Notes",
  subtitle: "",
  authors: "Michelle Gelman",
  info: "April 23, 2025",
)

#slide(title:"Tree")[
  #framed(title: "Tree Defintion")[
A tree $𝕋$ is then (non-uniquely) represented as a collection of $T$ paths $𝒫(𝕋) = \{p_1, …, p_T\}$

  ]
]

#slide(title:"Path Components")[
  #cols(columns: (2fr, 2fr), gutter: 2em)[
      #stress("Tree")
][

  ]
]





