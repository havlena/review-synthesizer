# Data sources and terms

Nothing in this repository is real business or reviewer data. Every data set below is
either synthetic, public under stated terms, or supplied by an operator about their own
property.

## 1. Harbor Light Inn (synthetic, in the repository)

`tests/fixtures/harbor_light_inn.json`. Twenty five reviews written for the capstone for a
fictional inn: twenty genuine in intent and five planted to simulate a coordinated burst.
Written by the project author; no license constraints. It is the golden set for the rule
based detector and is far too small to measure anything beyond "the demo reproduces".

## 2. Deceptive Opinion Spam Corpus v1.4 (Ott et al.), fetched on demand

1,600 reviews of 20 Chicago hotels: 400 truthful positive (TripAdvisor), 400 deceptive
positive (Mechanical Turk), 400 truthful negative (Expedia, Hotels.com, Orbitz, Priceline,
TripAdvisor, Yelp), 400 deceptive negative (Mechanical Turk). Published at
https://myleott.com/op-spam.html.

Terms, as stated by the authors: Creative Commons Attribution NonCommercial ShareAlike 3.0,
with a request to cite the papers:

- M. Ott, Y. Choi, C. Cardie, J. T. Hancock (2011). Finding Deceptive Opinion Spam by Any
  Stretch of the Imagination. ACL.
- M. Ott, C. Cardie, J. T. Hancock (2013). Negative Deceptive Opinion Spam. NAACL.

What that means here:

- The corpus is not committed. `scripts/fetch_ott_corpus.py` downloads it to `data/raw/`
  (git ignored) and writes `data/interim/ott_corpus.csv`.
- It is used to evaluate the detector and to train the demonstration classifier. Reported
  precision and recall figures come from it and are labelled as such.
- A classifier trained on it is for research and demonstration. A commercial deployment
  needs training data under different terms; the architecture keeps the classifier a
  swappable component for that reason.
- Known limits of the corpus as a proxy for today's fake reviews: the deceptive reviews
  were written by paid crowd workers in 2011 to 2013, for one city and one segment (large
  Chicago hotels), in English only. A model trained on it learns what "written without
  having stayed there" looked like then, not what coordinated manipulation looks like now.

## 3. Public hotel review sets for theme and aspect work (to be selected)

The capstone used a public TripAdvisor hotel review set for development of the synthesis
layer. The v2 choice is made when that layer is built, with its terms recorded here.

## 4. Operator supplied reviews (the production input)

What a real property feeds the system comes from channels the operator already controls:

- CSV or spreadsheet exports from their own booking and review platforms.
- The Google Business Profile API for their own listing, which is official and owner
  authorised.

Scraping review sites is out of scope: it breaks those sites' terms and the project's own
governance design (see the capstone Module 4 and Module 6 work).
