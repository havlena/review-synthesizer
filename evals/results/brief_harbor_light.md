# Operator brief: Harbor Light Inn

Run `harbor-light-run1`.

## Verified rating

**3.5** verified, from 20 of 25 reviews (displayed rating 3.8). 5 review(s) are held out by confirmed flags; see part 4.

## 1. Themes

| Theme | Sentiment | Reviews | Evidence |
|---|---|---|---|
| Cleanliness and mold issues | mixed | 11 (2, 4, 5, 6, 7, 10, 12, 14, 16, 18, 20) | "Rooms were spotless"; "Room smelled musty and the carpet felt damp"; "The bathroom had mold in the corner" |
| Harbor views and location | positive | 9 (1, 3, 10, 12, 13, 16, 18, 19, 20) | "The view of the harbor from our room was incredible at sunrise"; "Loved the harbor location and the walk to dinner"; "Good location right by the water" |
| Front desk service varies | mixed | 8 (1, 4, 5, 9, 12, 14, 17, 19) | "Staff at the front desk were warm and helpful"; "Front desk went above and beyond to get us a late checkout"; "Nobody answered the desk phone"; "the front desk did nothing" |
| Inconsistent breakfast offering | mixed | 8 (2, 3, 7, 9, 11, 17, 19, 20) | "Best breakfast on the lakeshore"; "Breakfast was just packaged muffins and weak coffee"; "out of food by eight" |
| Comfortable beds | positive | 5 (3, 9, 12, 13, 15) | "Comfortable bed and a quiet room"; "Comfortable beds"; "The bed was a little firm" |
| Dated rooms, heater problems | negative | 4 (8, 11, 16, 17) | "the heater rattled all night"; "The heater did not work and it was a cold night"; "a touch dated" |
| Thin walls and noise | negative | 3 (4, 14, 19) | "Thin walls, we could hear the hallway"; "The walls are a little thin so you hear the hallway"; "Noisy neighbors until two in the morning" |
| Tight parking | negative | 2 (8, 15) | "Parking was a nightmare"; "Parking is tight but we managed" |
| Unreliable wifi | negative | 2 (6, 13) | "Wifi dropped a few times in the evening"; "Wifi was slow for work calls" |

## 2. Aspect scorecard

| Aspect | Score (1 to 5) | Reviews | Evidence |
|---|---|---|---|
| cleanliness | 3 | 11 (2, 4, 5, 6, 7, 10, 12, 14, 16, 18, 20) | Most guests call rooms clean or spotless, but three report musty damp carpet, bathroom mold, and a room not properly cleaned. |
| service | 3 | 8 (1, 4, 5, 9, 12, 14, 17, 19) | Staff are often described as warm and helpful, but others report slow check-in, an unanswered desk phone, and no action on a noise complaint. |
| breakfast | 3 | 8 (2, 3, 7, 9, 11, 17, 19, 20) | Cinnamon rolls, coffee and pastries are praised, but others report packaged muffins, few hot options, running out of food, and minimal choice. |
| room comfort | 3 | 12 (2, 3, 5, 6, 8, 9, 11, 12, 13, 15, 16, 17) | Comfortable beds and cozy rooms are praised, but guests also report dated rooms, a rattling or broken heater, and damp musty conditions. |
| value | 3 | 2 (2, 15) | One guest found it great value in season; another found it disappointing for the price. |
| location | 5 | 9 (1, 3, 10, 12, 13, 16, 18, 19, 20) | Guests praise the harbor setting, water views, and walkability to docks, dinner, and the marina, with no complaints. |
| noise | 3 | 7 (3, 4, 10, 13, 14, 19, 20) | Four guests found rooms quiet, while three reported thin walls, hallway sound, or noisy neighbors until 2 a.m. |
| parking | 2 | 2 (8, 15) | Parking is described as a nightmare and as tight, with only one guest saying they managed. |
| wifi | 1 | 2 (6, 13) | Both guests who mention wifi report problems: evening dropouts and slowness for work calls. |
| other | 3 | 2 (5, 6) | One guest praised the Northern Michigan charm; another booked a lakeview room and got a parking lot view. |

## 3. Actions

1. **Inspect rooms for damp and mold, then fix moisture sources** Impact High, effort Medium. Reviews report musty damp carpet and bathroom mold, and cleanliness is a mixed theme across 11 reviews. Reviews 2, 5.
2. **Service and test every room heater before the cold season** Impact High, effort Medium. Guests report a rattling heater and a heater that failed on a cold night, with staff unable to move them. Reviews 8, 17.
3. **Add a housekeeping checklist with a supervisor spot-check per room** Impact Medium, effort Low. One guest found trash still in the bin and another a moldy bathroom, while most others call rooms spotless, so consistency is the gap. Reviews 5, 14, 2.
4. **Upgrade wifi coverage and capacity for evening and work-call use** Impact Medium, effort Low. Both guests who mention wifi report evening dropouts or slowness for work calls, and it scores 1 of 5. Reviews 6, 13.
5. **Set a baseline breakfast menu with hot items and restock plan** Impact Medium, effort Medium. Praise for cinnamon rolls sits beside reports of packaged muffins, minimal choice, and food running out by eight. Reviews 2, 3, 11, 17, 7.

## 4. Authenticity findings

Each flag is a signal for review, not a verdict. Confirm keeps its reviews out of the verified rating and the synthesis; dismiss returns them.

- **Confirmed**, duplicate_phrasing, reviews 21, 22, 23, 24, 25, high (0.92). 5 reviews share near duplicate phrasing: peak Jaccard similarity 0.92 (mean 0.79) against a threshold of 0.55.
- **Confirmed**, rating_burst, reviews 21, 22, 23, 24, 25, high (0.83). 5 5-star reviews posted within 2 days (2026-06-03 to 2026-06-05); threshold is 3 within 4 days.
- **Confirmed**, embedding_similarity, reviews 21, 22, 23, 24, 25, high (1.00). 5 reviews share near duplicate meaning: peak cosine similarity 1.00 (mean 0.99) against a threshold of 0.85; 0 of 10 pairs in the group were not caught by word overlap.
- **Confirmed**, deceptive_language, reviews 21, low (0.82). Language resembles reviews written without a stay: p = 0.82 (threshold 0.80). Cues toward deceptive: 'amazing', 'hotel in', 'staff and'; toward truthful: 'michigan', 'in', 'staff'.
- **Confirmed**, deceptive_language, reviews 23, low (0.82). Language resembles reviews written without a stay: p = 0.82 (threshold 0.80). Cues toward deceptive: 'amazing', 'hotel in', 'staff and'; toward truthful: 'michigan', 'by', 'in'.

## Methods

- Signals: duplicate_phrasing, rating_burst, embedding_similarity, deceptive_language, claim_duplicate.
- duplicate_phrasing: Jaccard word overlap at or above 0.55.
- rating_burst: 3 or more same rating reviews within 4 days.
- embedding_similarity: cosine similarity at or above 0.85.
- deceptive_language: classifier probability at or above 0.80.
- claim_duplicate: at least 2 shared specific claims, then a model judgment per pair (strict prompt).
- Models: main claude-sonnet-5-5, small claude-haiku-5-5, embeddings sentence-transformers/paraphrase-mpnet-base-v2.
- Flag decisions: automatic: every flag confirmed without review (evaluation run).
- Run id: harbor-light-run1.
