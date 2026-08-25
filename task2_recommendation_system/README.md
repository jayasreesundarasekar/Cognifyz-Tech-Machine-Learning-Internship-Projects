# Task 2: Restaurant Recommendation System
### Cognifyz Technologies — Data Science Internship

## Objective
Create a restaurant recommendation system based on user preferences.

## Contents
```
task2_recommendation_system/
├── data/
│   └── Dataset_.csv                          # Raw Zomato restaurant dataset
├── outputs/
│   ├── test1_italian_pizza_delhi.csv          # Sample recommendation output
│   ├── test2_chinese_japanese_premium.csv     # Sample recommendation output
│   ├── test3_similar_restaurants.csv          # Item-based recommendation output
│   ├── test_results.txt                       # Full text of all test runs
│   ├── recommendation_metrics.json             # Quality metrics
│   └── summary_report.txt                      # Written summary
├── task2_recommendation_system.py              # End-to-end pipeline script
└── README.md
```

## How to run
```bash
pip install pandas numpy scikit-learn
python3 task2_recommendation_system.py
```

## Pipeline steps

1. **Preprocessing**
   - Dropped 2,148 "Not rated" restaurants
   - Filled missing `Cuisines` values with the mode
   - Removed 1,249 duplicate restaurant entries (same name + city)
   - Result: 6,154 clean restaurant records

2. **Recommendation criteria**
   Content-based filtering over a combined feature space:
   - **Cuisine** — multi-hot encoded (143 unique cuisine tokens, via `CountVectorizer`)
   - **Price range** (1–4), normalized
   - **Average cost for two**, normalized
   - **Aggregate rating**, normalized (acts as a quality signal, biasing recommendations toward better restaurants)
   - **Table booking** / **Online delivery** availability

3. **Implementation** — two recommendation modes:
   - **Preference-based** (`recommend_from_preferences`): user supplies preferred cuisines, price range, city, and minimum rating. The system builds a synthetic "ideal restaurant" vector and ranks all restaurants by cosine similarity to it.
   - **Item-based** (`recommend_similar_restaurants`): given a restaurant the user already likes, finds the most similar restaurants by cosine similarity in the same feature space.

4. **Testing with sample preferences**
   - Test 1: Italian/Pizza lover, mid-range price, New Delhi → top match "Play Pizza" (similarity 0.80, rating 3.8)
   - Test 2: Chinese/Japanese lover, premium price, any city → top match "3 Wise Monkeys" (similarity 0.86, rating 4.2)
   - Test 3: "restaurants similar to Le Petit Souffle" → surfaced other Japanese/French fusion spots

## Evaluation of recommendation quality

| Metric | Value |
|---|---|
| Dataset average rating | 3.48 |
| Test 1 recommended avg rating | 3.92 |
| Test 2 recommended avg rating | 4.40 |
| Test 1 avg similarity | 0.718 |
| Test 2 avg similarity | 0.831 |

Recommended restaurants consistently rate **above the dataset average**, confirming the system successfully balances user preference matching with quality (rating) bias.

## Key Insights
- Cuisine match is the dominant driver of similarity since it has the most feature dimensions (143 tokens) — this correctly prioritizes matching what the user actually wants to eat.
- Including normalized rating in the feature vector nudges recommendations toward better-reviewed restaurants without ignoring user cuisine/price preferences.
- The item-based mode is useful for a "customers who liked X also liked..." style feature on a restaurant detail page.
