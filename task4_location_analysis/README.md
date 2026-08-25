# Task 4: Location-based Analysis
### Cognifyz Technologies — Data Science Internship

## Objective
Perform a geographical analysis of the restaurants in the dataset.

## Contents
```
task4_location_analysis/
├── data/
│   └── Dataset_.csv                     # Raw Zomato restaurant dataset
├── outputs/
│   ├── global_restaurant_map.png         # World scatter map, colored by country
│   ├── india_restaurant_map.png          # India scatter map, colored by rating
│   ├── top_cities_count.png              # Top 15 cities by restaurant count
│   ├── top_localities_count.png          # Top 15 localities by restaurant count
│   ├── avg_rating_top_cities.png         # Avg rating for top 15 cities
│   ├── price_range_heatmap.png           # Price range distribution, top 10 cities
│   ├── top20_cities_by_count.csv
│   ├── top20_localities_by_count.csv
│   ├── city_level_statistics.csv         # Full city-level stats table
│   ├── country_level_statistics.csv      # Country-level rollup
│   ├── top_cuisine_by_city.json          # Most common cuisine per top city
│   └── summary_report.txt                # Full written insights report
├── task4_location_analysis.py            # End-to-end analysis script
└── README.md
```

## How to run
```bash
pip install pandas numpy matplotlib seaborn
python3 task4_location_analysis.py
```

## Pipeline steps

1. **Explore lat/long distribution & visualize**
   - Identified 499 restaurants with (0,0) placeholder coordinates (missing data) and excluded them from map plots
   - 9,052 restaurants plotted on a global map, colored by country
   - Zoomed India map, colored by aggregate rating

2. **Group by city / locality — concentration analysis**
   - Grouped all 9,551 restaurants by `City` and `Locality`
   - Ranked top cities/localities by restaurant count

3. **City-level statistics**
   - Computed avg rating, avg cost for two, avg price range, and avg votes per city (min. 5 restaurants for statistical significance)
   - Computed most common cuisine per top-10 city
   - Built a price-range heatmap across top cities

4. **Insights identified**

| Finding | Detail |
|---|---|
| Geographic imbalance | Top 10 cities = **84.3%** of all listings; dataset is heavily NCR-India-centric |
| Largest city | **New Delhi** — 5,473 restaurants |
| Highest-rated city (≥5 restaurants) | **London** — avg rating 4.54 |
| Lowest-rated city (≥5 restaurants) | **Ghaziabad** — avg rating 3.10 |
| Country rollup | India avg rating **3.35**; most non-Indian countries average **4.0+** |
| Dominant cuisine | **North Indian** in most major Indian cities; Chinese in Bhubaneshwar/Guwahati |
| Priciest city | **Johannesburg** — highest share of price-range-4 restaurants |

## Key Insights
- The dataset skews heavily toward the Delhi NCR region (New Delhi, Noida, Gurgaon, Faridabad, Ghaziabad), so any model trained on it (e.g. Task 1's rating predictor) risks overfitting to Indian metro dining patterns and should be validated separately on non-Indian cities.
- Indian cities in the dataset average noticeably lower ratings (~3.3) than international cities (~4.0+), which could reflect either listing/rating-culture differences or genuinely different review distributions on the platform per region.
- Cuisine preference is strongly regional — North Indian dominates NCR, while East/Northeast Indian cities lean Chinese, reflecting real local dining culture.
- Price range varies meaningfully by city, useful for market-positioning or expansion-strategy questions (e.g. which cities are underserved by premium dining).
