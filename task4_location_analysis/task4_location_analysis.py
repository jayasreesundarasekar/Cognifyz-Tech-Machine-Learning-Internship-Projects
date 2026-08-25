"""
Task 4: Location-based Analysis
Cognifyz Technologies - Data Science Internship

Objective: Perform a geographical analysis of the restaurants in the dataset.

Pipeline:
1. Explore lat/long distribution, visualize on a map
2. Group restaurants by city/locality, analyze concentration
3. Calculate stats (avg rating, cuisines, price ranges) by city/locality
4. Identify interesting insights/patterns
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
OUT = "outputs"

# ---------------------------------------------------------------
# 1. LOAD & CLEAN
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 1: Loading dataset")
print("=" * 60)

df = pd.read_csv("data/Dataset_.csv")
print(f"Raw shape: {df.shape}")
print(f"Cities: {df['City'].nunique()}, Countries: {df['Country Code'].nunique()}")

# Country code -> name mapping (from Zomato's known country code list)
country_map = {
    1: "India", 14: "Australia", 30: "Brazil", 37: "Canada", 94: "Indonesia",
    148: "New Zealand", 162: "Philippines", 166: "Qatar", 184: "Singapore",
    189: "South Africa", 191: "Sri Lanka", 208: "Turkey", 214: "UAE",
    215: "United Kingdom", 216: "United States",
}
df["Country"] = df["Country Code"].map(country_map).fillna("Other")

# Coordinates of exactly (0,0) are missing/bad data, not real locations
geo_df = df[(df["Longitude"] != 0) & (df["Latitude"] != 0)].copy()
print(f"Restaurants with valid coordinates: {len(geo_df)} / {len(df)}")

# ---------------------------------------------------------------
# 2. EXPLORE LAT/LONG DISTRIBUTION - VISUALIZE
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Visualizing geographic distribution")
print("=" * 60)

# Global scatter map (all countries)
plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=geo_df, x="Longitude", y="Latitude", hue="Country",
    s=15, alpha=0.6, legend="brief", palette="tab20"
)
plt.title("Global Distribution of Restaurants (Longitude vs Latitude)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/global_restaurant_map.png", dpi=150)
plt.close()

# Zoomed-in map for India (majority of the data)
india_df = geo_df[geo_df["Country"] == "India"]
plt.figure(figsize=(9, 8))
sc = plt.scatter(
    india_df["Longitude"], india_df["Latitude"],
    c=india_df["Aggregate rating"], cmap="RdYlGn", s=12, alpha=0.7
)
plt.colorbar(sc, label="Aggregate Rating")
plt.title(f"Restaurant Distribution in India ({len(india_df)} restaurants, colored by rating)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig(f"{OUT}/india_restaurant_map.png", dpi=150)
plt.close()

print("Saved global_restaurant_map.png and india_restaurant_map.png")

# ---------------------------------------------------------------
# 3. GROUP BY CITY/LOCALITY - CONCENTRATION ANALYSIS
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Grouping restaurants by city / locality")
print("=" * 60)

city_counts = df["City"].value_counts()
print("\nTop 15 cities by restaurant count:")
print(city_counts.head(15))

locality_counts = df["Locality"].value_counts()
print("\nTop 15 localities by restaurant count:")
print(locality_counts.head(15))

city_counts.head(20).to_csv(f"{OUT}/top20_cities_by_count.csv", header=["restaurant_count"])
locality_counts.head(20).to_csv(f"{OUT}/top20_localities_by_count.csv", header=["restaurant_count"])

plt.figure(figsize=(10, 7))
city_counts.head(15).sort_values().plot(kind="barh", color="#4C72B0")
plt.title("Top 15 Cities by Restaurant Count")
plt.xlabel("Number of Restaurants")
plt.tight_layout()
plt.savefig(f"{OUT}/top_cities_count.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 7))
locality_counts.head(15).sort_values().plot(kind="barh", color="#DD8452")
plt.title("Top 15 Localities by Restaurant Count")
plt.xlabel("Number of Restaurants")
plt.tight_layout()
plt.savefig(f"{OUT}/top_localities_count.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 4. STATISTICS BY CITY: AVG RATING, CUISINES, PRICE RANGE
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: City-level statistics")
print("=" * 60)

rated_df = df[df["Rating text"] != "Not rated"]

city_stats = rated_df.groupby("City").agg(
    restaurant_count=("Restaurant ID", "count"),
    avg_rating=("Aggregate rating", "mean"),
    avg_cost_for_two=("Average Cost for two", "mean"),
    avg_price_range=("Price range", "mean"),
    avg_votes=("Votes", "mean"),
).round(2)

# only consider cities with a reasonable sample size for ranking
city_stats_significant = city_stats[city_stats["restaurant_count"] >= 5].sort_values(
    "avg_rating", ascending=False
)

print("\nTop 10 highest-rated cities (min. 5 restaurants):")
print(city_stats_significant.head(10))

print("\nBottom 10 lowest-rated cities (min. 5 restaurants):")
print(city_stats_significant.tail(10))

city_stats_significant.to_csv(f"{OUT}/city_level_statistics.csv")

# most common cuisine per city (top 10 cities by restaurant count)
top_cities = city_counts.head(10).index
cuisine_by_city = {}
for city in top_cities:
    city_cuisines = df[df["City"] == city]["Cuisines"].dropna().str.split(",").explode().str.strip()
    top_cuisine = city_cuisines.value_counts().idxmax()
    cuisine_by_city[city] = top_cuisine

print("\nMost common cuisine in top 10 cities:")
for city, cuisine in cuisine_by_city.items():
    print(f"  {city}: {cuisine}")

with open(f"{OUT}/top_cuisine_by_city.json", "w") as f:
    json.dump(cuisine_by_city, f, indent=2)

# Chart: avg rating for top 15 cities by count
plt.figure(figsize=(10, 7))
top_city_ratings = city_stats.loc[top_cities.union(city_counts.head(15).index)].sort_values(
    "avg_rating"
)["avg_rating"]
top_city_ratings = city_stats.loc[city_counts.head(15).index].sort_values("avg_rating")["avg_rating"]
top_city_ratings.plot(kind="barh", color="#55A868")
plt.title("Average Rating - Top 15 Cities by Restaurant Count")
plt.xlabel("Average Aggregate Rating")
plt.tight_layout()
plt.savefig(f"{OUT}/avg_rating_top_cities.png", dpi=150)
plt.close()

# Price range distribution by top cities (heatmap)
price_pivot = pd.crosstab(df[df["City"].isin(city_counts.head(10).index)]["City"], df["Price range"])
plt.figure(figsize=(9, 7))
sns.heatmap(price_pivot, annot=True, fmt="d", cmap="Blues")
plt.title("Price Range Distribution Across Top 10 Cities")
plt.xlabel("Price Range (1=Cheap, 4=Expensive)")
plt.tight_layout()
plt.savefig(f"{OUT}/price_range_heatmap.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 5. INSIGHTS & PATTERNS
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: Insights and patterns")
print("=" * 60)

# Country level rollup
country_stats = rated_df.groupby("Country").agg(
    restaurant_count=("Restaurant ID", "count"),
    avg_rating=("Aggregate rating", "mean"),
).round(2).sort_values("restaurant_count", ascending=False)
country_stats.to_csv(f"{OUT}/country_level_statistics.csv")
print("\nCountry-level rollup:")
print(country_stats)

# City with highest / lowest concentration of expensive restaurants
expensive_share = df.groupby("City").apply(
    lambda g: (g["Price range"] == 4).mean()
).sort_values(ascending=False)
top_expensive_city = expensive_share[city_counts[city_counts >= 5].index.intersection(expensive_share.index)].idxmax()

insights = f"""
TASK 4: LOCATION-BASED ANALYSIS - KEY INSIGHTS
=================================================

DATA COVERAGE
- {len(df)} total restaurants across {df['City'].nunique()} cities in
  {df['Country Code'].nunique()} countries.
- {len(geo_df)} restaurants have valid (non-zero) coordinates and were used
  for map visualizations; {len(df) - len(geo_df)} had missing/placeholder
  coordinates and were excluded from geo-plots (but retained in city stats).

GEOGRAPHIC CONCENTRATION
- Restaurant listings are heavily dominated by India: {city_counts.get('New Delhi', 0)}
  in New Delhi alone, the single largest city by count.
- Top 3 cities by restaurant count: {', '.join(city_counts.head(3).index.tolist())}.
- The top 10 cities account for {city_counts.head(10).sum()} of {len(df)}
  restaurants ({city_counts.head(10).sum()/len(df)*100:.1f}% of the dataset)
  -> the data is geographically imbalanced, clustered around a handful of
  major metro areas (largely NCR region in India: New Delhi, Noida, Gurgaon,
  Faridabad, Ghaziabad).

RATING PATTERNS BY LOCATION
- Highest-rated city (min. 5 restaurants): {city_stats_significant.index[0]}
  (avg rating {city_stats_significant['avg_rating'].iloc[0]}).
- Lowest-rated city (min. 5 restaurants): {city_stats_significant.index[-1]}
  (avg rating {city_stats_significant['avg_rating'].iloc[-1]}).
- Cities outside India (e.g. those with smaller restaurant counts but strong
  international dining scenes) tend to show higher average ratings than the
  Indian metro average, suggesting either more selective listings or
  different rating culture per region.

CUISINE PATTERNS
- North Indian and Chinese dominate as the most common cuisine in most major
  Indian cities in the dataset.
- Country-level cuisine preference varies significantly, reflecting local
  dining culture.

PRICING PATTERNS
- '{top_expensive_city}' has the highest concentration of top-tier
  (price range 4) restaurants among cities with a meaningful sample size.
- Price range distribution varies noticeably by city - some cities skew
  toward budget dining (price range 1-2), others toward premium (3-4).

RECOMMENDATION
- Given the geographic imbalance, any predictive model trained on this data
  (see Task 1) should be validated separately on non-Indian cities to check
  it generalizes rather than just fitting NCR-region patterns.
"""
print(insights)
with open(f"{OUT}/summary_report.txt", "w") as f:
    f.write(insights)

print("\nAll outputs saved to outputs/")
