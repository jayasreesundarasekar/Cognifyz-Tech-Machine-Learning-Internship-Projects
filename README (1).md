# Cognify Machine Learning Tasks

This repository contains selected Machine Learning tasks completed as part of the **Cognify internship program**. The projects use a restaurant dataset to demonstrate regression, recommendation systems, and geographical data analysis.

## Tasks Covered

| Task | Project | Main Focus |
|---|---|---|
| **Task 1** | Predict Restaurant Ratings | Regression |
| **Task 2** | Restaurant Recommendation System | Content-Based Filtering |
| **Task 4** | Location-Based Analysis | Geographical & Exploratory Data Analysis |

## Repository Structure

```text
Cognify-Machine-Learning-Tasks/
│
├── LICENSE
├── README.md
│
├── task1_predict_ratings/
│   └── ...
│
├── task2_recommendation_system/
│   └── ...
│
└── task4_location_analysis/
    └── ...
```

Each task is organized in its own directory with the relevant Python code, analysis, and supporting files.

---

# Task 1 — Predict Restaurant Ratings

## Objective

Build a machine learning model to predict the **aggregate rating of a restaurant** based on other available restaurant features.

## Workflow

1. Preprocess the dataset by:
   - Handling missing values
   - Cleaning the data
   - Encoding categorical variables
   - Separating features and target variables

2. Split the data into training and testing sets.

3. Select and train a suitable regression algorithm.

4. Evaluate the model using regression metrics such as:
   - Mean Squared Error (MSE)
   - R² Score

5. Interpret the model results and analyze the features that have the greatest influence on restaurant ratings.

## Machine Learning Type

**Supervised Learning — Regression**

The target variable is the restaurant's aggregate rating, a continuous numerical value.

---

# Task 2 — Restaurant Recommendation System

## Objective

Create a restaurant recommendation system that recommends restaurants based on **user preferences**.

The system considers restaurant characteristics such as cuisine and price range to find restaurants that match the user's requirements.

## Workflow

1. Preprocess the dataset by:
   - Handling missing values
   - Cleaning relevant restaurant attributes
   - Encoding categorical variables where required

2. Define recommendation criteria such as:
   - Cuisine preference
   - Price range
   - Other available restaurant characteristics

3. Implement a **content-based filtering** approach.

4. Compare restaurant characteristics with the user's preferences and generate suitable recommendations.

5. Test the system using sample user preferences and evaluate the relevance of the recommendations.

## Example

A user may provide preferences such as:

```text
Cuisine: Italian
Price Range: Moderate
```

The recommendation system then identifies restaurants whose characteristics best match these preferences.

## Recommendation Approach

**Content-Based Filtering**

The system focuses on the characteristics of restaurants and the user's selected preferences rather than relying primarily on other users' ratings.

---

# Task 4 — Location-Based Analysis

## Objective

Perform a **geographical analysis of restaurants** and identify patterns related to restaurant locations, ratings, cuisines, and price ranges.

## Workflow

1. Explore the latitude and longitude coordinates of restaurants.

2. Visualize the geographical distribution of restaurants.

3. Group restaurants by city or locality.

4. Analyze the concentration of restaurants across different areas.

5. Calculate statistics such as:
   - Average ratings
   - Popular cuisines
   - Price ranges
   - Number of restaurants

6. Identify interesting geographical patterns and relationships between restaurant location and other characteristics.

## Questions Explored

The analysis can help investigate questions such as:

- Which locations have the highest concentration of restaurants?
- Which cities or localities have higher average ratings?
- Which cuisines are common in different areas?
- How do restaurant price ranges vary by location?
- Are there noticeable geographical patterns in restaurant distribution?

---

# Technologies Used

- **Python** — Primary programming language
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical computation
- **Scikit-learn** — Machine learning
- **Matplotlib** — Data visualization
- **Seaborn** — Statistical visualization

---

# Skills Demonstrated

These projects demonstrate practical experience with:

- Data cleaning and preprocessing
- Missing-value handling
- Categorical variable encoding
- Exploratory Data Analysis (EDA)
- Regression
- Machine learning model development
- Model evaluation
- Feature analysis
- Content-based recommendation systems
- Geographical data analysis
- Statistical analysis
- Data visualization
- Python-based data science workflows

---

# Getting Started

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Cognify-Machine-Learning-Tasks
```

## 2. Install Dependencies

Install the commonly used Python libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

If a `requirements.txt` file is provided in the repository, use:

```bash
pip install -r requirements.txt
```

## 3. Run a Task

Navigate to the required task directory and run the corresponding Python file or notebook.

For example:

```bash
cd task1_predict_ratings
python <script_name>.py
```

The same approach can be followed for:

```text
task2_recommendation_system/
task4_location_analysis/
```

---

# Project Overview

The three tasks demonstrate different ways of extracting value from restaurant data:

**Prediction → Recommendation → Analysis**

- **Task 1:** Predict restaurant ratings using regression.
- **Task 2:** Recommend restaurants based on user preferences.
- **Task 4:** Analyze restaurant distribution and characteristics geographically.

Together, these projects demonstrate an end-to-end application of Python, data analysis, machine learning, recommendation techniques, and visualization to a real-world dataset.

---

# License

This project is distributed under the license included in the repository.

---

## Author

**Cognify Machine Learning Internship**

This repository was developed as part of the Cognify internship Machine Learning tasks.
