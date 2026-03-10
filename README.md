# NYC Yellow Taxi Fare Prediction Using Big Data and Machine Learning

## Overview

This project builds a scalable data pipeline to predict NYC Yellow Taxi fare amounts using large-scale trip records. It leverages Apache Spark (PySpark) and big data tools to demonstrate the full lifecycle of Data Engineering and Machine Learning, including:

- Ingesting and cleaning large-scale NYC taxi trip data
- Engineering features such as trip duration, time of day, pickup/dropoff zones, and distance
- Training and evaluating multiple regression models (Linear Regression, Random Forest, Hypertuned Random Forest, Gradient Boosted Trees) to predict fare amounts
- Scaling the entire workflow across varying dataset sizes and cluster configurations using Spark
- Visualizing key trends, fare distributions, and model performance

**Real-World Applications & Impact:**
- Help passengers estimate fare costs before booking a ride
- Enable city planners and taxi companies to analyze pricing trends and demand patterns
- Demonstrate end-to-end big data pipeline skills using real-world transportation data

---

## Project Objectives

- Build a scalable data pipeline to process and analyze NYC Yellow Taxi trip records using Apache Spark
- Engineer key features such as trip duration, pickup hour, day of week, and passenger count to improve fare prediction accuracy
- Train and compare four supervised regression models to forecast taxi fares
- Ensure scalability and performance optimization across dataset sizes (25%–100%) and cluster configurations (2, 3, and 4 worker nodes)
- Evaluate models using RMSE, MAE, R², and inference throughput for production readiness
- Visualize data distributions, trends, and model outputs to derive actionable insights

---

## Dataset Overview

This project utilizes three primary datasets combined to build a richer and more accurate fare prediction pipeline:

### 1. Taxi Data — NYC TLC Yellow Taxi Dataset
- **Source:** [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **Storage Format:** Parquet / CSV
- Core trip records including pickup/dropoff times, distances, passenger counts, and fare amounts

**Key Features Include:**

Trip Details:
- Pickup and dropoff location IDs (zones)
- Trip distance and duration
- Passenger count

Temporal Features:
- Pickup datetime (used to compute hour of day, day of week, month)
- Dropoff datetime

Fare Components:
- Fare amount *(target variable)*
- Extra charges, MTA tax, tip amount, tolls
- Total amount and payment type

### 2. Weather Data — Visual Crossing Weather API
- **Source:** [Visual Crossing Weather API](https://www.visualcrossing.com/)
- Historical weather conditions matched to taxi trip timestamps
- Features include temperature, precipitation, wind speed, and weather conditions (rain, snow, clear)
- Weather data enriches the model by capturing how conditions influence demand and fare patterns

### 3. Location Data — TLC Zone Lookup & Geolocation API
- **Source:** NYC TLC Zone Lookup Table + Geolocation API
- Maps numeric pickup/dropoff location IDs to named NYC zones and boroughs
- Geolocation API used to enrich zone data with latitude/longitude coordinates
- Enables spatial analysis of high-demand areas and cross-borough trip patterns

---

## Technologies Used

| Technology | Role & Purpose |
|---|---|
| Apache Spark (PySpark) | Distributed data processing and scalable ML model training |
| Python | Scripting, data pipeline orchestration, and model evaluation |
| Pandas / NumPy | Data manipulation and preprocessing |
| Spark MLlib | Machine learning library for regression model implementation |
| Matplotlib / Seaborn | Data visualization and EDA plots |
| Parquet | Columnar storage format optimized for big data processing |
| Jupyter Notebooks | Interactive development and exploratory data analysis |

---

## Data Processing & Feature Engineering

- Removed irrelevant columns such as store-and-forward flags and redundant charge fields
- Filtered out invalid records including trips with zero distance, negative fares, or passenger count of zero
- Engineered key features including:
  - `trip_duration_minutes`: calculated from pickup and dropoff timestamps
  - `pickup_hour`: hour of day extracted from pickup datetime to capture time-based demand
  - `day_of_week`: day extracted from pickup datetime to model weekday vs weekend patterns
  - `is_rush_hour`: binary indicator for morning and evening rush hours
- Handled missing values using mean/median imputation for continuous features
- Encoded categorical variables (e.g., payment type, vendor ID) for ML model compatibility
- Stored processed data efficiently in Parquet format for fast access and downstream modeling

---

## Exploratory Data Analysis (EDA)

- Analyzed fare distributions to detect pricing trends and outliers across boroughs and time periods
- Explored trip distance vs fare amount correlation to validate feature relevance
- Identified peak demand hours and high-traffic pickup zones across NYC
- Performed outlier detection using the Interquartile Range (IQR) method on fare and distance columns
- Examined tip amount patterns across payment types and time of day
- Visualized pickup/dropoff zone frequency to understand route popularity and demand hotspots

---

## Machine Learning Models

This project implements and compares four supervised regression models using PySpark MLlib:

### 1. Linear Regression (Baseline)
- Interpretable model to establish baseline performance
- Fast training time on large datasets
- Validates feature relevance and data quality

#### 2. Random Forest Regression
- Ensemble of decision trees reducing variance and overfitting
- Handles non-linear relationships between features and fare
- Robust to noisy features and missing values

#### 3. Hypertuned Random Forest Regression
- Builds on base Random Forest with optimized hyperparameters via cross-validation
- Tuned parameters include number of trees, max depth, and feature subsampling
- Achieves better generalization compared to default Random Forest settings

#### 4. Gradient Boosted Trees (GBT)
- Sequentially builds trees to minimize prediction error
- Captures complex non-linear pricing patterns driven by distance, time, and location
- Delivers the lowest MAE across all dataset sizes and cluster configurations

**MLlib Pipeline:**
```
StringIndexer → OneHotEncoder → VectorAssembler → Regressor
```

---

## Model Evaluation

All models were evaluated across three Spark cluster configurations and four dataset sizes (25%, 50%, 75%, 100%).

### Metrics Used

| Metric | Description |
|---|---|
| RMSE | Measures the average magnitude of prediction errors, penalizing larger errors more heavily. Lower is better. |
| MAE | Captures the average absolute difference between predicted and actual fare values. Lower is better. |
| R² Score | Indicates how well the model explains variance in fare amounts. Closer to 1.0 is better. |
| Throughput | Measures inference speed in predictions per second, indicating model suitability for real-time or batch deployment. Higher is better. |

**Why these metrics?**
- **RMSE** is sensitive to large errors, making it useful for catching outlier fare mispredictions
- **MAE** gives a more intuitive sense of average dollar error per trip prediction
- **R²** provides a normalized measure of model fit regardless of fare scale
- **Throughput** validates that the model can handle production-level data volumes efficiently

---

### Results

#### 1. Linear Regression

**1 Master + 2 Worker Nodes**

| Dataset | RMSE | MAE | R² | Throughput (preds/sec) |
|---|---|---|---|---|
| 25% | 3.945 | 1.904 | 0.915 | 10,994,420.85 |
| 50% | 3.955 | 1.905 | 0.915 | 32,688,220.92 |
| 75% | 3.951 | 1.906 | 0.915 | 30,691,696.36 |
| 100% | 3.975 | 1.909 | 0.914 | 115,768,281.71 |

**1 Master + 3 Worker Nodes**

| Dataset | RMSE | MAE | R² | Throughput (preds/sec) |
|---|---|---|---|---|
| 25% | 3.945 | 1.904 | 0.915 | 56,480,937.65 |
| 50% | 3.955 | 1.905 | 0.915 | 150,509,504.76 |
| 75% | 3.951 | 1.906 | 0.915 | 186,304,100.83 |
| 100% | 3.975 | 1.909 | 0.914 | 206,672,809.13 |

**1 Master + 4 Worker Nodes**

| Dataset | RMSE | MAE | R² | Throughput (preds/sec) |
|---|---|---|---|---|
| 25% | 3.967 | 1.907 | 0.915 | 26,671,240.30 |
| 50% | 3.967 | 1.907 | 0.915 | 29,387,240.09 |
| 75% | 3.962 | 1.905 | 0.915 | 202,153,250.72 |
| 100% | 3.965 | 1.908 | 0.915 | 186,919,730.39 |

---

#### 2. Random Forest Regression

**1 Master + 2 Worker Nodes**

| Dataset | RMSE | MAE | R² | Throughput (preds/sec) |
|---|---|---|---|---|
| 25% | 4.771 | 2.089 | 0.876 | 5,141,734.29 |
| 50% | 4.782 | 2.083 | 0.876 | 12,358,954.26 |
| 75% | 4.774 | 2.068 | 0.876 | 25,213,523.83 |
| 100% | 4.807 | 2.078 | 0.875 | 28,979,705.13 |

**1 Master + 3 Worker Nodes**

| Dataset | RMSE | MAE | R² | Throughput (preds/sec) |
|---|---|---|---|---|
| 25% | 4.658 | 2.069 | 0.881 | 31,444,816.40 |
| 50% | 4.671 | 2.077 | 0.881 | 81,646,249.40 |
| 75% | 4.667 | 2.063 | 0.882 | 107,397,188.21 |
| 100% | 4.705 | 2.065 | 0.880 | 28,306,195.14 |

**1 Master + 4 Worker Nodes**

| Dataset | RMSE | MAE | R² | Throughput (preds/sec) |
|---|---|---|---|---|
| 25% | 4.651 | 2.089 | 0.883 | 9,415,786.53 |
| 50% | 4.643 | 2.061 | 0.883 | 30,890,401.53 |
| 75% | 4.676 | 2.099 | 0.881 | 8,737,350.85 |
| 100% | 4.666 | 2.095 | 0.882 | 167,929,428.39 |

---

#### 3. Hypertuned Random Forest Regression

**1 Master + 2 Worker Nodes**

| Dataset | RMSE | MAE | R² | Throughput (preds/sec) |
|---|---|---|---|---|
| 25% | 4.649 | 1.958 | 0.883 | 8,786,700.77 |
| 50% | 4.657 | 1.958 | 0.882 | 22,454,003.95 |
| 75% | 4.650 | 1.943 | 0.882 | 42,840,358.32 |
| 100% | 4.680 | 1.948 | 0.881 | 23,845,681.49 |

**1 Master + 3 Worker Nodes**

| Dataset | RMSE | MAE | R² | Throughput (preds/sec) |
|---|---|---|---|---|
| 25% | 4.465 | 1.908 | 0.891 | 30,809,785.31 |
| 50% | 4.478 | 1.913 | 0.891 | 30,272,934.86 |
| 75% | 4.475 | 1.914 | 0.891 | 50,764,596.42 |
| 100% | 4.533 | 1.933 | 0.889 | 113,014,200.16 |

**1 Master + 4 Worker Nodes**

| Dataset | RMSE | MAE | R² | Throughput (preds/sec) |
|---|---|---|---|---|
| 25% | 4.442 | 1.923 | 0.893 | 194,020,235.30 |
| 50% | 4.466 | 1.914 | 0.892 | 30,325,527.84 |
| 75% | 4.491 | 1.933 | 0.890 | 48,834,613.10 |
| 100% | 4.460 | 1.920 | 0.892 | 101,697,542.18 |

---

#### 4. Gradient Boosted Trees (GBT)

**1 Master + 2 Worker Nodes**

| Dataset | RMSE | MAE | R² | Throughput (preds/sec) |
|---|---|---|---|---|
| 25% | 4.042 | 1.699 | 0.911 | 8,074,920.66 |
| 50% | 4.108 | 1.733 | 0.908 | 41,564,635.81 |
| 75% | 3.941 | 1.706 | 0.916 | 95,064,614.26 |
| 100% | 4.091 | 1.731 | 0.910 | 35,550,456.07 |

**1 Master + 3 Worker Nodes**

| Dataset | RMSE | MAE | R² | Throughput (preds/sec) |
|---|---|---|---|---|
| 25% | 4.042 | 1.699 | 0.911 | 22,667,750.04 |
| 50% | 4.108 | 1.733 | 0.908 | 79,387,499.53 |
| 75% | 3.941 | 1.706 | 0.915 | 23,727,244.61 |
| 100% | 4.091 | 1.731 | 0.909 | 37,944,591.02 |

**1 Master + 4 Worker Nodes**

| Dataset | RMSE | MAE | R² | Throughput (preds/sec) |
|---|---|---|---|---|
| 25% | 4.454 | 1.779 | 0.892 | 4,862,630.56 |
| 50% | 4.411 | 1.742 | 0.894 | 20,629,649.41 |
| 75% | 4.453 | 1.762 | 0.892 | 38,473,926.98 |
| 100% | 4.436 | 1.762 | 0.893 | 135,618,869.22 |

---

### Key Takeaways

- **Linear Regression** achieved the best R² (0.914–0.915) and lowest RMSE (~3.95) overall, making it a strong and efficient baseline for this dataset
- **GBT** delivered the lowest MAE (~1.70) across cluster sizes, indicating the most accurate per-prediction estimates
- **Hypertuned Random Forest** outperformed base Random Forest across all configurations, confirming the value of hyperparameter optimization
- **Throughput scaled significantly** with additional worker nodes — Linear Regression on 3 workers reached 206M+ predictions/sec on the full dataset
- All models maintained **consistent accuracy** across dataset sizes, demonstrating stable and scalable pipelines

---

## Scalability & Cluster Configuration

The pipeline was tested across three Spark cluster configurations to evaluate scalability:

| Configuration | Workers | Use Case |
|---|---|---|
| 1 Master + 2 Worker Nodes | 2 | Baseline performance |
| 1 Master + 3 Worker Nodes | 3 | Mid-scale optimization |
| 1 Master + 4 Worker Nodes | 4 | High throughput workloads |

Adding worker nodes significantly increased inference throughput while keeping model accuracy stable, validating the distributed design of the pipeline.

---

## Project Structure

```
NYC-Yellow-Taxi-Price-Prediction/
│
├── data_cleaning.py      # Data preprocessing and feature engineering
├── models.py             # ML model training and evaluation (LR, RF, Hypertuned RF, GBT)
├── plots.py              # Data visualization and EDA plots
└── README.md
```

---

## Project Setup & Execution

### Option 1: Run on Google Cloud Platform (Recommended for full dataset)

1. Create a Google Cloud account and set up a project with billing enabled.
2. Enable the **Dataproc** and **Cloud Storage APIs** in your GCP project.
3. Upload the NYC TLC Taxi dataset, Weather data (Visual Crossing API), and Location data (TLC Zone Lookup) to a **Google Cloud Storage (GCS) bucket**.
4. Create a **Dataproc cluster** with a suitable configuration — recommended: 1 master + 2/3/4 worker nodes with sufficient memory and CPU based on dataset size.
5. Open the **Jupyter notebook interface** on Dataproc to run the pipeline notebooks directly in the cloud environment.
6. Make sure your **PySpark environment and Spark MLlib libraries** are available on the cluster.
7. Run the pipeline notebooks in order:
   - `data_cleaning.py` — preprocess and engineer features
   - `models.py` — train and evaluate all four models
   - `plots.py` — generate visualizations and EDA plots

> **Note:** The full pipeline was designed and executed on GCP Dataproc with the complete dataset stored in Google Cloud Storage (GCS). This is the recommended approach for reproducing the results across all cluster configurations.

---

### Option 2: Run Locally (for small-scale testing)

This section provides a simplified local setup to test the pipeline with a small sample of the dataset. The full pipeline is designed for GCP Dataproc.

### Prerequisites

- Python 3.7+
- Java 8+ (required for Spark)
- Apache Spark 3.x
- PySpark

### Installation

```bash
# Clone the repository
git clone https://github.com/Angelina1225/NYC-Yellow-Taxi-Price-Prediction.git
cd NYC-Yellow-Taxi-Price-Prediction

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn pyspark
```

### Running the Pipeline

**Step 1 — Start a local PySpark session**
```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("nyc_taxi_fare_prediction").getOrCreate()
```

**Step 2 — Data Cleaning & Feature Engineering**
```bash
python data_cleaning.py
```

**Step 3 — Model Training & Evaluation**
```bash
python models.py
```

**Step 4 — Generate Visualizations**
```bash
python plots.py
```

> **Note:** This local setup is for simulation and testing purposes only. For full dataset processing and accurate benchmarking across cluster configurations, use Option 1 (GCP Dataproc).

---

## Skills Demonstrated

- **Big Data Engineering:** Efficient processing of large-scale NYC taxi records using Apache Spark and PySpark
- **Feature Engineering:** Creating meaningful features such as `trip_duration_minutes`, `is_rush_hour`, and temporal indicators for ML models
- **Machine Learning:** Building and comparing four regression models — Linear Regression, Random Forest, Hypertuned Random Forest, and GBT
- **Hyperparameter Tuning:** Cross-validation and grid search to optimize Random Forest performance
- **Scalability Testing:** Benchmarking model performance across 2, 3, and 4 worker node cluster configurations
- **Exploratory Data Analysis:** Outlier detection, correlation analysis, and demand pattern insights across NYC zones
- **Data Visualization:** Charts and plots to communicate fare trends, trip distributions, and model performance
- **End-to-End Pipeline Design:** From raw data ingestion and cleaning to model training, evaluation, and visualization
- **Data Storage & Formats:** Utilizing Parquet columnar format for efficient big data storage and querying

---

## Challenges & Learnings

- **Handling data quality issues:** NYC taxi data contains numerous invalid records (zero-distance trips, negative fares) requiring robust filtering strategies
- **Feature engineering complexity:** Extracting meaningful temporal features from raw timestamps required careful datetime handling
- **Balancing model complexity vs. scalability:** GBT and Hypertuned RF improved accuracy but required more compute resources and careful tuning
- **Cluster configuration impact:** Throughput varied significantly across cluster sizes, requiring experimentation to identify optimal resource allocation
- **Outlier impact on regression:** Extreme fare values significantly affected model performance, requiring careful outlier treatment strategies

---

## Future Work

- **Incorporate geospatial features:** Use pickup/dropoff coordinates and zone shapefiles to build richer location-based features
- **Expand dataset scope:** Include Green Taxi and FHV (rideshare) data for broader NYC transportation analysis
- **Real-time prediction service:** Develop a REST API for live fare estimation based on trip inputs
- **Automate pipeline orchestration:** Use Apache Airflow to automate ETL, model training, and deployment steps
- **Deploy on GCP/AWS:** Scale the pipeline on cloud infrastructure using Dataproc or EMR for full dataset processing
- **Deep learning exploration:** Experiment with neural network architectures for improved fare prediction accuracy
- **Comprehensive monitoring:** Add logging and monitoring to track pipeline health and model drift over time
