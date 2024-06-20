# Healthcare Provider Recommendation Engine

A Python-based recommendation system for healthcare providers that uses collaborative filtering, clustering, and Pareto optimization to generate personalized provider recommendations based on synthetic patient data.

## Features

- **Synthetic Data Generation**: Creates realistic healthcare provider and patient rating data
- **Collaborative Filtering**: Uses SVD to predict patient preferences
- **Provider Clustering**: Groups similar providers using K-means
- **Pareto Optimization**: Finds optimal trade-offs between quality, cost, and predicted ratings
- **Command Line Interface**: Easy-to-use CLI for getting recommendations

## Project Structure

```
Model/
├── generate.py             # Synthetic data generation
├── pipeline.py             # Data preprocessing
├── collaborative_filtering.py  # SVD implementation
├── clustering.py           # K-means clustering
├── pareto_optimization.py  # Pareto front optimization
├── recommender.py          # Main recommendation engine
├── cli.py                  # Command line interface
├── tests/
│   ├── test_pipeline.py        # Tests for pipeline module
│   ├── test_recommender.py     # Tests for recommendation functionality
├── notebooks/
│   ├── analysis.ipynb          # Analysis and visualization notebook
├── README.md               # This file
```

## Installation

1. Install the required packages:
   ```
   pip install numpy pandas scikit-learn matplotlib pytest jupyter
   ```

## Usage

### Generating Data

First, generate synthetic provider and rating data:

```
python Model/generate.py
```

This will create two CSV files in a `data` directory:
- `providers_data.csv`: Information about healthcare providers
- `ratings_data.csv`: Patient ratings for providers

### Getting Recommendations

Use the command-line interface to get provider recommendations for a specific patient:

```
python Model/cli.py --user_id 1234 --top_n 10
```

Parameters:
- `--user_id`: Patient ID to generate recommendations for (required)
- `--top_n`: Number of recommendations to generate (default: 10)
- `--data_dir`: Directory containing the data files (default: 'data')

### Running the Analysis Notebook

To explore the data and visualize the recommendation results:

1. Start Jupyter Notebook:
   ```
   jupyter notebook
   ```

2. Open the `Model/notebooks/analysis.ipynb` file from the Jupyter interface

## Testing

Run the test suite with pytest:

```
pytest Model/tests/
```

## Methodology

The recommendation system uses a hybrid approach combining:

1. **Collaborative Filtering**: Learns patient preferences based on past ratings using SVD
2. **Content-Based Filtering**: Groups providers by features using K-means clustering
3. **Multi-Objective Optimization**: Finds Pareto-optimal solutions balancing:
   - Provider quality (maximize)
   - Provider cost (minimize)
   - Predicted patient rating (maximize)

This approach ensures recommendations are both personalized to the patient's preferences and optimized for quality/cost considerations.

## Performance

The system is optimized for efficiency:
- Uses sparse matrices and vectorized operations
- SVD implementation is O(nk) where n is the number of ratings and k is the number of factors
- Pareto optimization is implemented with O(n²) comparisons in the worst case

For the default dataset (10,000 providers, 100,000 ratings), the system can generate recommendations in seconds on a standard computer.
