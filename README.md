# House Price Prediction

## Problem
- Housing prices are hard to estimate manually because many factors affect them simultaneously
- This project uses the California census dataset with ~20,000 districts
- The goal is to predict median_house_value for any given district
- this gives a rough idea of the house prices across the state for people to get educated on this 

## Project Structure
house-price-prediction/
├── data/
│   └── housing.csv
├── notebooks/
│   ├── 01_analyze_the_data.ipynb
│   ├── 02_creating_a_test_set.ipynb
│   ├── 03_further_preprocessing.ipynb
│   ├── 04_handling_categorical_attributes.ipynb
│   ├── 05_feature_scaling.ipynb
│   ├── 06_sklearn_pipelines.ipynb
│   └── 07_first_ml_model.ipynb
├── src/
│   └── pipeline.py
├── main.py
├── requirements.txt
└── README.md

## Dataset 
- Source: California Housing dataset from the 1990 census ~20,640 records, each representing one district
- Features include: median_income, housing_median_age, total_rooms, total_bedrooms, population, households, latitude, longitude, ocean_proximity
- Target variable: median_house_value
- One categorical feature (ocean_proximity), rest are numerical


## Approach
- Stratified Split — used StratifiedShuffleSplit on income categories instead of random split. Income is the strongest predictor of house prices so preserving its distribution in both train and test sets prevents a biased evaluation.
- Preprocessing Pipeline — built separate pipelines for numerical and categorical features using ColumnTransformer. Numerical: median imputation for missing values + standard scaling. Categorical: OneHotEncoding for ocean_proximity. Pipeline is saved with joblib so inference uses identical preprocessing.
- saved the pipeline first as if we fit and transform on the test data that would be data leakage so we input the learned preprocessed data from the train data to the test data which avoid data leakage
- Model — Random Forest Regressor. Chosen because it handles non-linear relationships well and is robust to outliers compared to Linear Regression.
- Hyperparameter Tuning — tuned manually by adjusting n_estimators, max_depth, min_samples_leaf, min_samples_split to reduce overfitting. Training RMSE intentionally increased from 18k to 33k while keeping test RMSE stable — showing better generalization.
- Two Mode Design — main.py runs in training mode if no model exists, inference mode if model is already saved. Mimics a real deployment pattern.


## Results
| Metric        | Before Tuning | After Tuning |
|---------------|---------------|--------------|
| Training RMSE | 18,342        | 33,110       |
| Test RMSE     | 47,197        | 47,312       |
| R² Score      | 0.8291        | 0.8282       |

- the gab between training and test RMSEs is overfitting
- Tuning reduced this gap by constraining tree depth and minimum sample requirements
- R² of 0.828 means the model explains 82.8% of variance in house prices
- Test RMSE of ~$47k on a median house price of ~$200k represents roughly 23% average error

## How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/house-price-prediction
cd house-price-prediction

# Install dependencies
pip install -r requirements.txt

# Train the model (first run)
python main.py

# Run inference (second run — uses saved model)
python main.py
```

## Key Learnings
- stratified sampling is better than normal spliting as it allows proper distribution of the data in both training and test set we did that keeping median_income as a parameter as that is one of the most important feature so it should be in both test and training data
- data leaking happens when we fit the scalling model on the test data we prevent that by doing fit_transform to the training data and then tranforming the test data from the learned parameters from he train_prepared
- large gab between training and test RMSEs means the model is overfitted like it will not predict good on unseen data as it as memorise the training data too much not learned pattern
- we use RMSE not accuracy for regression problems because we have to see how much off is the predicted label  from the actuall label as it is a contiues value not a binary or very specific like classification problem

## Future Improvements

- Add Streamlit UI for interactive house price prediction
- Systematic hyperparameter tuning using GridSearchCV
- Feature engineering with ratio features inside the pipeline
- REST API using FastAPI for model serving

