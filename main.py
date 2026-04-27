import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from src.pipeline import build_pipelines
from sklearn.metrics import mean_squared_error, r2_score
import math

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

if not os.path.exists(MODEL_FILE):
    
    
    housing = pd.read_csv("data/housing.csv")

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0., 1.5, 3, 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    for train_index, test_index in split.split(housing, housing["income_cat"]):
        test_set = housing.loc[test_index].drop("income_cat", axis=1)
        test_labels = test_set["median_house_value"].copy()
        test_set.drop("median_house_value",axis=1).to_csv("input_csv",index=False)
        housing = housing.loc[train_index].drop("income_cat", axis=1)

    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)

    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]
        
    pipeline = build_pipelines(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)
    
    model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=4,
    min_samples_split=10,
    random_state=42
)
    model.fit(housing_prepared, housing_labels)

    train_prediction = model.predict(housing_prepared)

    test_features = test_set.drop("median_house_value",axis=1)
    test_prepared = pipeline.transform(test_features)

    test_prediction = model.predict(test_prepared)

    train_rmse = math.sqrt(mean_squared_error(housing_labels,train_prediction))
    test_rmse = math.sqrt(mean_squared_error(test_labels,test_prediction))

    test_r2 = r2_score(test_labels,test_prediction)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("model is trained congrats!")

    print(f"Training RMSE: {train_rmse:,.2f}")
    print(f"Test RMSE:     {test_rmse:,.2f}")
    print(f"R² Score:      {test_r2:.4f}")


else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data["median_house_value"] = predictions
    input_data.to_csv("output.csv", index=False)
    print('inference is complete result save to output.csv')