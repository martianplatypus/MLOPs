import pandas as pd
from sklearn.datasets import load_wine


def load_data():
    wine = load_wine()
    data = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    data["target"] = wine.target
    return data
