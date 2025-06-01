import pandas as pd
from sklearn import datasets

from evidently import Report
from evidently.presets import DataDriftPreset

iris_data = datasets.load_iris(as_frame=True)
iris_frame = iris_data.frame

report = Report([
    DataDriftPreset(method="psi")
],
include_tests=True)
my_eval = report.run(iris_frame.iloc[:60], iris_frame.iloc[60:])
print(my_eval)