"""
This is a config file to store local settings like hyperparameters and paths
"""
import os

# paths
unocg_path = os.path.dirname(os.path.realpath(__file__))
# path to the directory where the data is stored
data_path = os.path.abspath(os.path.join(unocg_path, "..", "data"))
# path to the directory where the results should be stored
results_path = os.path.abspath(os.path.join(data_path, "results"))
# create results directory if it does not exists
if not os.path.exists(results_path):
    os.makedirs(results_path)
