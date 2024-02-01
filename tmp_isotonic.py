import pickle
from sklearn.isotonic import IsotonicRegression
import numpy as np
from models import utils as model_utils
import pandas as pd

Ys = {}
for label_name in ["ct_two_state", "ct_three_state", "fixed_time_horizon", "oracle"]:
    labels = model_utils.load_labels('/home/mdzaja/MasterThesis/artifacts/assets/AAPL/labels/all_labels_calibration_1m_2012-11-07_2012-11-20.pkl', label_name)
    Ys[label_name] = model_utils.get_Y_or_W_day_separated(labels, 60)

with open("/home/mdzaja/MasterThesis/artifacts/assets/AAPL/models/test_logs/detailed/calibration_probs.pkl", 'rb') as file:
    probs_calibration_dict = pickle.load(file)

with open("/home/mdzaja/MasterThesis/artifacts/assets/AAPL/models/test_logs/detailed/test_probs.pkl", 'rb') as file:
    probs_test_dict = pickle.load(file)

# Initialize the dictionary to store the calibrated test probabilities
probs_test_cal_dict = {}

for model_name, model_info in probs_calibration_dict.items():
    # Extract calibration data
    y_cal = Ys[model_info['label_name']]
    X_cal = model_info['probs'].loc[y_cal.index]
    
    # Fit the isotonic regression model
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(X_cal, y_cal)
    
    # Now calibrate the test probabilities using the fitted model
    X_test = probs_test_dict[model_name]['probs']
    X_test_nonzero = X_test[X_test != 0]
    y_test_calibrated = iso.transform(X_test_nonzero)
    probs_s = pd.Series(y_test_calibrated, index=X_test_nonzero.index)
    combined_index = probs_s.index.union(X_test.index)
    probs_s = probs_s.reindex(combined_index, fill_value=0)

    
    # Store the calibrated test probabilities in the new dict
    probs_test_cal_dict[model_name] = {
        'label_name': probs_test_dict[model_name]['label_name'],  # assuming you want to keep the same label name
        'probs': probs_s
    }

# Optionally, you can save the calibrated probabilities to a file
with open('/home/mdzaja/MasterThesis/artifacts/assets/AAPL/models/test_logs/detailed/test_probs_calibrated.pkl', 'wb') as file:
    pickle.dump(probs_test_cal_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
