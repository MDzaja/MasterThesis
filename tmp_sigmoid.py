import pickle
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from models import utils as model_utils

def parse_combination_name(name):
    parts = name.split(';')
    return {p.split('-')[0]: p.split('-')[1] for p in parts}

train_data = model_utils.load_data("/home/mdzaja/MasterThesis/artifacts/assets/AAPL/data/feat/train_1m_2010-10-11_2012-11-06.csv")
cal_data = model_utils.load_data("/home/mdzaja/MasterThesis/artifacts/assets/AAPL/data/feat/calibration_1m_2012-11-07_2012-11-20.csv")
cal_X = model_utils.get_X_day_separated(cal_data, 60, train_data)
test_data = model_utils.load_data("/home/mdzaja/MasterThesis/artifacts/assets/AAPL/data/feat/test_1m_2012-11-21_2012-12-04.csv")
test_X = model_utils.get_X_day_separated(test_data, 60, train_data)

cal_Ys = {}
for label_name in ["ct_two_state", "ct_three_state", "fixed_time_horizon", "oracle"]:
    labels = model_utils.load_labels('/home/mdzaja/MasterThesis/artifacts/assets/AAPL/labels/all_labels_calibration_1m_2012-11-07_2012-11-20.pkl', label_name)
    cal_Ys[label_name] = model_utils.get_Y_or_W_day_separated(labels, 60)

test_label = model_utils.load_labels('/home/mdzaja/MasterThesis/artifacts/assets/AAPL/labels/all_labels_test_1m_2012-11-21_2012-12-04.pkl', 'oracle')
test_y = model_utils.get_Y_or_W_day_separated(test_label, 60)

with open("/home/mdzaja/MasterThesis/artifacts/assets/AAPL/models/test_logs/detailed/test_probs.pkl", 'rb') as file:
    probs_test_dict = pickle.load(file)

# Initialize the dictionary to store the calibrated test probabilities
probs_test_cal_dict = {}

for model_name, model_info in probs_test_dict.items():
    parts = parse_combination_name(model_name)
    keras_file_name = f"{parts['D']}-{parts['L']}-{parts['W']}-{parts['M']}.keras"
    # Load the pre-trained Keras model
    model = load_model(f'/home/mdzaja/MasterThesis/artifacts/assets/AAPL/models/test_logs/detailed/saved_models/{keras_file_name}')
    
    # Extract calibration data
    y_cal = cal_Ys[model_info['label_name']]
    
    # Calibrate the classifier on the validation data
    sig_clf = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    sig_clf.fit(cal_X, y_cal)
    
    # Now calibrate the test probabilities using the fitted sigmoid model
    y_test_calibrated = sig_clf.predict_proba(test_X)[:, 1]  # Get the probability for the positive class
    
    # Reconstruct the calibrated probabilities series
    probs_s = pd.Series(y_test_calibrated, index=test_y.index)
    combined_index = probs_s.index.union(test_data.index)
    probs_s = probs_s.reindex(combined_index, fill_value=0)

    # Store the calibrated test probabilities in the new dict
    probs_test_cal_dict[model_name] = {
        'label_name': probs_test_dict[model_name]['label_name'],  # assuming you want to keep the same label name
        'probs': probs_s
    }

# Save the calibrated probabilities to a file
with open('/home/mdzaja/MasterThesis/artifacts/assets/AAPL/models/test_logs/detailed/test_probs_sigmoid_calibrated.pkl', 'wb') as file:
    pickle.dump(probs_test_cal_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
