from models import utils as model_utils
import os
from tensorflow.keras.models import load_model
import pickle
import pandas as pd

train_data = model_utils.load_data("/home/mdzaja/MasterThesis/artifacts/assets/AAPL/data/feat/train_1m_2010-10-11_2012-11-06.csv")
data = model_utils.load_data("/home/mdzaja/MasterThesis/artifacts/assets/AAPL/data/feat/test_1m_2012-11-21_2012-12-04.csv")
X = model_utils.get_X_day_separated(data, 60, train_data)
labels = model_utils.load_labels('/home/mdzaja/MasterThesis/artifacts/assets/AAPL/labels/all_labels_test_1m_2012-11-21_2012-12-04.pkl', 'oracle')
Y = model_utils.get_Y_or_W_day_separated(labels, 60)

model_dir = '/home/mdzaja/MasterThesis/artifacts/assets/AAPL/models/test_logs/detailed/saved_models'

results_dict = {}

for filename in os.listdir(model_dir):
    if filename.endswith('.keras'):
        # Parse the filename
        data_type, label, weight, model_name = filename.replace('.keras', '').split('-')
        
        # Load the model
        model_path = os.path.join(model_dir, filename)
        model = load_model(model_path)
        
        # Make predictions using the model
        probs_arr = model.predict(X).flatten()
        probs_s = pd.Series(probs_arr, index=Y.index)
        combined_index = probs_s.index.union(data.index)
        probs_s = probs_s.reindex(combined_index, fill_value=0)
        
        # Format the combination name and store the results
        combination_name = f'D-{data_type};L-{label};W-{weight};M-{model_name}'
        results_dict[combination_name] = {
            'label_name': label,
            'probs': probs_s
        }

# Save the dictionary to a .pkl file
with open('/home/mdzaja/MasterThesis/artifacts/assets/AAPL/models/test_logs/detailed/test_probs.pkl', 'wb') as handle:
    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)