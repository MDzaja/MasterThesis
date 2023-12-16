import sys
sys.path.insert(0, '../../')

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
import json

import LSTM as lstm_impl
import CNN_LSTM as cnn_lstm_impl
import utils as model_utils
import transformer as tr_impl

if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    raw_data, features_df, labels_dict = model_utils.get_aligned_raw_feat_lbl()

    raw_X = model_utils.get_X(raw_data, 60)
    feat_X = model_utils.get_X(features_df, 30)[30:]
    raw_Y_dict = {key: model_utils.get_Y(series, 60) for key, series in labels_dict.items()}
    feat_Y_dict = {key: model_utils.get_Y(series, 30)[30:] for key, series in labels_dict.items()}

    raw_X_train, raw_X_val, feat_X_train, feat_X_val = train_test_split(raw_X, feat_X, test_size=0.2, shuffle=False)
    raw_X_val, raw_X_test, feat_X_val, feat_X_test = train_test_split(raw_X_val, feat_X_val, test_size=0.25, shuffle=False)

    raw_Y_train_dict, raw_Y_val_dict, feat_Y_train_dict, feat_Y_val_dict = {}, {}, {}, {}
    raw_Y_test_dict, feat_Y_test_dict = {}, {}
    for labeling in labels_dict.keys():
        raw_Y_train_dict[labeling], raw_Y_val_dict[labeling], feat_Y_train_dict[labeling], feat_Y_val_dict[labeling] = train_test_split(raw_Y_dict[labeling], feat_Y_dict[labeling], test_size=0.2, shuffle=False)
        raw_Y_val_dict[labeling], raw_Y_test_dict[labeling], feat_Y_val_dict[labeling], feat_Y_test_dict[labeling] = train_test_split(raw_Y_val_dict[labeling], feat_Y_val_dict[labeling], test_size=0.25, shuffle=False)

    # Xs: raw_X_train, raw_X_val, raw_X_test, feat_X_train, feat_X_val, feat_X_test
    # Ys: raw_Y_train_dict, raw_Y_val_dict, raw_Y_test_dict, feat_Y_train_dict, feat_Y_val_dict, feat_Y_test_dict

    #########################################################

    labeling = 'triple_barrier'
    X_train, X_val, X_test = raw_X_train, raw_X_val, raw_X_test
    Y_train, Y_val, Y_test = raw_Y_train_dict[labeling], raw_Y_val_dict[labeling], raw_Y_test_dict[labeling]

    print(Y_train.shape)
    print("Y_train value counts:")
    print(Y_train.value_counts())

    early_stopping = EarlyStopping(monitor=model_utils.get_default_monitor_metric(), patience=20)

    classes = np.unique(Y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=Y_train.values.flatten())
    class_weight_dict = dict(zip(classes, class_weights))

    model = lstm_impl.build_model_raw(X_train.shape[-2], X_train.shape[-1])
    model.fit(X_train, Y_train, epochs=200, validation_data=(X_val, Y_val), 
              batch_size=64, class_weight=class_weight_dict, callbacks=[early_stopping])
    
    train_eval = model.evaluate(X_train, Y_train)
    val_eval = model.evaluate(X_val, Y_val)
    test_eval = model.evaluate(X_test, Y_test)

    print(f"Model metric names: {model.metrics_names}")
    print(f"Train: {train_eval}")
    print(f"Val: {val_eval}")
    print(f"Test: {test_eval}")