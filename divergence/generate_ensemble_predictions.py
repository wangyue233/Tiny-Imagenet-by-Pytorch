import divergence.ensemble as de
import divergence.data as dd

import os
import numpy as np
import pandas as pd

if __name__ == "__main__":
    
    ######
    # 1. Load predictions from files
    ######
    
    this_folder = os.path.dirname(__file__)
    prediction_folder = os.path.join(this_folder, "predictions")
    
    individual_predictions = dict()
    
    for prediction_csv in os.listdir():
        individual_predictions[prediction_csv.split(".")[0]] = pd.read_csv(
            os.path.join(prediction_folder, prediction_csv), sep=',', index_col=0,
            header=0
        )['label']
    
    all_predictions = pd.DataFrame(individual_predictions)


    ######
    # 2. Load probability predictions from file
    ######
    
    probabilities_folder = os.path.join(this_folder, "probability_predictions")
    
    test_1 = np.load(os.path.join(probabilities_folder, "ensemble_probs.npz"))
    test_2 = np.load(os.path.join(probabilities_folder, "ensemble_probs_2.npz"))
    test_3 = np.load(os.path.join(probabilities_folder, "ensemble_probs_3.npz"))

    test_probs = dict()

    try:
        for name in test_1.files:
            if "vgg16" in name:
                continue
            assert name not in test_probs, "doubled {}".format(name)
            test_probs[name] = test_1[name]
        for name in test_2.files:
            if "vgg16" in name:
                continue
            assert name not in test_probs, "doubled {}".format(name)
            test_probs[name] = test_2[name]
        for name in test_3.files:
            if "vgg16" in name:
                continue
            assert name not in test_probs, "doubled {}".format(name)
            test_probs[name] = test_3[name]


    except AssertionError as e:
        print(e)

    finally:
        test_1.close()
        test_2.close()
        test_3.close()
        
    train_1 = np.load(os.path.join(probabilities_folder, "ensemble_train.npz"))
    train_2 = np.load(os.path.join(probabilities_folder, "ensemble_train_2.npz"))

    train_probs = dict()

    try:
        for name in train_1.files:
            if 'vgg16' in name:
                continue
            assert name not in train_probs, "doubled {}".format(name)
            train_probs[name] = train_1[name]
        for name in train_2.files:
            if 'vgg16' in name:
                continue
            assert name not in train_probs, "doubled {}".format(name)
            train_probs[name] = train_2[name]

    except AssertionError as e:
        print(e)

    finally:
        train_1.close()
        train_2.close()
        
    model_names = [k for k in sorted(list(train_probs.keys())) if ("vgg16" not in k)]
    X_train = np.concatenate([train_probs[k] for k in model_names[5:]], axis=-1)
    X_test = np.concatenate([test_probs[k] for k in model_names[3:]], axis=-1)
    
    
    ######
    # 3. Make predictions using different methods
    ######
    
    npz_dump = os.path.join(probabilities_folder, "npz_dump.npz")
    train_set = dd.ACSE44Dataset(npz_dump, train=True)
    test_set = dd.ACSE44Dataset(npz_dump, train=False)
    
    filenames = test_set.filenames
    train_labels = train_set.targets
    
    # XG-Boost
    xgboost_predictions = de.combine_predictions(X_test, filenames, method=de.CombinationMethod.xgboost, train_probabilities=X_train, train_labels=train_labels)
    print(xgboost_predictions)
    
    # Hard voting
    hard_voting_predictions = de.hard_voting(all_predictions.to_numpy())
    hard_voting_predictions = pd.DataFrame({"Filename": filenames, "Label": hard_voting_predictions})
    print(hard_voting_predictions)
    
    # Logistic regression
    logistic_predictions = de.combine_predictions(X_test, filenames, method=de.CombinationMethod.logistic_regression, train_probabilities=X_train, train_labels=train_labels)
    print(logistic_predictions)
