import pickle
import argparse
import pandas as pd
from utils import load_model, prepare_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path", help="Model pkl file path you wish to use", type=str)
    parser.add_argument(
        "test_path", help="Data file path you wish to use for testing", type=str)

    args = parser.parse_args()
    
    # load the model
    model_path = args.model_path
    model = load_model(model_path)

    # load the data file
    test_path = args.test_path
    testdata = pd.read_csv(test_path)

    # Preprocess the data
    testdata = prepare_data(testdata)

    # predict and save them to preds.txt
    predictions = model.predict(testdata)

    

    with open("preds.txt", "w") as fp:
        fp.writelines('%s\n' % '\t'.join(class_out) for class_out in predictions)

