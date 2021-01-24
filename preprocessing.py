import os

import pandas as pd

from LP_toolkits import normalizer
from constants import data_file_path


def get_processed_data(csv_file_add):
    """

    :param csv_file_add: (string) path to were csv file is stored at (including it's filename)
    :return: data: (dictionary) contains only required data for classification tasks
    (which in the texts has been cleaned)

    data dictionary structure:
    {   'review': list of normalized reviews text (list of strings),
        'condition':list of normalized conditions name (list of strings),
         'drug': list of normalized drugs name (list of strings)
    }
     the reason for normalizing conditions and drugs name is that they might also be appeared in the reviews
    """

    # read csv file
    csv_data = pd.read_csv(csv_file_add)

    # define a dictionary to store required data
    data = {'review': [], 'condition': [], 'drug': []}

    for index in range(csv_data.shape[0]):
        # add normalized condition
        data['condition'].append(normalizer(str(csv_data['condition'][index])))
        # add normalized drug name
        data['drug'].append(normalizer(str(csv_data['drugName'][index])))
        # add normalized review
        data['review'].append(normalizer(str(csv_data['review'][index])))

    return data


if __name__ == '__main__':
    # define csv data paths
    train_add = os.path.join(data_file_path, 'drugsComTrain_raw.csv')
    test_add = os.path.join(data_file_path, 'drugsComTest_raw.csv')
    # call get_preprocessed_data on training data
    train = get_processed_data(train_add)
    # see the result of preprocessed training data
    print('--------------------------------     See Train Results    --------------------------------')
    for condition, drug, review in zip(train['condition'], train['drug'], train['review']):
        print('{}:  {}:     {}'.format(condition, drug, review))

    # call get_preprocessed_data on test data
    test = get_processed_data(test_add)
    # see the result of preprocessed test data
    print('--------------------------------     See Test Results    --------------------------------')
    for condition, drug, review in zip(test['condition'], test['drug'], test['review']):
        print('{}:  {}:     {}'.format(condition, drug, review))
