from constants import data_file_path
from LP_toolkits import normalizer
import pandas as pd


def get_processed_data(csv_file_add):
    csv_data = pd.read_csv(csv_file_add)

    data = {'review': [], 'condition': [], 'drug': []}

    for index in range(csv_data.shape[0]):
        data['condition'].append(normalizer(str(csv_data['condition'][index])))
        data['drug'].append(normalizer(str(csv_data['drugName'][index])))
        data['review'].append(normalizer(str(csv_data['review'][index])))

    return data


if __name__ == '__main__':
    train_add = data_file_path + 'DrugsComTrain_raw.csv'
    test_add = data_file_path + 'DrugsComTest_raw.csv'
    train = get_processed_data(train_add)
    print('--------------------------------     See Train Results    --------------------------------')
    for condition, drug, review in zip(train['condition'], train['drug'], train['review']):
        print('{}:  {}:     {}'.format(condition, drug, review))

    test = get_processed_data(test_add)
    print('--------------------------------     See Test Results    --------------------------------')
    for condition, drug, review in zip(test['condition'], test['drug'], test['review']):
        print('{}:  {}:     {}'.format(condition, drug, review))
