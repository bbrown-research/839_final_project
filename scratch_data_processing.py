import numpy
import numpy as np
import pandas
import pandas as pd


def process_conditions_data(processed_file_path=None):
    data = pandas.read_csv("data/conditions.csv")

    used_cols = ['START', 'STOP', 'PATIENT', 'DESCRIPTION']
    save_data = []

    unique_patient_ids = data['PATIENT'].unique()
    patient_id_map = {id:patient_num for patient_num, id in enumerate(unique_patient_ids)}

    for i, row in data.iterrows():
        row = row[used_cols]
        if '(disorder)' in row['DESCRIPTION']:
            row['PATIENT'] = patient_id_map[row['PATIENT']]
            row['DESCRIPTION'] = row['DESCRIPTION'].replace(" (disorder)", "")

            if row['STOP'] is numpy.nan:
                row['STOP'] = 'No Stop Date'
            save_data.append(row)

    final_pd = pandas.DataFrame(save_data, columns=used_cols)

    if processed_file_path is not None:
        final_pd.to_csv(processed_file_path, header=None, index=None, sep=',')

    return final_pd

def make_train_test_split_data(raw_data_path='raw_data/', training_percentage=0.7, save_as_csv=True, random_seed=12345):
    allergies_pd = pd.read_csv(f"{raw_data_path}/allergies.csv")
    conditions_pd = pd.read_csv(f"{raw_data_path}/conditions.csv")
    medications_pd = pd.read_csv(f"{raw_data_path}/medications.csv")
    observations_pd = pd.read_csv(f"{raw_data_path}/observations.csv")
    patients_pd = pd.read_csv(f"{raw_data_path}/patients.csv")

    unique_patient_ids = patients_pd['Id'].unique()

    np.random.seed(random_seed)
    np.random.shuffle(unique_patient_ids) #note: in-place shuffle

    train_ids = unique_patient_ids[:round(training_percentage*len(unique_patient_ids))]
    test_ids = unique_patient_ids[round(training_percentage*len(unique_patient_ids)):]

    train_allergies = allergies_pd[allergies_pd['PATIENT'].isin(train_ids)]
    test_allergies = allergies_pd[allergies_pd['PATIENT'].isin(test_ids)]

    train_conditions = conditions_pd[conditions_pd['PATIENT'].isin(train_ids)]
    test_conditions = conditions_pd[conditions_pd['PATIENT'].isin(test_ids)]

    train_medications = medications_pd[medications_pd['PATIENT'].isin(train_ids)]
    test_medications = medications_pd[medications_pd['PATIENT'].isin(test_ids)]

    train_observations = observations_pd[observations_pd['PATIENT'].isin(train_ids)]
    test_observations = observations_pd[observations_pd['PATIENT'].isin(test_ids)]

    train_patients = patients_pd[patients_pd['Id'].isin(train_ids)]
    test_patients = patients_pd[patients_pd['Id'].isin(test_ids)]

    if save_as_csv:
        train_allergies.to_csv('data/train_allergies.csv', header=None, index=None, sep=',')
        test_allergies.to_csv('data/test_allergies.csv', header=None, index=None, sep=',')

        train_conditions.to_csv('data/train_conditions.csv', header=None, index=None, sep=',')
        test_conditions.to_csv('data/test_conditions.csv', header=None, index=None, sep=',')

        train_medications.to_csv('data/train_medications.csv', header=None, index=None, sep=',')
        test_medications.to_csv('data/test_medications.csv', header=None, index=None, sep=',')

        train_observations.to_csv('data/train_observations.csv', header=None, index=None, sep=',')
        test_observations.to_csv('data/test_observations.csv', header=None, index=None, sep=',')

        train_patients.to_csv('data/train_patients.csv', header=None, index=None, sep=',')
        test_patients.to_csv('data/test_patients.csv', header=None, index=None, sep=',')


if __name__ == '__main__':
    # process_conditions_data('testing_conditions.txt')
    make_train_test_split_data(raw_data_path='raw_data/', training_percentage=0.7, save_as_csv=True)