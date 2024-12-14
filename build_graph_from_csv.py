import pandas as pd
from os import path
from dataclasses import dataclass
import networkx
from eval import edit_dist, spectral_dist


@dataclass
class Data:
    patients: pd.DataFrame
    conditions: pd.DataFrame
    medications: pd.DataFrame
    # observations: pd.DataFrame
    allergies: pd.DataFrame

    OBJECT_TYPES = [
        "patients",
        "conditions",
        "medications",
        # "observations",
        "allergies",
    ]

    @staticmethod
    def get_all(file: str) -> str:
        return path.join("raw_data", f"{file}.csv")

    @staticmethod
    def get_train(file: str) -> str:
        return path.join("data", f"train_{file}.csv")

    @staticmethod
    def get_test(file: str) -> str:
        return path.join("data", f"test_{file}.csv")

    @classmethod
    def init_with_data(cls, get_filename=get_test):
        pd_files = {key: pd.read_csv(get_filename(key)) for key in cls.OBJECT_TYPES}
        return cls(**pd_files)

    @staticmethod
    def hash_series(series: pd.Series, keys: list[str], sep="|") -> str:
        return sep.join(str(series[k]) for k in keys)

    @staticmethod
    def hash_series_agg_date_w_exclude(series: pd.Series, keys: list[str], sep="|", date_agg='m', excludes=['PATIENT']) -> str:
        sep_values = []
        for k in keys:
            if k not in excludes:
                sep_val = series[k]
                if k in ['START', 'DATE']:
                    y_m_d = series[k].split("T")[0]
                    if date_agg == 'y':
                        sep_val = y_m_d.split("-")[0]
                    elif date_agg == 'm':
                        sep_val = "-".join(y_m_d.split("-")[:2])
                    else:
                        sep_val = y_m_d
                sep_values.append(str(sep_val))
        return sep.join(sep_values)

    @staticmethod
    def hash_conditions(condition_row: pd.Series) -> str:
        return Data.hash_series(
            condition_row, keys=["START", "PATIENT", "CODE", "DESCRIPTION"]
        )
    
    @staticmethod
    def hash_medications(medication_row: pd.Series) -> str:
        return Data.hash_series(
            medication_row, keys=[]
        )
    
    HASH_KEYS = {
        "conditions": ["START", "PATIENT", "CODE"],
        "medications": ["START", "PATIENT", "CODE"],
        # "observations": ["DATE", "PATIENT", "CODE"],
        "allergies": ["START", "PATIENT", "CODE"]
    }


def make_graph(get_filename=Data.get_test, hash_type=None, replace_id_w_name=False):
    data = Data.init_with_data(get_filename=get_filename)

    graph_ = networkx.Graph()

    unique_patient_ids = data.patients["Id"].unique()
    patient_id_to_name_map = {unique_patient_ids[i]:[" ".join(x) for x in data.patients[['FIRST', 'LAST']].values][i] for i in range(len(unique_patient_ids))}


    for pt in unique_patient_ids:
        if replace_id_w_name:
            graph_.add_node(patient_id_to_name_map[pt])
        else:
            graph_.add_node(pt)

    data_store = {}
    for attr in Data.OBJECT_TYPES:
        if attr == "patients":
            continue

        if replace_id_w_name:
            data_pd = getattr(data, attr)
            data_pd['PATIENT'] = data_pd['PATIENT'].apply(lambda x: patient_id_to_name_map[x])

        data_store[attr] = data_pd

        for _index, row in getattr(data, attr).iterrows():
            hash_keys = Data.HASH_KEYS[attr]
            if hash_type is None:
                hash = Data.hash_series(row, keys=hash_keys)
            elif hash_type == 'np_m':
                hash = Data.hash_series_agg_date_w_exclude(row, keys=hash_keys, date_agg='m', excludes=['PATIENT'])
            elif hash_type == 'np_nt':
                hash = Data.hash_series_agg_date_w_exclude(row, keys=hash_keys, date_agg='m', excludes=['PATIENT', 'START', 'DATE'])
            else:
                raise NotImplementedError(f"Hash type {hash_type} is not implemented.")
            graph_.add_node(hash)
            graph_.add_edge(row["PATIENT"], hash)

    return graph_, data_store, patient_id_to_name_map


if __name__ == "__main__":
    graph = make_graph(get_filename=Data.get_all)
    
    print()
    print("Should be 0")
    print(edit_dist(graph, graph))
    print("Running spectral distance")
    print(spectral_dist(graph, graph))
    print("Done!")
