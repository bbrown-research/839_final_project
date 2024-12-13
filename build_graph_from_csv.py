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
    observations: pd.DataFrame
    allergies: pd.DataFrame

    OBJECT_TYPES = [
        "patients",
        "conditions",
        "medications",
        "observations",
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
        "observations": ["DATE", "PATIENT", "CODE"],
        "allergies": ["START", "PATIENT", "CODE"]
    }


def make_graph(get_filename=Data.get_test):
    data = Data.init_with_data(get_filename=get_filename)

    graph_ = networkx.Graph()

    unique_patient_ids = data.patients["Id"].unique()

    for pt in unique_patient_ids:
        graph_.add_node(pt)

    for attr in Data.OBJECT_TYPES:
        if attr == "patients":
            continue

        for _index, row in getattr(data, attr).iterrows():
            hash_keys = Data.HASH_KEYS[attr]
            hash = Data.hash_series(row, keys=hash_keys)
            graph_.add_node(hash)
            graph_.add_edge(row["PATIENT"], hash)

    return graph_


if __name__ == "__main__":
    graph = make_graph(get_filename=Data.get_all)
    
    print()
    print("Should be 0")
    print(edit_dist(graph, graph))
    print("Running spectral distance")
    print(spectral_dist(graph, graph))
    print("Done!")
