#!/usr/bin/python3
import pandas
from pandas import DataFrame
from exception import csvException

class CsvManager:
    def __init__(self, path):
        self.path = path

    def read_csv(self, path):
        # Lecture du fichier
        try:
            return pandas.read_csv(path)
        except FileNotFoundError:
            print(f"File not found : {path}")

    def head_csv(self, file):
        # Lecture de la tête du fichier
        try:
            return file.head()
        except FileNotFoundError:
            print(f"File not found : {file}")

    def drop_csv(self, file: DataFrame, column=None):
        # Suppression d'une colonne du fichier
        try:
            return file.drop(columns=[column])
        except Exception:
            raise csvException.csvException(f"Column : '{column}' not found in the file")
