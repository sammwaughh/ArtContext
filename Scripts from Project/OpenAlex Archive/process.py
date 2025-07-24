import pandas as pd


def list_columns(filename="vermeer_works.xlsx"):
    df = pd.read_excel(filename, engine="openpyxl")
    print("Columns in file:", df.columns.tolist())


if __name__ == "__main__":
    list_columns()
