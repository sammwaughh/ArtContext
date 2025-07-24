import pandas as pd

df = pd.read_excel("paintings.xlsx")
# Count cells in "Sentence 1" that are not NaN and not empty (after stripping whitespace).
count = df["Sentence 1"].dropna().apply(lambda x: str(x).strip() != "").sum()
print("Number of non-empty cells in 'Sentence 1':", count)
