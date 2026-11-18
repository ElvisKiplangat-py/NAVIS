import pandas as pd

# Load datasets
df = pd.read_csv("Bank_Churn_Data_Dictionary.csv")
DF = pd.read_excel("Bank_Churn_Messy.xlsx")

#Remove missing values and inspect datasets
for data in[df, DF]:
    data.dropna(inplace = True)
    print(data.isna().sum())
    data.drop_duplicates(inplace = True)
    print(data.duplicated().sum())
    print(data.info())
    