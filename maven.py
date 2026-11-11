import pandas as pd

# Load datasets
df = pd.read_csv("Bank_Churn_Data_Dictionary.csv")
DF = pd.read_excel("Bank_Churn_Messy.xlsx")
DataFrame = pd.read_csv("Bank_Churn.csv")

#Remove missing values and inspect datasets
for data in[df, DF, DataFrame]:
    data.dropna(inplace = True)
    print(data.isna().sum())
    data.drop_duplicates(inplace = True)
    print(data.info())
    