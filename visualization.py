import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Bank_Churn.csv')
plt.xlabel('Credit Score')
plt.ylabel('Estimated Salary')
plt.title('Credit Score vs Estimated Salary')
plt.hist(df['CreditScore'], bins=40)
plt.grid(True)
plt.show()

