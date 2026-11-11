import pandas as pd
import matplotlib.pyplot as plt
#visit https://matplotlib.org/stable/api/pyplot_summary.html for more pyplot functions
df = pd.read_csv('Bank_Churn.csv')
plt.xlabel('Credit Score')
plt.ylabel('Estimated Salary')
plt.title('Credit Score vs Estimated Salary')
plt.hist(df['CreditScore'], bins=40, alpha = 1)
plt.grid(True)
plt.show()

