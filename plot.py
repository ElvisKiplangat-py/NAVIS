import matplotlib.pyplot as plt

ages = ['18-25', '26-35', '36-45', '46-55', '56+']
customers = [120, 250, 180, 100, 50]

bars = plt.bar(ages, customers, color='skyblue', fontsize=12)

plt.bar_label(bars, color='green', label_type='center')

plt.xlabel('Age Groups')
plt.ylabel('Number of Customers')
plt.title('Customers by Age Group')

plt.show()