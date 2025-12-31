import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
df = pd.read_csv("sales_pricing_data.csv")
df
# Check missing values
df.isnull().sum()
# Remove missing values if any
df = df.dropna()
X = df[['price', 'marketing_spend']]
X = sm.add_constant(X)

y = df['units_sold']

model = sm.OLS(y, X).fit()
model.summary()
# Price Elasticity Calculation
df['price_elasticity'] = (model.params['price'] * df['price']) / df['units_sold']
df[['price', 'units_sold', 'price_elasticity']]
# Revenue Calculation
df['revenue'] = df['price'] * df['units_sold']
df[['price', 'units_sold', 'revenue']]
#Revenue vs Price Visualization
plt.figure(figsize=(8,5))
plt.plot(df['price'], df['revenue'], marker='o')
plt.xlabel("Price")
plt.ylabel("Revenue")
plt.title("Revenue vs Price")
plt.grid(True)
plt.show()
#Identify Optimal Price
optimal_row = df.loc[df['revenue'].idxmax()]
optimal_row
#Final Business Recommendation
print(
    f"Optimal price is approximately ₹{optimal_row['price']}, "
    "where revenue is maximized. Prices beyond this reduce revenue "
    "due to elastic demand."
)