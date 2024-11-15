import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

sm = pd.read_csv(r"C:\Users\Downloads\SuperMarket.csv")

# Retrieve info about the dataset
sm.info()
print(sm.head())
print(sm.describe())
#Convert Date column to datetime
sm['Date']=pd.to_datetime(sm['Date'])
sm['Time']=pd.to_datetime(sm['Time'], format='%I:%M:%S %p').dt.time

#Group by 'date' and find maximum sales over each date
total_sales = sm.groupby('Date')['Sales'].sum()

#Plot Sales over time

plt.figure(figsize=(12,6))
plt.plot(total_sales.index, total_sales.values, marker="o", c='darkblue')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.title('Total Sales over Time')
plt.show()

# Plot Sales per product
plt.figure(figsize=(12,6))
plt.bar( sm['Product line'],sm['Sales'], color='green')
plt.xlabel('Sales')
plt.ylabel('Product line')
plt.title('Sales per Product Line')
plt.show()

#Total Sales by city
city_sales = sm.groupby('City')['Sales'].sum()

plt.figure(figsize=(12,6))
plt.barh(city_sales.index, city_sales.values, color='red')
plt.xlabel('City')
plt.ylabel('Total Sales')
plt.title('Total Sales by City')
plt.show()

gross_income_by_product_line = sm.groupby('Product line')['gross income'].sum()
gross_margin_by_product_line = sm.groupby('Product line')['gross margin percentage'].mean()

fig, ax1 = plt.subplots()

# Plot on the first axis (Gross Income)

ax1.bar(gross_income_by_product_line.index,gross_income_by_product_line.values, color='b', label="Gross Income")
ax1.set_label('Product Line')
ax1.set_label('Total Gross Income')

# Create a second axis for Gross Margin
ax2 = ax1.twinx()
ax2.plot(gross_margin_by_product_line.index, gross_margin_by_product_line.values, color='g', label='Gross Margin')
ax2.set_ylabel('Gross Margin (%)')
plt.title('Gross Income and Gross Margin Per Product Line')
fig.tight_layout()  # Adjust layout to fit both labels
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#linear regression - Sales per Unit price
#Selecting the columns for the analysis
x = sm['Unit price'].values.reshape(-1,1) #independent variable
y = sm['Sales'].values.reshape(-1,1)# dependent variable

#Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

#Creating and Training the LinearRegression Model
model = LinearRegression()
model.fit(x_train, y_train)

#Making Predictions on the test set
y_pred = model.predict(x_test)

# Displaying the model coefficients
slope = model.coef_[0][0]
intercept = model.intercept_[0]

#Plotting the Results
plt.figure(figsize=(8,5))
plt.scatter(x,y,color='blue',alpha=0.5,label="Actual Data")
plt.plot(x_test, y_pred,color ='red', linewidth=2,label='Regression Line')
plt.xlabel("Unit Price")
plt.ylabel("Sales")
plt.title("Linear Regression: Unit Price vs Sales")
plt.legend()
plt.show()
slope, intercept
