
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Prepare the data for linear regression
# Convert the month into a numerical format (number of months since the start)
demat_data['Months_Since_Start'] = np.arange(len(demat_data))

# Features and target
X = demat_data[['Months_Since_Start']]
y = demat_data['Account_Count']

# Train a Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict for January 2025, which is 61 months since the start (January 2019)
future_month = 61
predicted_account_count = model.predict([[future_month]])

print(f"Predicted DEMAT Account Count for January 2025: {int(predicted_account_count[0])}")

# Plot the predictions along with the original data
plt.plot(demat_data['Months_Since_Start'], demat_data['Account_Count'], label='Actual Data', marker='o')
plt.scatter(future_month, predicted_account_count, color='red', label='Prediction for Jan 2025')
plt.xlabel('Months Since Start (Jan 2019)')
plt.ylabel('DEMAT Account Count')
plt.legend()
plt.title('Linear Regression: DEMAT Account Prediction')
plt.show()
