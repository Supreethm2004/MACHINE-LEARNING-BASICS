import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data: house size (in square feet) vs price (in $1000s)
x = np.array([600, 800, 1000, 1200, 1400, 1600, 1800]).reshape(-1, 1)
y = np.array([150, 180, 200, 220, 250, 270, 300])

# Create and train the model
model = LinearRegression()
model.fit(x, y)

# Make a prediction
predicted_price = model.predict([[1500]])
print(f"Predicted price for 1500 sqft: ${predicted_price[0] * 1000:.2f}")

# Plot the data
plt.scatter(x, y, color='blue', label='Actual Prices')
plt.plot(x, model.predict(x), color='red', linewidth=2, label='Regression Line')
plt.xlabel('House Size (sqft)')
plt.ylabel('Price ($1000s)')
plt.legend()
plt.show()
