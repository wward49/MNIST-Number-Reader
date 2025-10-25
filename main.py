#MNIST database of numbers!

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the digits dataset (1797 images of 8x8 pixel digits)
digits = load_digits()

# Show the first digit
plt.gray()
plt.matshow(digits.images[0])
plt.title(f"Label: {digits.target[0]}")
plt.show()

# Flatten the image data for the model
X = digits.data  # shape: (1797, 64)....digits.data[0] is a single array from left to right, then top to bottom of the first image
y = digits.target # digits.target[0] is the number digits.data[0] is supposed to represent.

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
