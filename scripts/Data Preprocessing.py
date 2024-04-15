#Data Preprocessing
# One-hot encode categorical feature "application type"
data = pd.get_dummies(data, columns=["application type"])

# Separate features and target
X = data.drop("resource allocation", axis=1)
y = data["resource allocation"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)