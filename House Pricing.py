import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("Housing Price Analysis - Improved Version")
print("=" * 60)

print("Loading data...")
data = pd.read_csv("Housing.csv")
print(f"Loaded {len(data)} records and {len(data.columns)} columns")

print("\nData shape:", data.shape)
print(" Columns:", list(data.columns))

print("\nMissing values:")
missing_data = data.isnull().sum()
if missing_data.sum() == 0:
    print("No missing values found")
else:
    print(missing_data[missing_data > 0])

print("\nFirst 5 rows:")
print(data.head())

print("\nData info:")
print(data.info())

print("\nDescriptive statistics:")
print(data.describe())

print("\nData visualization:")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Housing Price Analysis', fontsize=16, fontweight='bold')

axes[0, 0].hist(data['price'], bins=30, alpha=0.7, color='skyblue')
axes[0, 0].set_title('Price Distribution')
axes[0, 0].set_xlabel('Price')
axes[0, 0].set_ylabel('Frequency')

axes[0, 1].scatter(data['area'], data['price'], alpha=0.6, color='green')
axes[0, 1].set_title('Area vs Price')
axes[0, 1].set_xlabel('Area')
axes[0, 1].set_ylabel('Price')

bedroom_price = data.groupby('bedrooms')['price'].mean()
axes[0, 2].bar(bedroom_price.index, bedroom_price.values, color='orange')
axes[0, 2].set_title('Average Price by Bedrooms')
axes[0, 2].set_xlabel('Bedrooms')
axes[0, 2].set_ylabel('Average Price')

bathroom_price = data.groupby('bathrooms')['price'].mean()
axes[1, 0].bar(bathroom_price.index, bathroom_price.values, color='red')
axes[1, 0].set_title('Average Price by Bathrooms')
axes[1, 0].set_xlabel('Bathrooms')
axes[1, 0].set_ylabel('Average Price')

stories_price = data.groupby('stories')['price'].mean()
axes[1, 1].bar(stories_price.index, stories_price.values, color='purple')
axes[1, 1].set_title('Average Price by Stories')
axes[1, 1].set_xlabel('Stories')
axes[1, 1].set_ylabel('Average Price')

ac_price = data.groupby('airconditioning')['price'].mean()
axes[1, 2].bar(ac_price.index, ac_price.values, color='brown')
axes[1, 2].set_title('Average Price by Air Conditioning')
axes[1, 2].set_xlabel('Air Conditioning')
axes[1, 2].set_ylabel('Average Price')

plt.tight_layout()
plt.show()

print("\nEncoding categorical data...")
categorical_columns = data.select_dtypes(include=['object']).columns
print(f"Categorical columns: {list(categorical_columns)}")

data_encoded = data.copy()
label_encoders = {}

for column in categorical_columns:
    le = LabelEncoder()
    data_encoded[column] = le.fit_transform(data_encoded[column])
    label_encoders[column] = le
    print(f"Encoded {column}")

print("\nAnalyzing correlations...")
correlation_matrix = data_encoded.corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
           center=0, square=True, linewidths=0.5)
plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

price_correlations = correlation_matrix['price'].sort_values(ascending=False)
print("\nCorrelations with price:")
print(price_correlations)

print("\nPreparing data for training...")

X = data_encoded.drop('price', axis=1)
y = data_encoded['price']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training data: {X_train.shape}")
print(f"Testing data: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data scaled successfully")

print("\nTraining Linear Regression model...")

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')

print(f"   MAE: {mae:,.0f}")
print(f"   RMSE: {rmse:,.0f}")
print(f"   R²: {r2:.4f}")
print(f"   CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

print(f"\nModel coefficients:")
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print(coefficients.to_string(index=False, float_format='%.2f'))

plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Linear Regression: Predictions vs Actual')
plt.tight_layout()
plt.show()

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(model.coef_)
}).sort_values('Importance', ascending=False)

print(f"\nFeature importance (absolute coefficients):")
print(feature_importance.to_string(index=False, float_format='%.2f'))

plt.figure(figsize=(10, 8))
bars = plt.barh(feature_importance['Feature'], feature_importance['Importance'], 
               color='lightblue', alpha=0.7)
plt.title('Feature Importance in Linear Regression', fontsize=14, fontweight='bold')
plt.xlabel('Absolute Coefficient Value')
plt.ylabel('Feature')

for bar, importance in zip(bars, feature_importance['Importance']):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{importance:.2f}', va='center')

plt.tight_layout()
plt.show()

print(f"\nPredicting new sample...")
sample_house = pd.DataFrame({
    'area': [5000],
    'bedrooms': [3],
    'bathrooms': [2],
    'stories': [2],
    'mainroad': ['yes'],
    'guestroom': ['no'],
    'basement': ['yes'],
    'hotwaterheating': ['no'],
    'airconditioning': ['yes'],
    'parking': [2],
    'prefarea': ['yes'],
    'furnishingstatus': ['furnished']
})

print("Sample house features:")
print(sample_house.to_string(index=False))

sample_encoded = sample_house.copy()
for column in label_encoders:
    if column in sample_encoded.columns:
        sample_encoded[column] = label_encoders[column].transform(sample_encoded[column])

sample_scaled = scaler.transform(sample_encoded)

prediction = model.predict(sample_scaled)[0]

print(f"\nPredicted price: {prediction:,.0f}")

print(f"\nModel Summary:")
print(f"Model: Linear Regression")
print(f"Training accuracy (R²): {model.score(X_train_scaled, y_train):.4f}")
print(f"Testing accuracy (R²): {r2:.4f}")
print(f"Cross-validation accuracy (R²): {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
print(f"Mean Absolute Error: {mae:,.0f}")
print(f"Root Mean Square Error: {rmse:,.0f}")

print(f"\nAnalysis completed successfully!")
print(f"The Linear Regression model achieved a reasonable accuracy of {r2:.2%}")
print(f"The model can be used for predicting house prices based on the given features")
