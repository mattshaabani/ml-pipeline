from src.data.ingestion import DataIngestion
from src.data.validation import DataValidator
from src.data.splitter import DataSplitter
from src.features.feature_store import FeatureStore

# Load and split data
ingestion = DataIngestion()
df        = ingestion.load()

validator = DataValidator()
report    = validator.validate(df)
print(f"Validation: {report['passed']}")

splitter = DataSplitter()
train, val, test = splitter.split(df)

# Materialize features
store = FeatureStore()
store.materialize(train, val, test)

# Load back
X_train, X_val, X_test, y_train, y_val, y_test = store.get_training_features()

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape:   {X_val.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train distribution: {y_train.mean():.3f} positive rate")

# Feature names
names = store.get_feature_names()
print(f"Total features: {len(names)}")
print(f"First 10 features: {names[:10]}")
print(f"Last 5 features:   {names[-5:]}")

# Test online features
entity = {
    "age": 35,
    "workclass": "Private",
    "fnlwgt": 200000,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Married-civ-spouse",
    "occupation": "Tech-support",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States",
    "income": "<=50K",
}
online_features = store.get_online_features(entity)
print(f"Online feature vector shape: {online_features.shape}")
print("Feature pipeline working end-to-end!")