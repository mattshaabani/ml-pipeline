from src.data.ingestion import DataIngestion
from src.data.validation import DataValidator
from src.data.splitter import DataSplitter

# Load
ingestion = DataIngestion()
df = ingestion.load()
print(f"Loaded: {df.shape}")
print(df.head(3))

# Validate
validator = DataValidator()
report    = validator.validate(df)
print(f"Validation passed: {report['passed']}")
print(f"Warnings: {len(report['warnings'])}")
print(f"Target distribution: {report['stats']['target_distribution']}")

# Split
splitter = DataSplitter()
train, val, test = splitter.split(df)
print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
splitter.save_splits(train, val, test)
print("Splits saved to data/processed/")