from src.training.trainer import ModelTrainer

trainer = ModelTrainer()

# Use only 5 trials for quick verification
# We'll use 20+ in the real Airflow run
results = trainer.train_all(n_tuning_trials=5)

print("\nTraining complete!")
print(f"Models trained: {list(results.keys())}")