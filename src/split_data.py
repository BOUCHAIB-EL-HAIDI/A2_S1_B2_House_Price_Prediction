import pandas as pd


# Load the original dataset
df = pd.read_csv("data/raw/House_Prices.csv")

print("Original shape:", df.shape)


# Separate the original Kaggle train/test rows
train_df = df.iloc[:1460].copy()
test_df = df.iloc[1460:].copy()


# Remove SalePrice from the test data
test_df = test_df.drop(columns=["SalePrice"])


# Save the separated datasets
train_df.to_csv(
    "data/raw/train.csv",
    index=False
)

test_df.to_csv(
    "data/raw/test.csv",
    index=False
)


# Display the results
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)