import torch
from tests.utils.data_utils import get_adult_loaders

print("--- Running Data Loader Test ---")

try:
    # Test the standard data loading
    (train_loader, val_loader, test_loader), num_features = get_adult_loaders(
        data_path='./data', batch_size=256
    )

    print(f"Test successful!")
    print(f"Number of features detected: {num_features}")
    print(f"Train loader has {len(train_loader)} batches.")

    # Check a batch
    X_batch, y_batch = next(iter(train_loader))
    print(f"A single batch of data has shape: {X_batch.shape}")
    print(f"A single batch of labels has shape: {y_batch.shape}")

    assert X_batch.shape[1] == num_features, "Feature dimension mismatch!"
    assert len(y_batch.shape) == 1, "Labels should be a 1D tensor!"

    print("\n--- All tests passed! ---")

except Exception as e:
    print(f"\n--- TEST FAILED ---")
    print(f"An error occurred: {e}")
    import traceback

    traceback.print_exc()