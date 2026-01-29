# ================= LOAD & PROCESS DATA =================
X_list = []
y_list = []

print("Loading data...")
for idx, letter in enumerate(LETTERS):
    npy_path = os.path.join(DATA_DIR, f"{letter}.npy")

    if not os.path.exists(npy_path):
        print(f"Missing file: {npy_path}")
        continue

    # Load raw landmarks (N, 63)
    raw_data = np.load(npy_path)

    # Handle shape variations just in case
    if raw_data.ndim == 3: # (N, 1, 63) or (N, 30, 63)
        raw_data = raw_data[:, 0, :]

    print(f"  {letter}: {raw_data.shape[0]} samples")

    # Apply feature engineering to every sample
    features = np.array([featurize_pose(x) for x in raw_data])

    X_list.append(features)
    y_list.append(np.full(len(features), idx))

# Concatenate all
X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)

print(f"\n Total Data Shape: {X.shape}")
print(f"Features per sample: {X.shape[1]} (should be 81)")

# Train/Val Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"   Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
