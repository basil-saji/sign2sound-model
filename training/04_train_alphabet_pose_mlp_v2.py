from sklearn.utils.class_weight import compute_class_weight

# ================= CLASS WEIGHTING =================
# Calculate weights to balance the loss function
unique_classes = np.unique(y_train)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=unique_classes,
    y=y_train
)

# Explicitly map class index to weight
class_weight_dict = {
    int(cls): float(w) for cls, w in zip(unique_classes, class_weights)
}

print("Class Weights calculated:")
print(f"  Min weight: {min(class_weights):.4f}")
print(f"  Max weight: {max(class_weights):.4f}")

# ================= MODEL DEFINITION =================
def build_pose_mlp(input_dim, num_classes):
    inputs = keras.Input(shape=(input_dim,))

    # Layer 1: Wide enough to capture geometry combinations
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dropout(0.3)(x)

    # Layer 2: Bottleneck for stable features
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="alphabet_pose_mlp")
    return model

model = build_pose_mlp(input_dim=X.shape[1], num_classes=len(LETTERS))

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

# ================= TRAIN =================
print("Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)
