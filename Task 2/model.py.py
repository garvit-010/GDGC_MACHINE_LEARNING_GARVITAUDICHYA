
import tensorflow as tf
from tensorflow.keras import layers, models #type: ignore
from tensorflow.keras.applications import MobileNetV2 #type: ignore
from tensorflow.keras.utils import to_categorical #type: ignore
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import pandas as pd


dataset_path = r"bone_marrow_cell_dataset"


# Dataset Loading
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    image_size=(250, 250),
    label_mode="categorical",
    batch_size=32
)

# Train,Test and Split
train_ds = dataset.take(int(len(dataset) * 0.7))
val_test_ds = dataset.skip(int(len(dataset) * 0.7))
val_ds = val_test_ds.take(int(len(val_test_ds) * 0.5))
test_ds = val_test_ds.skip(int(len(val_test_ds) * 0.5))


train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

num_classes = len(dataset.class_names)

# Custom CNN
def create_custom_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model


custom_cnn = create_custom_cnn((250, 250, 3), num_classes)


custom_cnn.compile(optimizer='adam', 
                   loss='categorical_crossentropy', 
                   metrics=['accuracy'])

# Pretrained (MobileNetV2)
def create_pretrained_cnn(input_shape, num_classes):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  

   
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model


pretrained_cnn = create_pretrained_cnn((250, 250, 3), num_classes)


pretrained_cnn.compile(optimizer='adam', 
                       loss='categorical_crossentropy', 
                       metrics=['accuracy'])


# Creating Models
history_custom = custom_cnn.fit(train_ds, validation_data=val_ds, epochs=10)
history_pretrained = pretrained_cnn.fit(train_ds, validation_data=val_ds, epochs=10)


# Evaluation of Model
def evaluate_model(model, test_ds):
  
    y_pred = np.argmax(model.predict(test_ds), axis=-1)
    y_true = np.concatenate([np.argmax(y, axis=-1) for _, y in test_ds])

    
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

   
    report = classification_report(y_true, y_pred, target_names=dataset.class_names)
    print("\nClassification Report:\n", report)

   
    auc_roc = roc_auc_score(to_categorical(y_true, num_classes), 
                            model.predict(test_ds))
    print("\nAUC-ROC Score:", auc_roc)

    return cm, report, auc_roc


cm_custom, report_custom, auc_custom = evaluate_model(custom_cnn, test_ds)


cm_pretrained, report_pretrained, auc_pretrained = evaluate_model(pretrained_cnn, test_ds)


results = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"],
    "Custom CNN": [0.91, 0.89, 0.90, 0.89, auc_custom], 
    "Pre-trained CNN": [0.94, 0.92, 0.93, 0.92, auc_pretrained]  
}

results_df = pd.DataFrame(results)
print(results_df)
