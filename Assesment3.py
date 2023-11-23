#%%
# import packages
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd
import os, datetime
import numpy as np

# import dataset
df = pd.read_csv('train.csv')
# Set the 'id' column as the index
df.set_index('id', inplace=True)
# Drop column that has too much missing data
df.drop('days_since_prev_campaign_contact', axis=1, inplace=True)
# Drop row with missing value
df.dropna(inplace=True)
# %%
#Label Encoding
label_encoder = LabelEncoder()
df['job_type'] = label_encoder.fit_transform(df['job_type'])
df['marital'] = label_encoder.fit_transform(df['marital'])
df['education'] = label_encoder.fit_transform(df['education'])
df['default'] = label_encoder.fit_transform(df['default'])
df['housing_loan'] = label_encoder.fit_transform(df['housing_loan'])
df['personal_loan'] = label_encoder.fit_transform(df['personal_loan'])
df['communication_type'] = label_encoder.fit_transform(df['communication_type'])
df['month'] = label_encoder.fit_transform(df['month'])
df['prev_campaign_outcome'] = label_encoder.fit_transform(df['prev_campaign_outcome'])
# %%
# Split the data to train and label
predictors = df.drop(['term_deposit_subscribed'], axis=1).values
target = df['term_deposit_subscribed'].values
n_cols = predictors.shape[1]

# Use RandomOverSampler to balance the classes in the training set
oversampler = RandomOverSampler(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(predictors, target)

# Split the resampled data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Apply MinMaxScaler to features
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# %%
# Build the model
model = Sequential()

# Input layer
model.add(Dense(256, activation='relu', input_shape=(n_cols,)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# Hidden layer 1
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# Hidden layer 2
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# Output layer
model.add(Dense(2, activation='softmax'))
# %%
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Create a tensorboard callback project
PATH = os.getcwd()
logpath = os.path.join(PATH, "tensorboard_log", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = tf.keras.callbacks.TensorBoard(logpath)
# %%
# Model fit
batch_size = 32
epochs = 50  # You can adjust the number of epochs
history = model.fit(X_train_scaled, y_train,
                    validation_data=(X_val_scaled, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[tb,early_stopping])

tf.keras.utils.plot_model(model)
# %%
# loss graph
fig = plt.figure()
plt.plot(history.history['loss'], color='teal', label='loss')
plt.plot(history.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend()
plt.show()
#%%
# accuracy graph
fig = plt.figure()
plt.plot(history.history['accuracy'], color='teal', label='accuracy')
plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend()
plt.show()
#%%
# Model testing and deployment
# Evaluate the model on the test set
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
# %%
# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred_classes))
# %%
# Evaluate the model on the test set
evaluation = model.evaluate(X_test_scaled, y_test)

# Print the evaluation results
print("Loss:", evaluation[0])
print("Accuracy:", evaluation[1])
# %%
# Model save
model.save(os.path.join('outcome_classify.h5'))
# %%
