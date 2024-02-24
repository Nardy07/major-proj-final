import pandas as pd
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from keras.models import load_model

from sklearn.metrics import accuracy_score # for calculating accuracy of model
from sklearn.metrics import classification_report

X_new = pd.read_csv('X_val_nulti_with_ping.csv')
y_new = pd.read_csv('y_val_multi_with_ping.csv')

y1 = y_new
le = LabelEncoder()
y_new = le.fit_transform(y_new)

# Print the mapping of labels to digits
print("Label to digit mapping:")
for i, label in enumerate(le.classes_):
    print(f"{label}: {i}")

X_new_reshaped = X_new.values.reshape((X_new.shape[0], 1, X_new.shape[1]))

model = load_model("micmultiBB1.keras")

predictions = model.predict(X_new_reshaped)

#convert to categorical
y_new = to_categorical(y_new,3)

comparison = np.argmax(predictions,axis=1) == np.argmax(y_new,axis=1)

accuracy = accuracy_score(np.argmax(y_new,axis=1), np.argmax(predictions,axis=1))
print("accuracy on the new dataset: {:.2f}%".format(accuracy*100))


# Initialize a LabelEncoder
#le = LabelEncoder()
y_test_labels1 = y_new.argmax(axis=1)
# Fit and transform the labels in y_true
y_true_encoded = le.fit_transform(y_test_labels1)

# Convert the predicted probabilities to labels
y_pred_labels = np.argmax(predictions, axis=1)


# Check the confusion matrix
from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(y_true_encoded, y_pred_labels)

print("Confusion matrix:")
print(confusion_mat)

# Convert predictions to labels
#predicted_labels = [0 if prediction < 0.5 else 1 for prediction in predictions]



#yo talako chai label encoded gareko step bala ma ho
# Decode the predicted labels back to string values
#predicted_labels = le.inverse_transform(predictions)
#print(predicted_labels)
logr_bin_df = pd.DataFrame({'Actual': y_true_encoded, 'Predicted': y_pred_labels})
print(logr_bin_df)

#report = classification_report(y_true_encoded,y_pred_labels)
#print(report)