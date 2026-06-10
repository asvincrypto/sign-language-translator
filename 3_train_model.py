import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print('Loading dataset...')

with open('./model/sign_model_data.pkl', 'rb') as f:
    dataset = pickle.load(f)

X = np.array(dataset['data'])
y = np.array(dataset['labels'])

print(f'Total samples: {len(X)}')
print(f'Letters: {sorted(set(y))}')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f'\nTraining on {len(X_train)} samples...')

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc * 100:.2f}%')

with open('./model/sign_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print('\nModel saved to ./model/sign_model.pkl')
print('Ready for real-time detection!')