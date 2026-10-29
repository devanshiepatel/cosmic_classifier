import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df=pd.read_csv('star_classification.csv')

#dropping the telescope meta data and tracking ids
col_to_drop=['obj_ID','run_ID','rerun_ID','cam_col','field_ID','spec_obj_ID','plate','MJD','fiber_ID']
df_cleaned=df.drop(columns=col_to_drop)

#separating the features and target variable
X=df_cleaned.drop(columns=['class'])
y=df_cleaned['class']

#splitting the data into training and testing sets (80:20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

#training the baselise Random Forest Classifier
model=RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

#evaluation
y_pred=model.predict(X_test)
print('\n**********Classification Report**********\n')
print(classification_report(y_test, y_pred))