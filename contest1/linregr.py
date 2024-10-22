import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('train.csv').dropna()

columns_to_drop = ['ApplicationDate', 'MaritalStatus', 'HomeOwnershipStatus', 
                   'LoanPurpose', 'EmploymentStatus', 'EducationLevel']
train_df = train_df.drop(columns=columns_to_drop)

train_df = train_df[(train_df['RiskScore'] >= 0) & (train_df['RiskScore'] <= 99)]

X = train_df.drop(columns=['RiskScore'])
y = train_df['RiskScore'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

test_df = pd.read_csv('test.csv').dropna()
test_ids = test_df['ID']
test_df = test_df.drop(columns=columns_to_drop + ['ID'])
test_df = test_df.reindex(columns=X.columns, fill_value=0)
predictions = model.predict(test_df)
output_df = pd.DataFrame({'ID': test_ids, 'RiskScore': predictions})

output_df.to_csv('submission.csv', index=False)
