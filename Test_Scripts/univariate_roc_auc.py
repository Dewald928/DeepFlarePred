# import the DecisionTree Algorithm and evaluation score.
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

# list of the resulting scores.
roc_values = []

# loop over all features and calculate the score.
for feature in x_train.columns:
    clf = DecisionTreeClassifier()
    clf.fit(x_train[feature].to_frame(), y_train)
    y_scored = clf.predict_proba(x_test[feature].to_frame())
    roc_values.append(roc_auc_score(y_test, y_scored[:, 1]))

# create a Pandas Series for visualisation.
roc_values = pd.Series(roc_values)
roc_values.index = X_train.columns

# show the results.
print(roc_values.sort_values(ascending=False))