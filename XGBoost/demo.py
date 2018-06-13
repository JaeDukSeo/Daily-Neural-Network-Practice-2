from sklearn import datasets  
from sklearn.model_selection import train_test_split  
from sklearn.model_selection import cross_val_score

X, y = datasets.make_classification(n_samples=10000, n_features=20,  
                                    n_informative=2, n_redundant=10,
                                    random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,  
                                                    random_state=42)

from xgboost.sklearn import XGBClassifier  
from xgboost.sklearn import XGBRegressor

xclas = XGBClassifier()  # and for classifier  
xclas.fit(X_train, y_train)  
xclas.predict(X_test)  


print(cross_val_score(xclas, X_train, y_train)  )