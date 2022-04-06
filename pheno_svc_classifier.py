import pandas as pd
import pickle

class pheno_classifier():
    
    def __init__(self):
        
        path = "SVC_classifier.pkl"
        with open(path, "rb") as model_file:
            clf = pickle.load(model_file)
            
        self.clf = clf
        
    def predict(self, X):
        prediction_prob = self.clf.predict_proba(X)
        return prediction_prob

X_test = pd.read_csv("test_df.csv")
print("Shape of df :{}".format(X_test.shape))

cls = pheno_classifier()
y_prob = cls.predict(X_test)

print("Prediction Probability are : {}".format(y_prob))




