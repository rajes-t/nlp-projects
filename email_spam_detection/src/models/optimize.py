import pandas as pd
import joblib
from collections import Counter
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
 
from src.models.build_model import build_model
 
 
df = pd.read_csv("data/processed/cleaned_emails.csv")
df["clean_email"] = df["clean_email"].fillna("")
 
# Keep all ham, but only 25 spam -> a deliberately imbalanced set
ham = df[df["label"] == "ham"]
spam = df[df["label"] == "spam"].head(25)
imbalanced = pd.concat([ham, spam])
 
print("Class counts:", Counter(imbalanced["label"]))
