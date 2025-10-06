import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold


fake_df = pd.read_csv("archive\BuzzFeed_fake_news_content.csv")
real_df = pd.read_csv("archive\BuzzFeed_real_news_content.csv")

# fake_df = pd.read_csv("archive/PolitiFact_fake_news_content.csv")
# real_df = pd.read_csv("archive/PolitiFact_real_news_content.csv")

# fake1_df = pd.read_csv("archive\PolitiFact_fake_news_content.csv")
# real1_df = pd.read_csv("archive\PolitiFact_real_news_content.csv")


#  replace the empty authors with unknown for better eff
fake_df['authors'] = fake_df['authors'].fillna("unknown")
fake_df['source'] = fake_df['source'].fillna("unknown")
fake_df['canonical_link'] = fake_df['canonical_link'].fillna("unknown")

real_df['authors'] = real_df['authors'].fillna("unknown")
real_df['source'] = real_df['source'].fillna("unknown")
real_df['canonical_link'] = real_df['canonical_link'].fillna("unknown")


# Add labels
# 1 = fake, 0 = real
fake_df['label'] = 1
real_df['label'] = 0
# fake1_df['label'] = 1
# real1_df['label'] = 0


# Combine into one DataFrame
df = pd.concat([fake_df, real_df], axis=0).reset_index(drop=True)


#  replace the empty authors with unknown for better eff after concat

df['movies'] = df['movies'].fillna("unknown")
df['publish_date'] = df['publish_date'].fillna("unknown")


# mean leangth for better train 
avg_len = df['text'].str.split().apply(len).mean()
print("\nAverage text length:", avg_len)

X = df['text']
y = df['label']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("\nTrain size:", X_train.shape)
print("Validation size:", X_val.shape)


# Create vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1,2))

# Fit only on training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform validation and test data using same vocabulary
X_val_tfidf = tfidf.transform(X_val)
# X_test_tfidf = tfidf.transform(X_test)   # if you have test set

#  selection and model fitting

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_tfidf, y_train)


# # Step 3: Predictions
# y_pred = log_reg.predict(X_val_tfidf)

# step 3: cross valadition

# Stratified K-Fold (keeps class balance in each fold)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Vectorize on the full dataset for cross-validation
X_tfidf = tfidf.fit_transform(X)

# Cross-validation (Accuracy as example)
scores = cross_val_score(log_reg, X_tfidf, y, cv=cv, scoring='accuracy')


# step 4 : evalution   "while testing all data"

print("Cross-validation scores:", scores)
print("Mean Accuracy:", scores.mean())
print("Std Deviation:", scores.std())


# # Step 4: Evaluation "while testing predection"
# print("\n--- Classification Report ---")
# print(classification_report(y_val, y_pred))

# print("\n--- Confusion Matrix ---")
# print(confusion_matrix(y_val, y_pred))
