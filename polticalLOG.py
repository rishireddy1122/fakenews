import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold

fake1_df = pd.read_csv("archive/PolitiFact_fake_news_content.csv")
real1_df = pd.read_csv("archive/PolitiFact_real_news_content.csv")








#  replace the empty authors with unknown for better eff
fake1_df['authors'] = fake1_df['authors'].fillna("unknown")
fake1_df['source'] = fake1_df['source'].fillna("unknown")
fake1_df['canonical_link'] = fake1_df['canonical_link'].fillna("unknown")

real1_df['authors'] = real1_df['authors'].fillna("unknown")
real1_df['source'] = real1_df['source'].fillna("unknown")
real1_df['canonical_link'] = real1_df['canonical_link'].fillna("unknown")






















fake1_df['label'] = 1
real1_df['label'] = 0

# Combine into one DataFrame
df = pd.concat([fake1_df, real1_df], axis=0).reset_index(drop=True)




df['movies'] = df['movies'].fillna("unknown")
df['publish_date'] = df['publish_date'].fillna("unknown")


# print(df.head())
# print(df.tail())
# print(df.info())
# print(df.isnull().sum())



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
tfidf = TfidfVectorizer(stop_words='english', max_features=100, ngram_range=(1,2))

# Fit only on training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform validation and test data using same vocabulary
X_val_tfidf = tfidf.transform(X_val)
# X_test_tfidf = tfidf.transform(X_test)   # if you have test set

#  selection and model fitting

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_tfidf, y_train)


# Step 3: Predictions
y_pred = log_reg.predict(X_val_tfidf)


# Step 4: Evaluation "while testing predection"
print("\n--- Classification Report ---")
print(classification_report(y_val, y_pred))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_val, y_pred))


# print(df['text'].head(10))
# print(df['text'].apply(lambda x: len(x.split())).describe())
# print(df[df['text'].str.len() < 50].shape)


print(df['label'].value_counts())


print(fake1_df['source'].value_counts().head())
print(real1_df['source'].value_counts().head())



# # step 3: cross valadition

# # Stratified K-Fold (keeps class balance in each fold)
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Vectorize on the full dataset for cross-validation
# X_tfidf = tfidf.fit_transform(X)

# # Cross-validation (Accuracy as example)
# scores = cross_val_score(log_reg, X_tfidf, y, cv=cv, scoring='accuracy')

# print(df['label'].value_counts())

# print(df['text'].head())
# print(df['text'].isna().sum())


# print("Cross-validation scores:", scores)
# print("Mean Accuracy:", scores.mean())
# print("Std Deviation:", scores.std())


