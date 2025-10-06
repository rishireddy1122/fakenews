# import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression



# # Download latest version
# path = kagglehub.dataset_download("mdepak/fakenewsnet")

# print("Path to dataset files:", path)

# df = pd.read_csv('archive\BuzzFeed_fake_news_content.csv')
fake_df = pd.read_csv("archive\BuzzFeed_fake_news_content.csv")
real_df = pd.read_csv("archive\BuzzFeed_real_news_content.csv")



# fake_df = pd.read_csv("archive/PolitiFact_fake_news_content.csv")
# real_df = pd.read_csv("archive/PolitiFact_real_news_content.csv")


# print("Fake shape:", fake_df.shape)
# print("Real shape:", real_df.shape)

# print(real_df.head())
# print(real_df.tail())
# print(real_df.info())
# print(real_df.isnull().sum())

#  replace the empty authors with unknown for better eff
fake_df['authors'] = fake_df['authors'].fillna("unknown")
fake_df['source'] = fake_df['source'].fillna("unknown")
fake_df['canonical_link'] = fake_df['canonical_link'].fillna("unknown")

real_df['authors'] = real_df['authors'].fillna("unknown")
real_df['source'] = real_df['source'].fillna("unknown")
real_df['canonical_link'] = real_df['canonical_link'].fillna("unknown")

#  to remove unwanted values 
# df.dropna(subset=['movies'], inplace=True)

# print("\nbefore cleaning, dataset shape:", df.shape)
# print("\nAfter cleaning, dataset shape:", df.shape)


# Add labels
# 1 = fake, 0 = real
fake_df['label'] = 1
real_df['label'] = 0


# Combine into one DataFrame
df = pd.concat([fake_df, real_df], axis=0).reset_index(drop=True)


# print(df.head())
# print(df.tail())
# print(df.info())
# print(df.isnull().sum())

df['movies'] = df['movies'].fillna("unknown")
df['publish_date'] = df['publish_date'].fillna("unknown")


# print("Combined shape:", df.shape)
# print(df.head())

avg_len = fake_df['text'].str.split().apply(len).mean()
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


# Step 2: Baseline Model (Logistic Regression) next multinominal
# model = MultinomialNB()
# model.fit(X_train_tfidf, y_train)








log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_tfidf, y_train)

y_pred_lr = log_reg.predict(X_val_tfidf)
print(classification_report(y_val, y_pred_lr))







# Stratified K-Fold (keeps class balance in each fold)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation (Accuracy as example)
scores = cross_val_score(log_reg, X, y, cv=cv, scoring='accuracy')


# model = LogisticRegression(max_iter=1000)
# model.fit(X_train_tfidf, y_train)

# Step 3: Predictions
y_pred = log_reg.predict(X_val_tfidf)

# Step 4: Evaluation
print("\n--- Classification Report ---")
print(classification_report(y_val, y_pred))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_val, y_pred))

#  while testing all data

# print("Cross-validation scores:", scores)
# print("Mean Accuracy:", scores.mean())
# print("Std Deviation:", scores.std())

