# =====================================================
# Fakenews Classification Pipeline: TF-IDF + Metadata + Logistic Regression + GridSearchCV
# =====================================================

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =====================================================
# 1. Load Datasets
# =====================================================
fake_df = pd.read_csv("archive/BuzzFeed_fake_news_content.csv")
real_df = pd.read_csv("archive/BuzzFeed_real_news_content.csv")

# =====================================================
# 2. Handle Missing Values
# =====================================================
for df in [fake_df, real_df]:
    df['authors'] = df['authors'].fillna("unknown")
    df['source'] = df['source'].fillna("unknown")
    df['canonical_link'] = df['canonical_link'].fillna("unknown")
    df['movies'] = df['movies'].fillna("unknown")
    df['publish_date'] = df['publish_date'].fillna("unknown")

# =====================================================
# 3. Add Labels
# =====================================================
fake_df['label'] = 1  # Fake news
real_df['label'] = 0  # Real news

# =====================================================
# 4. Combine into a single DataFrame
# =====================================================
df = pd.concat([fake_df, real_df], axis=0).reset_index(drop=True)

# =====================================================
# 5. Create Metadata Features
# =====================================================
df['text_length'] = df['text'].str.split().apply(len)     # Number of words in article
df['title_length'] = df['title'].str.split().apply(len)   # Number of words in title

# =====================================================
# 6. Combine Text Columns for TF-IDF
# =====================================================
df['combined'] = (
    df['title'].fillna('') + ' ' +
    df['text'].fillna('') + ' ' +
    df['authors'].fillna('') + ' ' +
    df['source'].fillna('')
)

# =====================================================
# 7. Define Features and Target
# =====================================================
X_text = df['combined']                           # Text feature for TF-IDF

meta_features = df[['text_length', 'title_length', 'source']]  # Metadata features
y = df['label']                                  # Target labels

# =====================================================
# 8. Split Train/Validation Sets
# =====================================================
# ✅ Split once on the FULL feature set
df_train, df_val = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# ✅ Now extract correctly
train_df = df_train[['combined', 'source', 'text_length', 'title_length']]
val_df   = df_val[['combined', 'source', 'text_length', 'title_length']]

y_train = df_train['label']
y_val   = df_val['label']



# =====================================================
# 9. Create ColumnTransformer to process text + metadata
# =====================================================
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(stop_words='english'), 'combined'),     # TF-IDF for text
        ('source', OneHotEncoder(handle_unknown='ignore'), 'source'),    # One-hot encoding for categorical source
        ('numbers', 'passthrough', ['text_length', 'title_length'])      # Numeric metadata
    ]
)

# =====================================================
# 10. Create Pipeline with Preprocessing + Classifier
# =====================================================
pipeline = Pipeline([
    ('prep', preprocessor),                      # Preprocessing step
    ('clf', LogisticRegression(max_iter=2000))  # Logistic Regression classifier
])

# =====================================================
# 11. Define Hyperparameter Grid for GridSearchCV
# =====================================================
param_grid = {
    'prep__text__max_features': [5000, 10000],          # Number of TF-IDF features
    'prep__text__ngram_range': [(1, 1), (1, 2)],       # Unigrams vs. unigrams+bigrams
    'clf__C': [0.1, 1, 10],                             # Regularization strength
    'clf__solver': ['lbfgs', 'liblinear'],             # Solver options
    'clf__penalty': ['l2']                              # Regularization type
}

# =====================================================
# 12. Set up GridSearchCV with Stratified 5-Fold
# =====================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# =====================================================
# 13. Fit the Model on Training Data
# =====================================================






 # Combine text + metadata
 train_df = pd.concat([
    X_text_train.rename("combined"),
    meta_train.reset_index(drop=True)
], axis=1)
grid.fit(train_df, y_train)





# =====================================================
# 14. Output Best Parameters and Cross-Validation Score
# =====================================================

print("\nBest Parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

# =====================================================
# 15. Evaluate on Validation Set
# =====================================================
val_df = pd.concat([
    X_text_val.rename("combined"),
    meta_val.reset_index(drop=True)
], axis=1)
# Combine text + metadata
y_pred = grid.predict(val_df) 

print("\nValidation Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))
