# Data Directory

Place your datasets here for training and testing.

## Expected Format

Your dataset should be a CSV file with the following structure:

```csv
text,label
"News article text here",0
"Another news article",1
```

Where:
- `text`: The news article or statement text
- `label`: 0 for real news, 1 for fake news

## Sample Datasets

You can use publicly available fake news datasets such as:

1. **LIAR dataset**: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
2. **Fake News Dataset**: https://www.kaggle.com/c/fake-news/data
3. **ISOT Fake News Dataset**: https://www.uvic.ca/engineering/ece/isot/datasets/

## Data Preprocessing

The training script will automatically:
- Clean the text (remove URLs, punctuation, numbers)
- Remove stopwords
- Apply stemming
- Create train/test splits

## Notes

- Larger datasets (10,000+ samples) will produce better models
- Balance your dataset (roughly equal fake and real examples)
- Ensure text quality and consistency
