"""
Script to collect fake news dataset.
This script downloads a publicly available fake news dataset.
"""

import pandas as pd
import os

def collect_data():
    """
    Collects fake news dataset from a CSV file or creates a sample dataset.
    For this implementation, we'll create a sample dataset that mimics
    real fake news data structure.
    """
    
    # Sample dataset with fake and real news
    data = {
        'text': [
            'Scientists discover cure for all diseases overnight',
            'Local government announces new infrastructure project',
            'Aliens confirmed to be living among us says anonymous source',
            'Stock market shows steady growth over quarter',
            'Breaking: Celebrity caught in shocking scandal with no evidence',
            'Research team publishes peer-reviewed study on climate change',
            'Miracle pill makes you lose 50 pounds in one week',
            'Economic report shows unemployment rate decreased',
            'Shocking truth they don\'t want you to know about water',
            'University researchers develop new vaccine through clinical trials',
            'Government secretly controlling weather patterns',
            'Transportation department releases new traffic guidelines',
            'Doctors hate this one weird trick',
            'Medical journal publishes findings on new treatment',
            'Celebrity endorses miracle product that cures everything',
            'Annual budget allocation announced by finance ministry',
            'Secret society controls all world governments',
            'Educational institution reports increased enrollment',
            'Shocking conspiracy revealed by unnamed insider',
            'Technology company releases quarterly earnings report',
        ],
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        # 1 = fake news, 0 = real news
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = 'data/fake_news_data.csv'
    os.makedirs('data', exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Data collected and saved to {output_path}")
    print(f"Total samples: {len(df)}")
    print(f"Fake news samples: {sum(df['label'] == 1)}")
    print(f"Real news samples: {sum(df['label'] == 0)}")
    
    return df

if __name__ == "__main__":
    collect_data()
