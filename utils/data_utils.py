import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_lendingclub_data(filepath: str):
    df = pd.read_csv(filepath)
    df = df.dropna()
    
    # Encode 'purpose'
    df['purpose'] = LabelEncoder().fit_transform(df['purpose'])
    
    feature_cols = [
        'credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc',
        'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util',
        'inq.last.6mths', 'delinq.2yrs', 'pub.rec'
    ]
    
    X = df[feature_cols]
    y = df['not.fully.paid']  # <-- ✅ Use 'not.fully.paid' as your target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
