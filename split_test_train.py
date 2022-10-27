import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset_in_train_test(df: pd.DataFrame, test_size: float, random_state: int = 42):
    dataset = df.copy()
    train_dataset, test_dataset = train_test_split(dataset, test_size=test_size, random_state=random_state)
    return train_dataset, test_dataset

if __name__ == '__main__':

   dataset = pd.read_csv('data/iris_featurized.csv')
   train_dataset, test_dataset = split_dataset_in_train_test(dataset, test_size=0.2, random_state=42)
   train_dataset.to_csv('data/train.csv', index=False)
   test_dataset.to_csv('data/test.csv', index=False)
