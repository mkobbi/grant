import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Loading data and stripping accents
url = "https://raw.githubusercontent.com/mkobbi/subvention-status-datacamp/master/data/subventions-accordees-et-refusees.csv"
data = pd.read_csv(url, sep=";")
data = data.rename(columns=lambda x: x.decode('utf-8').encode('ascii', errors='ignore'))

# Generating y
y = np.asarray(pd.DataFrame([data["Total vot"] > 0.0]).astype(int))

# Dropping "Total vot" column
data = data.drop(columns=['Total vot'])

# Splitting X and y into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(data.values, y.T, test_size=0.1, random_state=42)

# Concatenate headers
headers = list(data) + ['y']

# Recombining previous 4 subsets into train and test data frames
train = pd.DataFrame(data=np.hstack((X_train, y_train)), columns=headers)
test = pd.DataFrame(data=np.hstack((X_test, y_test)), columns=headers)

# Rewriting the files
train.to_csv(path_or_buf="data/train.csv", sep=';', mode='w')
test.to_csv(path_or_buf="data/test.csv", sep=';', mode='w')
