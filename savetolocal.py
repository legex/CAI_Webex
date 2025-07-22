import pandas as pd

data = {
    "query": [],
    "response":[]

}

df = pd.DataFrame(data)
df.to_csv('people_pandas.csv', index=False)

print("CSV file 'people_pandas.csv' created successfully using pandas.")
