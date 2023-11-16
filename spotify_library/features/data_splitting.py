from sklearn.model_selection import train_test_split

#function to split the data into train and test data
def splitdata(df, test_size, random_state):
    traindf, testdf = train_test_split(df, test_size=test_size, random_state=random_state) 
    traindf = traindf.reset_index(drop=True) 
    testdf = testdf.reset_index(drop=True)
    return traindf, testdf
