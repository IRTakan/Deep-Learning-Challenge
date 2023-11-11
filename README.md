# Deep-Learning-Challenge

For this challenge I worked with The nonprofit foundation Alphabet Soup. They wanted for me to create a tool that can help them select the applicants for 
funding with the best chance of success in their ventures. With my knowledge of machine learning and neural networks, I used the 
features in the provided dataset to create a binary classifier that can predict whether applicants would be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, I received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. 
Within the dataset were a number of columns that captured metadata about each organization, such as:

- EIN and NAME—Identification columns.
  
- APPLICATION_TYPE—Alphabet Soup application type.
  
- AFFILIATION—Affiliated sector of industry.
  
- CLASSIFICATION—Government organization classification.
  
- USE_CASE—Use case for funding.

- ORGANIZATION—Organization type.
  
- STATUS—Active status.

- INCOME_AMT—Income classification.
  
- SPECIAL_CONSIDERATIONS—Special considerations for application.

- ASK_AMT—Funding amount requested.
  
- IS_SUCCESSFUL—Was the money used effectively.

To complete this task, I had to go through a number of steps:

--Step 1: Preprocess the Data--

- I used my knowledge of Pandas and scikit-learn’s StandardScaler() to preprocess the dataset. This step helped prepare for Step 2, where I compiled, trained, 
and evaluated the neural network model.

- Charity_data.csv was read to a Pandas DataFrame, where I had to identify the following in the dataset:
What variable(s) were the target(s) for your model?
What variable(s) were the feature(s) for your model?

- Afterwards I Droped the EIN and NAME columns, determined the number of unique values for each column and for columns that had more than 10 unique values,
determine the number of data points for each unique value.

- The number of data points for each unique value were used to pick a cutoff point to bin "rare" categorical variables together in a new value,
Other, and then checked if the binning was successful.

- Used pd.get_dummies() to encode categorical variables.

- Split the preprocessed data into a features array, X, and a target array, y. Used the arrays and the train_test_split function to split the data into training and testing datasets.

- Scaled the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.
  
