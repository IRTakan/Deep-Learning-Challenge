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

--Step 1: Preprocessed the Data--

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

--Step 2: Compiled, Trained, and Evaluated the Model--

- Used my knowledge of TensorFlow, to design a neural network, or deep learning model, to create a binary classification model that could predict if an Alphabet Soup-funded organization would be successful based on the features in the dataset. I needed to think about how many inputs there were before determining the number of neurons and layers in my model. Once I completed that step, I compiled, trained, and evaluated my binary classification model to calculate the model’s loss and accuracy.

- Continued using the file in Google Colab in which you performed the preprocessing steps from Step 1.

- Created a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

- Created the first hidden layer and choose an appropriate activation function.

- Added a second hidden layer with an appropriate activation function.

- Created an output layer with an appropriate activation function.

- Checked the structure of the model.

- Compiled and trained the model.

- Created a callback that saved the model's weights every five epochs.

- Evaluated the model using the test data to determine the loss and accuracy.

- Saved and exported your results to an HDF5 file. Named the file AlphabetSoupCharity.h5.

--Step 3: Optimized the Model--

Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Had different methods of the following to optimize my model: 

Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:

- Dropping more or fewer columns.

- Creating more bins for rare occurrences in columns.

- Increasing or decreasing the number of values for each bin.

- Add more neurons to a hidden layer.

- Add more hiden layers.
  
- Use different activation functions for the hidden layers.

- Add or reduce the number of epochs to the training regimen.

- Created a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.

- Imported my dependencies and read it in the charity_data.csv to a Pandas DataFrame.

- Preprocessed the dataset as I did in Step 1.

- Designed a neural network model, and adjusted it for modifications that would optimize my model to achieve higher than 75% accuracy.

- Saved and exported myresults to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

--Step 4: Wrote a Report on the Neural Network Model--

For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.

The report should contain the following:

Overview of the analysis: Explain the purpose of this analysis.

Results: Using bulleted lists and images to support your answers, address the following questions:

Data Preprocessing

What variable(s) are the target(s) for your model?
What variable(s) are the features for your model?
What variable(s) should be removed from the input data because they are neither targets nor features?
Compiling, Training, and Evaluating the Model

How many neurons, layers, and activation functions did you select for your neural network model, and why?
Were you able to achieve the target model performance?
What steps did you take in your attempts to increase model performance?
Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
  
