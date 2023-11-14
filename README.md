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

For this part of the assignment, I wrote a report on the performance of the deep learning model you created for Alphabet Soup.

The report contained the following:

- Overview of the analysis: Explaining the purpose of this analysis.

Results: Used bulleted lists and images to support my answers and addressed the following questions:

Data Preprocessing

- What variable(s) were the target(s) for your model?

- What variable(s) were the features for your model?

- What variable(s) should have been removed from the input data because they are neither targets nor features?

Compiling, Training, and Evaluating the Model

- How many neurons, layers, and activation functions were selected for my neural network model, and why?

- Was I able to achieve the target model performance?

- What steps did I take in my attempts to increase model performance?

- Summary: Summarized the overall results of the deep learning model. Included a recommendation for how a different model could solve this classification problem,
and then explained my recommendation.

# Report

After running different tests and various combinations, the following settings were selected based on results, demonstrating the most optimized model in terms of higher accuracy:

- Hidden layer 1: 500 neurons using ReLU as the activation function.

- Hidden layer 2: 300 neurons using Sigmoid as the activation function.

- Hidden layer 3: 150 neurons using Sigmoid as the activation function.

Sigmond used as the activation function with fifty epochs.

Whilst there many other combinations that could be used to produce improved results, these settings represented the best peforming
configuration among the other tested examples. Multiple configurations were used to try and maximize the accuracy of the model with this example 
being one of the best.

The highest accuracy achieved with the initial settings was 0.7272303104400635 (73%). After incorporating changes to the input data, the predictive accuracy reached it's highest target of  0.7537026405334473 (75%).

--Increasing Model Performance--

Initial Model: 2 layers, 85 and 40 neurons and fifty epochs.

IMAGE

First approach: Changing the number of epochs:

Model 2: 2 layers. 80, 30 neurons and thirty epochs.

After experimenting with different epoch values, With high number like like 150 and gradually reducing it, the accuracy showed improvement with smaller values. 
This can be explained by the fact that reducing the number of epochs can sometimes leads to better results, particularly when the model starts overfitting or becomes too specialized to the training data. So thirty epochs were chosen which produced an accuarcy of 0.750437319278717, so an improvement from the 73% of the initial model.

Model 3: Adding a third layer. 500, 300, 150 neurons and fifteen epochs.

After adding a third layer and combining different activation functions, implementing three hidden layers with the following settings resulted in an accuracy of 
Accuracy:  0.7514868974685669 (75%), so the similar to model 2.

Second approach: Change the data input:

--Creating more bins for rare occurrences in columns and increasing or decreasing the number of values for each bin:--

Initially, creating bins for the 'ASK_AMT' (funding amount requested) column was explored, but it did result in a decrease in accuracy. This outcome highlights the significance of this column for the neural network model. Ultimately, due to the observed decrease in accuracy, the idea of using bins was abandoned, and the binned data was not incorporated into the model,

The cutoff for the column 'APPLICATION_TYPE' was set at 600, and the column 'CLASSIFICATION' was set at 1000. After running a number of tests, it was determined that choosing 200 to bin the column APPLICATION_TYPE and 1000 to bin the column CLASSIFICATION seemed to be the values that would work.

--Dropping more columns:--

Dropping the STATUS and SPECIAL_CONSIDERATIONS columns resulted in a very small improvement in accuracy. This highlights the neutral impact of these two columns on the model.

--Creating bins for the column 'NAME' before dropping it:--

Creating bins for the 'NAME' column enables gathering insights into the distribution of name occurrences in the dataset.
The addition of this new column had a significantly positive impact, playing an important role in reaching the 75% accuracy.

Overall, by optimizing the model I was able to increase the accuracy from 73 to 75%. This means that was able to correctly classify each of the points in the test data 75% of the time. This means that applicants had a 75% chance of being successful if they had the following:

- The NAME of the applicant appears more than 5 times (they have applied more than 5 times).
- The type of APPLICATION is one of the following: T3, T4, T5, T6 and T19.
- The application has the following values for CLASSIFICATION: C1000, C1200, C2000,C2100 and C3.

I decided to explore alternatives like the Random Forest classifier. Despite a slightly lower accuracy of 0.734 it does offers some advantages.
In contrast, Linear Regression, which was also considered, did not provide meaningful insights or contribute significantly to understanding the dataset better.
The Random Forest model is used in scenarios where interpretability is crucial. Its strong, lower susceptibility to overfitting making it a more appealing choice for specific applications that are looking to offer a more comprehensive set of metrics for a nuanced evaluation of model performance. The choice between the neural network and the Random Forest depends on what the desired goal is. If the primary goal here is maximize accuracy, the neural network is better. However, if interpretability is a priority and a slight reduction in accuracy is something that yu don't mind then the Random Forest model provides valuable insights.

*Technologies used: Microsoft Visual Studio Code. Languages: Python
