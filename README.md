# Machine Learning Developer Test Assignment 

**Goal**
The purpose of this exercise is to simulate a real life problem that a Machine Learning Developer will handle while working with our team.
We want the candidate to “feel” one of the research domains the company is exploring, and we want to be able to assess the technical/coding skills of the candidate.

**Guidelines**
- The output for this exercise should be Python file/s (.py) + .pdf report that summarizes the work with the different steps taken, insights and instructions on how to reproduce the experiment and visualization (if any is relevant)
- You have 5 days to deliver the test via e-mail with both files.

**Exercise**
- You will be provided with a pickle file that contains all the necessary data. They are embeddings from a classification model. 
- The classification task was to classify the genetic syndromes (syndrome_id) of a given image.
- The structure of the dictionary saved in the pickle file: {'syndrome_id': { 'subject_id': {'image_id': [320x1 encoding]}}}
- If you get the "numpy.core._multiarray_umath" error when loading the pickle file, please upgrade your numpy package.
- The steps to perform:
    a. Plotting tSNE of the inputs, explaining the statistics and the data
    b. Do a 10 fold cross validation for the following steps:
        Calculate cosine distance from each test set vector to the gallery vectors
        Calculate euclidean distance from each test set vector to the gallery vectors
        Classify each image (vector) or each subject to syndrome Ids based on KNN algorithm for both cosine and euclidean distances.
    c. Create automatic tables in a txt / pdf file for both algorithms, to enable comparison (please specify top-k, AUC etc.)
    d. Create an ROC AUC graph comparing both algorithms (2 outputs in the same graph, averaged across gallery / test splits)
- **Bonus:** Create 1-2 simple unit tests to your choice (preferably with pytest or similar).
