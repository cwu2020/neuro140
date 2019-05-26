# neuro140
Using Machine Learning Techniques to Decode Neural Signals

Neuro 140, Spring 2019
Carra Wu

Introduction 

In “Using EEG to Improve Massive Open Online Courses Feedback Interaction” (2013), Wang et al. investigate the feasibility of using commercial EEG devices to detect students’ mental states and improve tutor-relevant feedback in the context of Massive Open Online Courses (MOOC), where student feedback is not as readily available as in a classroom context. Their pilot study, which used the signal from adults wearing single-channel EEG headsets while watching MOOC video clips, found that Gaussian Naïve Bayes classifiers trained and tested on user feedback were able to detect student confusion with above-chance probability. While their classifier performance was relatively weak, they found that it was comparable to human observers observing body language in predicting students’ confusion. These outcomes indicate that MOOC-deployable EEG devices may show promise in capturing tutor relevant information up to existing classroom standards.

In this paper, I seek to replicate the results achieved by Wang et al. by implementing my own Gaussian Naïve Bayes classifier on their published raw data. While the data collected on observer-defined confusion is not included in the public dataset, I evaluate my classifier performance by comparing my results to theirs using rates of test accuracy. For further pedagogical exploration and to investigate the neural-decoding capabilities of newer deep learning methods, I also implement three additional classifiers: a TensorFlow ANN, an XGBoost classifier, and a logisitic regression classifier. While each of these classification methods is by now well established in the AI/ML community, I’ve included some additional commentary on my process and observations for each method in order to demonstrate an earnest effort at grasping first principles.

My replication of the Naïve Bayes classifiers was able to achieve significantly better predictive accuracy than those implemented by Wang et al. (2013). In the Analysis section, I will discuss two hypotheses accounting for this difference. I will also analyze the three other classifiers implemented and include some areas for further exploration.

Experimental Design and Data Collection 

The pilot study conducted by Wang et al. consisted of 10 college students watching ten 2-minute long MOOC video clips. Each student watched 5 videos that were assumed to confuse the average college student (on topics such as quantum mechanics and stem cell research) and 5 videos that were assumed to be not confusing for an average college student (on topics such as basic algebra and geometry). In reporting the data, the leading 30 seconds and the final 30 seconds were removed by the researchers in order to account for the possibility that the student was not ready when the video started or had lost focus when the video was about to end.

Following each video, the student was asked to rate their confusion on a scale of 1-7 (with 1 corresponding to least confusing and 7 corresponding to most confusing). These ratings were then mapped to a student-specific binary label by taking a middle split and mapping scores less than or equal to the median to “not confusing” and scores greater than or equal to the median to “confusing.” In this way, the researchers were able to control for students’ differing senses of what was confusing and turn the multi-classification problem into a binary classification problem, which would perform better given the sparseness of the dataset.

The EEG device used was the NeuroSky “MindSet,” which is an audio headset equipped with a single-channel electroencephalography (EEG) sensor, which measures the voltage between an electrode resting on the forehead (positioned at Fp1) and electrodes in contact with the ear. This signal comes from large areas of coordinated neural activity, which manifest as groups of neurons firing at the same rate. While the synchronization behavior is a function of many things, including development, mental state, and cognitive activity, brain states such as focused attentional processing, frustration, and engagement can be captured by measuring the relative level of activity within several frequency bands in the EEG signal. NeuroSky’s API returned the raw EEG signals, two proprietary signals called attention and mediation, as well as signal streams corresponding to the standard named frequency bands, reported at 8 Hz. More information about the 11 classifier features can be found in Figure 1, below

Methodology 

One of the first hurdles I faced in replicating Wang et al. (2013)’s results was cleaning the raw dataset. Despite what was ostensibly a standardized video length and sampling rate, some students ended up with many more rows of data than other students, and it was initially unclear to me how I ought to manipulate the time series data to make it interpretable by a machine learning algorithm. I ended up following Wang et al. (2013)’s lead and characterizing the overall activity in each signal stream by the mean value across the timespan of each video. While this approach was able to minimize noise and standardize feature reporting, averaging across the entirety of each video reduced the number of rows of data from over 10,000 to exactly 100, making it difficult for the machine learning algorithms to learn and perform well. If a follow up study were conducted using a sophisticated EEG device with more sensors (as well as more fine-grained labelling), then a machine learning algorithm could perhaps search for patterns within the signal streams for each time interval, leading to additional insight into the link between neural patterns and cognition. In the interest of not having to learn Pandas (or another Python data frame package), I decided to manipulate the data using macros in Excel. This part of the project ended up being very time consuming, not to mention error-prone, so an area of future exploration is learning best practices for cleaning ML data.

The other data processing note is that when I examined the raw data, I found that the attention and mediation signal streams were all “0.00E+00” for subject 6. Suspecting some sort of hardware malfunction in the device itself, I discarded all the data for subject 6, even though the other feature statistics seemed to be similar to those of other subjects.

Figure 1: Classifier features 

Signal stream	Sampling rate	Statistic
Raw	1 Hz	Mean
Attention	1 Hz	Mean
Mediation	512 Hz	Mean
Delta: too much activity in this frequency band is linked to learning problems like severe ADHD, while too little is linked to the inability to revitalize the brain. Optimal delta wave activity is inked with strong immune system and restorative sleep.	8 Hz	Mean
Theta: too much activity in this frequency band is linked to hyperactivity, while too little is linked to stress. Optimal theta wave activity is linked with creativity, intuition, and relaxation.	8 Hz	Mean
Alpha 1: too much activity in this frequency band is linked to daydreaming and lack of focus, while too little is linked to OCD. Optimal alpha wave activity is linked with relaxation.	8 Hz	Mean
Alpha 2	8 Hz	Mean
Beta 1: too much activity in this frequency band is linked to adrenaline and high arousal, while too little is linked to depression and poor cognition. Optimal beta wave activity is linked with conscious focus, memory, and problem solving.	8 Hz	Mean
Beta 2 	8 Hz	Mean
Gamma 1: too much activity in this frequency band is linked to anxiety and stress, while too little is linked to learning disabilities. Optimal theta wave activity is linked with cognition, perception, and learning.	8 Hz	Mean
Gamma 2 	8 Hz	Mean

Figure 2: Visual representation of EEG frequency bands 

 

In this project, I implemented 4 different ML algorithms: Gaussian Naïve Bayes, XGBoost, Logistic Regression, and then a TensorFlow Artificial Neural Network (ANN).

The Gaussian Naïve Bayes classifier is a simple probabilistic classifier which assumes strong independence assumptions between each of the features.  My limited understanding of neurobiology makes it difficult for me to assess the independence of the signal streams. One potential pitfall may be that the raw EEG signal stream and the proprietary signal streams may have a dependent relationship with the power spectrum signal streams. Because Wang et al. used the Gaussian Naïve Bayes classifier, however, I still decided it was worth attempting. Naïve Bayes was chosen by the paper’s authors because it performs well on binary classification, which was something I considered when choosing an activation function for my ANN.

The XGBoost classifier is an implementation of gradient boosted decision trees, which optimizes for speed and performance.  Boosting is an approach to machine learning where new models are added to correct the errors made by existing models until no further improvements can be made, and gradient boosting is a modification to that approach where new models are added that predict the residuals of the previous models and whose outputs minimize the loss when added to the output of the existing sequence of trees.  One drawback to basic gradient boosting is that, like all decision trees, it is a greedy algorithm that can overfit a training dataset very quickly. 

Logistic regression is a regression analysis describes the relationship between the dependent variable (confusing or not confusing) and one or more independent variables (the features).  While also suitable for binary classification, logistic regression generally does not perform well with sparse and noisy training data, which was why the paper authors opted against it. However, I thought it would be interesting to implement a logistic regression and see if that assumption was correct in this case. 

My deep learning neural network was implemented with a sigmoid activation function (which I chose because, like the Gaussian Naïve Bayes method, it performs well on binary classification) and 10 hidden layers with Rectified Linear Unit (ReLU) activation functions between them. The sigmoid function (shown below in Figure 3) is a smooth, differentiable function that produces output in the (0,1) range, which means it is particularly useful for models where probability is the output.  The ReLu function (shown below in Figure 3) is the maximum of 0 and the input and is used between hidden layers because it looks and acts like a linear function, but converges quickly, is sparsely activated, and overcomes the vanishing gradient problem, which affects the sigmoid function at either end of the “S” curve. ,  The optimizer was stochastic gradient descent, which minimizes error by following the negative gradient of the objective (or loss) function, approaching a local minimum.  While I did not explore the space of hyperparameter optimization, I did fiddle around with the learning rate, number of epochs, hidden layers, and seed in order to arrive at a rough optimization and develop intuition.

Figure 3: Sigmoid versus ReLU activation functions 

 

One way the paper authors performed cross validation (to minimize overfitting), was by training two independent classifiers: a student-specific classifier, which reserved one video as the testing dataset for each student and averaged prediction accuracy across all students, and a student-independent classifier, which reserved one student as the testing dataset and outputted a single prediction and prediction accuracy. They used 10-fold cross-validation, split along the 10 students and the 10 videos each. Since I left out subject 6, I was not able to use 10-fold cross-validation and instead chose a slightly different approach. Using the sklearn train-test-split function, I simply reserved 20% of each dataset for testing. Otherwise, my implementation of student-specific and student-independent classifiers was consistent with Wang et al.’s implementation.

Results and Analysis

Gaussian Naïve Bayes test accuracy:
 

XGBoost test accuracy:
  

Logistic regression test accuracy (only the student-independent classifier):  

TensorFlow ANN test accuracy (only the student-independent classifier) 

For their Gaussian Naïve Bayes classifier, Wang et al. achieved student-specific and student-independent test accuracies of 56% and 51% respectively. My implementation achieved student-specific and student-independent test accuracies of 67% and 61% respectively. Some of the disparity may be due to the fact that I removed subject number 6 from the test data, did not use k-fold cross validation, and used data with only two significant digits (it is unclear how many significant digits their data was, but I expect they simplified and compressed the public dataset).

One surprising outcome is that at 67%, the logistic regression classifier had a student-independent test accuracy higher than the Gaussian Naïve Bayes classifier, which was assumed to perform better for sparse datasets. An expected outcome was that the XGBoost student-specific classifier, which was trained on a dataset consisting of only 10 data points per feature, performed atrociously on the test set, with an accuracy of only 33%, while the student-independent classifier trained on 80 data points per feature, performed significantly better, at 72%. 

My deep learning algorithm (the paper authors later commented on Kaggle that deep learning was not as popular in neural decoding when the paper was published, which was why they did not opt for CNNs or ANNs) out-performed both the Gaussian and Naïve Bayes classifiers with a test accuracy of 69%. The deep learning and XGBoost test accuracies performed similarly, which is consistent with general findings in the AI/ML field. 

Discussion

As I was beginning my dive into deep learning and machine learning methods, I first referenced a kernel that an independent Kaggle user had created, which claimed it could attain predictive accuracy of up to 90% using XGBoost.  Upon further inspection, I found that in their data processing step, they merged the demographic data (ethnicity, age, and gender) of each subject with the raw EEG dataset. The comments on that kernel raise ethical questions about whether ethnic and demographic data should be used to predict student performance outcomes like confusion. While it is not currently understood whether ethnicity and gender impact brainwaves, from a scientific standpoint, one clear failure of this method is that when students of other ethnicities (not included in the training set) use the device, the prediction for these students could be rendered useless due to its potential for large variance.
 
In this study, some factors that may have limited prediction accuracy across all classifiers were small sample size and low number of sensors, which produced noise in the dataset. Even without increasing the number of sensors, it is worth investigating whether certain frequency bands are more strongly predictive of confusion than others. Since it was not in the scope of Wang et al.’s 2013 paper, further research should be done in the feature selection space, which not only furthers the practical purposes of the 2013 paper by improving classification algorithms, but also contributes to the body of knowledge surrounding the interpretation of neural activity. For example, Wang et al. indicate that the theta frequency band may have been most strongly correlated with the outcome variable. Without this knowledge, it seems reasonable that any or all of the frequency bands could be the strongest feature(s) to select. Knowing that theta was most predictive of confusion, however, we can hypothesize that since theta is the only frequency linked with creativity and intuition, the ability to make horizontal connections is more strongly linked with not being confused than the other feature attributes.

Conclusion

Because it is not particularly burdensome for students to manually label concepts as confusing or not confusing, it is unlikely that MOOCs will distribute expensive EEG headsets to students anytime soon. While the results from Wang et al.’s paper are not practically significant, the well-above-chance predictive power of the classifiers used in “Using EEG to Improve Massive Open Online Courses Feedback Interaction” and my replication serve to validate machine learning and deep learning algorithms as tools to better understand the link between various signal streams and mental state outcomes.

Areas for further exploration and application include using EEG to classify mental health conditions and neurocognitive disorders such as depression, bipolar disorder, and schizophrenia. Fine-grained data collected from sophisticated, non-commercial EEG devices could also open up possibilities for multi-classification algorithms, which may yield more insight to advance the field of neural decoding.





Sources
Brownlee, Jason. “A Gentle Introduction to the Gradient Boosting Algorithm for Machine Learning.” Machine Learning Mastery, 20 Nov. 2018, machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/.
Brownlee, Jason. “A Gentle Introduction to the Rectified Linear Unit (ReLU) for Deep Learning Neural Networks.” Machine Learning Mastery, 15 May 2019, machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/.
Brownlee, Jason. “A Gentle Introduction to XGBoost for Applied Machine Learning.” Machine Learning Mastery, 21 Sept. 2016, machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/.
Brownlee, Jason. “Naive Bayes for Machine Learning.” Machine Learning Mastery, 22 Sept. 2016, machinelearningmastery.com/naive-bayes-for-machine-learning/.
“Greek Alphabet Soup – Making Sense of EEG Bands.” NeuroSky, 19 May 2015, neurosky.com/2015/05/greek-alphabet-soup-making-sense-of-eeg-bands/.
Liu, Danqing. “A Practical Guide to ReLU.” Medium, TinyMind, 30 Nov. 2017, medium.com/tinymind/a-practical-guide-to-relu-b83ca804f1f7.
Ng, Andrew. “Optimization: Stochastic Gradient Descent.” Unsupervised Feature Learning and Deep Learning Tutorial, 2013, deeplearning.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/.
Nowak, William. “Classifying Mental State (Sklearn Logistic, Tf NN).” Kaggle, 2017, www.kaggle.com/wpncrh/classifying-mental-state-sklearn-logistic-tf-nn.
Rathee, Amandeep. “Random Forest vs XGBoost vs Deep Neural Network.” Kaggle, 18 May 2017, www.kaggle.com/arathee2/random-forest-vs-xgboost-vs-deep-neural-network.
Sharma, Sagar. “Activation Functions in Neural Networks.” Towards Data Science, Towards Data Science, 6 Sept. 2017, towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6.
Slater, Neil. “Gradient Boosting Tree: ‘the More Variable the Better’?” Data Science Stack Exchange, 5 Mar. 2017, datascience.stackexchange.com/questions/17364/gradient-boosting-tree-the-more-variable-the-better.
Wang, Haohan. “Using EEG to Improve Massive Open Online Courses Feedback Interaction.” Carnegie Mellon University, 2013, www.cs.cmu.edu/~kkchang/paper/WangEtAl.2013.AIED.EEG-MOOC.pdf.
“What Is Logistic Regression?” Statistics Solutions, 2019, www.statisticssolutions.com/what-is-logistic-regression/.

