eval_dataset = [
    {
        "question": "What is the difference between supervised and unsupervised learning?",
        "expected_answer": "In supervised learning, the algorithm learns from labeled input-output pairs provided by a user. In unsupervised learning, only input data is given with no labels, and the algorithm must find structure on its own.",
    },
    {
        "question": "How does the k-Nearest Neighbors algorithm make predictions?",
        "expected_answer": "k-NN finds the k closest data points in the training set to the new point and predicts the majority class among those neighbors. In its simplest form (k=1), it simply assigns the label of the single nearest training point.",
    },
    {
        "question": "What is overfitting and how can it be detected?",
        "expected_answer": "Overfitting occurs when a model is too complex and performs very well on training data but poorly on unseen test data. It is detected by comparing training accuracy to test accuracy — a large gap indicates overfitting.",
    },
    {
        "question": "What is the difference between Ridge and Lasso regression?",
        "expected_answer": "Both are linear regression variants with regularization controlled by an alpha parameter. Ridge uses L2 regularization, which shrinks all coefficients, while Lasso uses L1 regularization, which can set some coefficients to exactly zero, effectively performing feature selection.",
    },
    {
        "question": "Why do SVMs and neural networks require feature scaling?",
        "expected_answer": "SVMs and neural networks are sensitive to the scale of input features. Without scaling, features with larger ranges dominate the model. Preprocessing with StandardScaler (zero mean, unit variance) ensures all features contribute equally.",
    },
    {
        "question": "What is the purpose of cross-validation, and what is k-fold cross-validation?",
        "expected_answer": "Cross-validation provides a more robust estimate of a model's generalization performance than a single train/test split. In k-fold cross-validation, the data is split into k equal parts (folds); the model is trained k times, each time using a different fold as the test set and the remaining folds as training data.",
    },
    {
        "question": "What does GridSearchCV do in scikit-learn?",
        "expected_answer": "GridSearchCV exhaustively searches over a specified parameter grid, evaluating each combination using cross-validation. After fitting, it exposes best_params_ (optimal parameters) and best_estimator_ (the best model retrained on the full training data).",
    },
    {
        "question": "How does StandardScaler work and why should it only be fit on training data?",
        "expected_answer": "StandardScaler transforms features to have zero mean and unit variance. It must be fit only on training data to avoid data leakage — applying test set statistics during training would give the model information it shouldn't have, leading to overly optimistic performance estimates.",
    },
    {
        "question": "What is Principal Component Analysis (PCA) and what is it used for?",
        "expected_answer": "PCA rotates the dataset so that the new features (principal components) are statistically uncorrelated and ordered by how much variance they explain. It is used for dimensionality reduction, visualization, and noise removal by retaining only the top components.",
    },
    {
        "question": "How does DBSCAN differ from k-Means clustering?",
        "expected_answer": "DBSCAN is a density-based algorithm that does not require specifying the number of clusters in advance, and it can identify noise points (labeled -1). k-Means requires specifying n_clusters beforehand and assigns every point to a cluster, creating roughly equal-sized groups.",
    },
    {
        "question": "What are the key parameters of DBSCAN?",
        "expected_answer": "DBSCAN has two main parameters: eps (the radius of a point's neighborhood) and min_samples (the minimum number of points required within the neighborhood for a point to be considered a core point). Points not reachable from any core point are labeled as noise (-1).",
    },
    {
        "question": "What is the difference between precision and recall?",
        "expected_answer": "Precision is the fraction of predicted positives that are truly positive (TP / (TP + FP)), measuring how many predictions are correct. Recall is the fraction of actual positives that were correctly predicted (TP / (TP + FN)), measuring how many true positives were found.",
    },
    {
        "question": "Why is ROC AUC a better metric than accuracy for imbalanced datasets?",
        "expected_answer": "Accuracy can be misleadingly high when one class dominates — a classifier that always predicts the majority class achieves high accuracy without being useful. ROC AUC evaluates the ranking of positive vs. negative examples regardless of class balance, and a random classifier always scores 0.5.",
    },
    {
        "question": "What is a Pipeline in scikit-learn and why is it useful?",
        "expected_answer": "A Pipeline chains multiple processing steps (e.g., a scaler followed by a classifier) into a single object. It ensures that preprocessing is consistently applied during cross-validation and grid search, preventing data leakage by fitting transformers only on training folds.",
    },
    {
        "question": "How does Random Forest improve upon a single Decision Tree?",
        "expected_answer": "A Random Forest builds many decision trees on random subsets of features and training data, then aggregates their predictions by voting. This reduces overfitting and variance compared to a single tree, while providing more reliable feature importances.",
    },
    {
        "question": "What is the bag-of-words representation for text data?",
        "expected_answer": "The bag-of-words model represents each document as a vector of token counts from a fixed vocabulary, ignoring word order. In scikit-learn, CountVectorizer implements this by building a vocabulary from the training corpus and counting token occurrences per document.",
    },
    {
        "question": "What is TF-IDF and how does it differ from raw word counts?",
        "expected_answer": "TF-IDF (Term Frequency-Inverse Document Frequency) downweights tokens that appear frequently across many documents (like 'the'), giving more importance to words that are distinctive to particular documents. Unlike raw counts from CountVectorizer, TF-IDF highlights informative terms.",
    },
    {
        "question": "What are n-grams and how are they used in text feature extraction?",
        "expected_answer": "N-grams are contiguous sequences of n tokens (bigrams = 2, trigrams = 3). They capture some word-order context that bag-of-words misses. In scikit-learn, the ngram_range parameter of CountVectorizer or TfidfVectorizer controls which n-gram lengths are included as features.",
    },
    {
        "question": "What is the purpose of one-hot encoding for categorical variables?",
        "expected_answer": "One-hot encoding converts a categorical feature into multiple binary (0/1) columns, one per category. This prevents machine learning algorithms from incorrectly interpreting arbitrary integer codes as ordinal (ordered) values, since distance between categories is not meaningful.",
    },
    {
        "question": "What is t-SNE and what are its limitations compared to PCA?",
        "expected_answer": "t-SNE is a manifold learning technique primarily used for 2D or 3D visualization of high-dimensional data. Unlike PCA, t-SNE cannot transform new data points after fitting — it must be re-run on the full dataset — and is not suitable for general dimensionality reduction or preprocessing.",
    },
]


eval_rag_data = [
    # ==========================================
    # CH. 1 INTRODUCTION (~10 QAs)
    # ==========================================
    {
        "question": "What is machine learning?",
        "ground_truth": "Machine learning is about extracting knowledge from data. It is a research field at the intersection of statistics, artificial intelligence, and computer science and is also known as predictive analytics or statistical learning."
    },
    {
        "question": "What are two major disadvantages of manually crafting expert decision rules using if/else structures?",
        "ground_truth": "The two major disadvantages are: 1) The logic required to make a decision is specific to a single domain and task, meaning a slight change requires a full rewrite. 2) Designing rules requires a deep understanding of how a decision should be made by a human expert."
    },
    {
        "question": "What is the key difference between supervised and unsupervised machine learning?",
        "ground_truth": "In supervised learning, the user provides the algorithm with pairs of inputs and desired outputs, and the algorithm generalizes from known examples. In unsupervised learning, only the input data is known, and no known output data is given to the algorithm."
    },
    {
        "question": "In scikit-learn terminology, what do the rows and columns of a data table represent?",
        "ground_truth": "Each row represents a sample (or data point), while each column represents a property that describes those entities, called a feature."
    },
    {
        "question": "Why is the capital letter 'X' used to denote the feature data while a lowercase 'y' is used to denote the labels?",
        "ground_truth": "This is inspired by the standard mathematical formulation f(x)=y. A capital 'X' is used because the input data is a two-dimensional array (a matrix) and a lowercase 'y' is used because the target is a one-dimensional array (a vector)."
    },
    {
        "question": "Why can you not use the same data to build a model and evaluate its performance?",
        "ground_truth": "Because the model can always simply remember the whole training set, yielding a perfect prediction score. This remembering does not indicate whether the model will generalize well to new, unseen data."
    },
    {
        "question": "What is the standard rule of thumb for splitting datasets into training and testing portions using scikit-learn's train_test_split?",
        "ground_truth": "The standard rule of thumb extracts 75% of the rows in the data as the training set, and declares the remaining 25% of the data as the test set."
    },
    {
        "question": "Why does train_test_split shuffle data before performing a train-test split?",
        "ground_truth": "To ensure that the test set represents all classes. If data points are sorted by their label, taking the last 25% without shuffling could result in a test set containing only a single class, which fails to show how well the model generalizes."
    },
    {
        "question": "What purpose does the random_state parameter serve in functions like train_test_split?",
        "ground_truth": "It provides a fixed seed to the pseudorandom number generator, making the data shuffle outcome completely deterministic so that running the function multiple times yields the exact same output splits."
    },
    {
        "question": "What basic steps are required to evaluate a k-Nearest Neighbors classifier model using scikit-learn?",
        "ground_truth": "Instantiate the KNeighborsClassifier class with the desired parameters, call the fit method passing the training data (X_train) and labels (y_train), and then evaluate it on the test set using the score method."
    },

    # ==========================================
    # CH. 2 SUPERVISED LEARNING (~20 QAs)
    # ==========================================
    {
        "question": "What is the primary objective of a classification task in supervised learning?",
        "ground_truth": "The goal in classification is to predict a discrete class label, which is a choice chosen from a predefined list of discrete categorical possibilities."
    },
    {
        "question": "How can you easily distinguish between classification and regression tasks based on their outputs?",
        "ground_truth": "By checking whether there is some kind of continuity in the output. If there is continuity between possible outcomes (e.g., predicting an amount or a continuous number), it is a regression problem; otherwise, it is a classification problem."
    },
    {
        "question": "What does it mean if a machine learning model is able to 'generalize'?",
        "ground_truth": "It means the model built on the training data is able to make accurate predictions on new, unseen data that has the same characteristics as the training set."
    },
    {
        "question": "What is overfitting in supervised machine learning?",
        "ground_truth": "Overfitting occurs when you fit a model too closely to the particularities and noise of the training set, obtaining a model that works perfectly on the training data but fails to generalize well to new data."
    },
    {
        "question": "What is underfitting in supervised machine learning?",
        "ground_truth": "Underfitting occurs when your model choice is too simple to capture all the aspects, variations, and dependencies present in the data, causing it to perform poorly even on the training set."
    },
    {
        "question": "How does the size of your training dataset relate to acceptable model complexity?",
        "ground_truth": "Model complexity is intimately tied to the variation of inputs in the dataset. Collecting a larger variety of data points allows you to build more complex models without overfitting."
    },
    {
        "question": "How does the k-Nearest Neighbors algorithm predict a target value for a new data point?",
        "ground_truth": "The algorithm finds the closest data points ('nearest neighbors') in the stored training dataset. For classification, it assigns the majority class via voting; for regression, it takes the average or mean of the neighbors' target values."
    },
    {
        "question": "What is a decision boundary?",
        "ground_truth": "The decision boundary is the divide in feature space between where a classification algorithm assigns one class versus where it assigns another class."
    },
    {
        "question": "How does changing the number of neighbors (k) in a k-NN model impact its complexity?",
        "ground_truth": "Using a small number of neighbors corresponds to high model complexity (ragged decision boundaries following the training data closely), whereas using many neighbors corresponds to low model complexity (smoother decision boundaries)."
    },
    {
        "question": "What is the general prediction formula for a linear regression model?",
        "ground_truth": "The general prediction formula is: ŷ = w[0] * x[0] + w[1] * x[1] + ... + w[p] * x[p] + b, where x represent features, w represent slopes/weights, b represents the offset/intercept, and ŷ is the prediction."
    },
    {
        "question": "What attribute naming convention does scikit-learn use for properties derived from training data?",
        "ground_truth": "scikit-learn stores anything derived from the training data in public attributes that end with a trailing underscore (e.g., coef_ and intercept_) to separate them from parameters set by the user."
    },
    {
        "question": "What is ordinary least squares (OLS) linear regression?",
        "ground_truth": "It is the simplest classic linear regression method that finds parameters w and b by minimizing the mean squared error (the sum of squared differences between predictions and true values) on the training set. It has no internal parameters to control complexity."
    },
    {
        "question": "What is Ridge regression, and how does it prevent overfitting?",
        "ground_truth": "Ridge regression is a linear model that imposes an additional constraint on the coefficients: we want the magnitude of w to be as small as possible (close to zero). This constraint is called L2 regularization."
    },
    {
        "question": "How does the alpha parameter impact a Ridge regression model?",
        "ground_truth": "The alpha parameter controls the trade-off between the simplicity of the model and its training set performance. Increasing alpha forces coefficients more toward zero (decreasing complexity), while decreasing alpha allows the coefficients to be less restricted."
    },
    {
        "question": "What is Lasso regression, and how does its regularization differ from Ridge regression?",
        "ground_truth": "Lasso uses L1 regularization to restrict coefficients to be close to zero. Unlike Ridge, L1 regularization pushes some coefficients to be exactly zero, effectively ignoring those features entirely as an automatic feature selection mechanism."
    },
    {
        "question": "What are the two most common linear classification algorithms?",
        "ground_truth": "The two most common linear classification algorithms are logistic regression (LogisticRegression) and linear support vector machines (LinearSVC)."
    },
    {
        "question": "What parameter controls regularization strength in LogisticRegression and LinearSVC, and how does it work?",
        "ground_truth": "The trade-off parameter is C. Higher values of C correspond to less regularization (trying to fit each individual training point correctly), while lower values of C put more emphasis on finding a coefficient vector close to zero."
    },
    {
        "question": "How does a binary linear classifier extend to a multiclass setting using the one-vs.-rest approach?",
        "ground_truth": "It learns a distinct binary model for each class to separate it from all other classes. To make a prediction, all binary classifiers run on a test point, and the class with the highest confidence score wins."
    },
    {
        "question": "How do Naive Bayes classifiers achieve high computational efficiency?",
        "ground_truth": "They learn parameters by looking at each feature individually, collecting simple per-class statistics without modeling correlations between features."
    },
    {
        "question": "What are the two primary strategies used to prevent decision trees from overfitting?",
        "ground_truth": "The two strategies are pre-pruning (stopping the tree creation early by limiting max depth, maximum leaf nodes, or minimum samples per leaf) and post-pruning (building the tree fully, then removing or collapsing uninformative nodes)."
    },

    # ==========================================
    # CH. 3 UNSUPERVISED LEARNING & PREPROCESSING (~15 QAs)
    # ==========================================
    {
        "question": "What are the two main types of unsupervised learning discussed in the book?",
        "ground_truth": "The two main types are unsupervised transformations (creating a new representation of the data that is easier to understand or process) and clustering algorithms (partitioning data into distinct groups of similar items)."
    },
    {
        "question": "What is a major challenge when working with unsupervised learning algorithms?",
        "ground_truth": "Evaluating whether the algorithm learned something useful. Because the data does not contain any label information, we don't know what the right output should look like, often requiring manual visual inspection."
    },
    {
        "question": "How does scikit-learn's StandardScaler preprocess features?",
        "ground_truth": "It ensures that for each feature, the mean is 0 and the variance is 1, bringing all features to the same relative magnitude."
    },
    {
        "question": "What is the difference between StandardScaler and RobustScaler?",
        "ground_truth": "StandardScaler uses the mean and variance to scale features, which can be sensitive to outliers. RobustScaler uses the median and quartiles, making it ignore data points that deviate drastically from the rest."
    },
    {
        "question": "How does scikit-learn's MinMaxScaler transform a dataset?",
        "ground_truth": "It shifts and rescales the data so that all feature values fall exactly between a specified minimum and maximum, which by default is 0 and 1."
    },
    {
        "question": "Why is it critical to use the exact same scaler transformation on both the training set and the test set?",
        "ground_truth": "Because scaling them independently shifts the arrangement of data points arbitrarily, causing test points to move inconsistently relative to what the supervised model learned during training."
    },
    {
        "question": "What is the purpose of the fit_transform method in scikit-learn transformers?",
        "ground_truth": "It fits the transformer model on the dataset and immediately returns the transformed representation in a computationally efficient single-step process."
    },
    {
        "question": "What is Principal Component Analysis (PCA)?",
        "ground_truth": "PCA is an unsupervised method that rotates a dataset to ensure that the new, rotated features are statistically uncorrelated, typically followed by selecting a subset of the most important directions of variance."
    },
    {
        "question": "What are principal components in PCA?",
        "ground_truth": "They are the main directions of maximum variance discovered within the data along which features are most correlated with each other."
    },
    {
        "question": "What is a major downside of utilizing PCA for feature representation?",
        "ground_truth": "The resulting principal axes are often very difficult to interpret because they represent complex linear combinations of all original input features mixed together."
    },
    {
        "question": "What mathematical constraint does Non-Negative Matrix Factorization (NMF) enforce?",
        "ground_truth": "NMF forces both the extracted components and their weights/coefficients to be greater than or equal to zero (non-negative), making it applicable only to data where each original feature is non-negative."
    },
    {
        "question": "For what kind of data structure is NMF particularly helpful?",
        "ground_truth": "It is helpful for data created as the addition or overlay of several independent sources (additive structures), such as audio tracks with multiple people speaking or gene expression data."
    },
    {
        "question": "What is the primary focus of the t-SNE algorithm, and how is it used?",
        "ground_truth": "t-SNE is a manifold learning algorithm focused on data visualization. It maps high-dimensional data down to 2D by preserving local neighborhood distances, keeping close points close together."
    },
    {
        "question": "How does the k-means clustering algorithm iteratively discover clusters?",
        "ground_truth": "It alternates between two steps: assigning each data point to its closest cluster center, and then updating each cluster center to be the mean of all data points assigned to it."
    },
    {
        "question": "What is vector quantization?",
        "ground_truth": "Vector quantization is a view of k-means clustering as a decomposition method where each data point is represented using only a single component given by its closest cluster center."
    },

    # ==========================================
    # CH. 4 FEATURE ENGINEERING (~15 QAs)
    # ==========================================
    {
        "question": "What is feature engineering?",
        "ground_truth": "Feature engineering is the process of choosing or creating the best representation of data features for a particular application, which can have a larger influence on model performance than the exact parameters chosen."
    },
    {
        "question": "What is one-hot encoding (dummy variables)?",
        "ground_truth": "It is a process where a categorical variable is expanded into multiple binary features—one new feature for each unique category value—where only one feature is active (set to 1) at a time."
    },
    {
        "question": "What pandas function is commonly used to automatically convert categorical variables into a one-hot encoding?",
        "ground_truth": "The pandas `get_dummies` function is used. It automatically transforms all columns that have an object type (like strings) or are explicitly marked as categorical."
    },
    {
        "question": "What dangerous mistake should you avoid when preparing your target variable for one-hot encoding?",
        "ground_truth": "You must avoid accidentally including the output variable, or a directly derived property of the output variable, into the feature representation passed to your supervised learning model."
    },
    {
        "question": "Why must you exercise caution when a categorical variable is encoded as integers rather than strings?",
        "ground_truth": "Because pandas `get_dummies` treats all numbers as continuous by default and will skip them unless you explicitly list them in the `columns` parameter or cast them to strings."
    },
    {
        "question": "What is binning (discretization)?",
        "ground_truth": "Binning is a feature engineering technique where a single continuous feature is split across multiple discrete bins to convert it into a categorical feature, making linear models much more flexible."
    },
    {
        "question": "How do tree-based models interact with binned features compared to linear models?",
        "ground_truth": "Binning features generally has no beneficial effect for tree-based models because trees can learn to split data anywhere automatically. However, linear models benefit immensely in expressiveness."
    },
    {
        "question": "What are interaction features?",
        "ground_truth": "Interaction features are product features created by multiplying two distinct features together (such as a bin indicator and the original continuous value) to model combined dependencies."
    },
    {
        "question": "What scikit-learn class is used to automatically generate polynomial and interaction features?",
        "ground_truth": "The `PolynomialFeatures` class from the `sklearn.preprocessing` module."
    },
    {
        "question": "How do univariate nonlinear transformations like log or exp functions help linear models?",
        "ground_truth": "They compress or expand relative scales to make highly skewed, non-symmetrical distributions (like count data) look loosely Gaussian (bell-curved), which linear models can capture much better."
    },
    {
        "question": "What is the core philosophy behind automatic feature selection?",
        "ground_truth": "Reducing the total number of features to only the most useful ones to discard noise, lower model complexity, and build simpler models that generalize better."
    },
    {
        "question": "How does univariate feature selection work?",
        "ground_truth": "It computes whether there is a statistically significant relationship between each individual feature and the target independently, dropping features that fall below a confidence threshold."
    },
    {
        "question": "What is model-based feature selection?",
        "ground_truth": "It uses a separate supervised machine learning model (like a random forest or a linear model with L1 penalty) to judge and rank the absolute importance of all features simultaneously."
    },
    {
        "question": "How does recursive feature elimination (RFE) select features?",
        "ground_truth": "It is an iterative method that starts with all features, builds a model, discards the single least important feature according to that model, and repeats the process until a target number of features remain."
    },
    {
        "question": "Why is integrating expert domain knowledge useful in feature engineering?",
        "ground_truth": "Because it allows you to explicitly augment your data with highly informative features (like holiday markers for predicting flight prices) that a model cannot mathematically infer from standard timestamps alone."
    },

    # ==========================================
    # CH. 5 MODEL EVALUATION (~15 QAs)
    # ==========================================
    {
        "question": "What is k-fold cross-validation?",
        "ground_truth": "It is a statistical evaluation method where data is partitioned into k equal parts (folds). A sequence of k models is trained, each using a different fold as the test set and the remaining k-1 folds as the training set."
    },
    {
        "question": "What is a major advantage of cross-validation over a single train-test split?",
        "ground_truth": "It eliminates the risk of getting an unreliably high or low score due to an exceptionally 'lucky' or 'unlucky' single random split, as every single data point gets to be in the test set exactly once."
    },
    {
        "question": "What is stratified k-fold cross-validation, and why is it used for classification?",
        "ground_truth": "It splits the data such that the exact class proportions are maintained identically within each fold. It prevents standard k-fold from creating a fold containing only one class, which is uninformative for testing."
    },
    {
        "question": "Does cross_val_score return a trained model for future predictions?",
        "ground_truth": "No. Cross-validation does not return a final model. Its sole purpose is to evaluate how well a given algorithm generalizes when trained on a specific dataset."
    },
    {
        "question": "What is grid search?",
        "ground_truth": "Grid search is an automated parameter tuning method where the user defines a grid of specific parameter values to test, and the algorithm evaluates combinations to find the setting with the best performance."
    },
    {
        "question": "Why is using the final test set to perform grid search parameter tuning dangerous?",
        "ground_truth": "Because selecting parameters based on test set scores 'leaks' information into the model. The test set is no longer independent, resulting in overly optimistic generalization estimates."
    },
    {
        "question": "What is a threefold data split?",
        "ground_truth": "Splitting the data into three separate sets: a training set to build the model, a validation set to select optimal parameters, and a strictly isolated test set for final performance evaluation."
    },
    {
        "question": "What does GridSearchCV do automatically after finding the best parameter combination via cross-validation?",
        "ground_truth": "It automatically fits a brand-new final model on the entire training dataset using those optimal parameter settings."
    },
    {
        "question": "What is nested cross-validation?",
        "ground_truth": "It is an evaluation structure featuring an outer cross-validation loop for data splitting and an independent inner cross-validation loop to run grid search parameter selection within each outer split."
    },
    {
        "question": "What is an imbalanced dataset?",
        "ground_truth": "A dataset where one target class is significantly more frequent than the other class (e.g., 99% 'no click' vs 1% 'click')."
    },
    {
        "question": "What is a confusion matrix?",
        "ground_truth": "A comprehensive two-by-two (or n-by-n) table where rows correspond to the true target classes and columns correspond to the predicted classes, explicitly showing correct classifications and misclassifications."
    },
    {
        "question": "Define Precision and Recall mathematically using confusion matrix parameters.",
        "ground_truth": "Precision = TP / (TP + FP), which measures how many predicted positives are truly positive. Recall = TP / (TP + FN), which measures how many true positive samples were successfully captured."
    },
    {
        "question": "What is the f1-score?",
        "ground_truth": "The f1-score is the harmonic mean of precision and recall: F = 2 * (precision * recall) / (precision + recall). It balances both metrics into a single score for imbalanced data evaluation."
    },
    {
        "question": "What does Area Under the Curve (AUC) measure, and what is the AUC score of a random guesser?",
        "ground_truth": "AUC measures the quality of sample ranking across all possible thresholds. A completely random guesser always produces an AUC score of exactly 0.5, regardless of class imbalance."
    },
    {
        "question": "How can you specify a custom metric (like AUC) for hyperparameter tuning inside GridSearchCV?",
        "ground_truth": "By passing the desired metric string name (e.g., scoring='roc_auc') into the `scoring` parameter of the GridSearchCV constructor."
    },

    # ==========================================
    # CH. 6 PIPELINES (~10 QAs)
    # ==========================================
    {
        "question": "Why is performing preprocessing scaling outside of a cross-validation loop a subtle error?",
        "ground_truth": "Because computing scaling properties (like minimum/maximum) across the entire dataset uses data points that will later act as test folds, leaking test information into the training process."
    },
    {
        "question": "What is the primary function of scikit-learn's Pipeline class?",
        "ground_truth": "It encapsulates a chain of multiple data processing transformers together with a final estimator into a single unified object adhering to the standard scikit-learn interface."
    },
    {
        "question": "What is the absolute requirement for all steps in a Pipeline except the final step?",
        "ground_truth": "All steps except the final step must possess a `transform` method so they can generate a new data representation to feed into the next step."
    },
    {
        "question": "What naming syntax must you use to specify hyperparameters for steps inside a Pipeline dictionary during a grid search?",
        "ground_truth": "The syntax requires the exact step string name, followed by a double underscore `__`, followed by the parameter parameter name (e.g., 'svm__C')."
    },
    {
        "question": "How does make_pipeline differ from the standard Pipeline constructor?",
        "ground_truth": "The standard Pipeline requires explicit tuple naming for each step, whereas `make_pipeline` automatically names each step based on a lowercased version of its class name."
    },
    {
        "question": "How does make_pipeline handle naming duplicate steps of the same class type?",
        "ground_truth": "It appends an incremented integer suffix to the lowercase class name (e.g., 'standardscaler-1' and 'standardscaler-2')."
    },
    {
        "question": "How can you access a specific trained step's attributes inside a pipeline object?",
        "ground_truth": "By referencing the step name as a key inside the pipeline's `named_steps` dictionary attribute."
    },
    {
        "question": "How can you access the optimal model found by a GridSearchCV object wrapped around a pipeline?",
        "ground_truth": "By using the `best_estimator_` attribute of the fitted GridSearchCV object, which returns the complete optimized pipeline."
    },
    {
        "question": "Is it possible to optimize preprocessing parameters (like polynomial degree) inside a pipeline using grid search?",
        "ground_truth": "Yes. By defining the parameter prefixed with the step name (e.g., 'polynomialfeatures__degree'), GridSearchCV evaluates those preprocessing variants based on the final estimator's score."
    },
    {
        "question": "How do you tell a pipeline to skip a particular processing step completely during a grid search space sweep?",
        "ground_truth": "By setting the step parameter to the python value `None` within that specific grid dictionary grid space layout."
    },

    # ==========================================
    # CH. 7 TEXT DATA (~10 QAs)
    # ==========================================
    {
        "question": "What is the bag-of-words representation for text processing?",
        "ground_truth": "It is a text representation that discards syntax, grammar, and structural order completely, describing each document solely by counting the frequencies of individual words from a shared vocabulary."
    },
    {
        "question": "What are the three core steps involved in computing a bag-of-words representation?",
        "ground_truth": "1) Tokenization: Splitting text into individual token words. 2) Vocabulary building: Collecting and indexing all unique tokens. 3) Encoding: Counting token frequencies per document."
    },
    {
        "question": "Why are bag-of-words text datasets stored in sparse matrices rather than dense arrays?",
        "ground_truth": "Because any individual document contains only a small fraction of all possible words in the dictionary, storing the massive quantity of zeros in a dense layout would exhaust system memory."
    },
    {
        "question": "What does the min_df parameter control in scikit-learn's CountVectorizer?",
        "ground_truth": "It sets the minimum number of distinct documents a word token must appear in to be included in the vocabulary, eliminating rare words and misspellings to reduce feature space size."
    },
    {
        "question": "What are stopwords?",
        "ground_truth": "Stopwords are words that appear so frequently within a language (such as 'the', 'and', 'is') that they carry no descriptive semantic meaning for text classification tasks."
    },
    {
        "question": "What is the fundamental philosophy of the term frequency-inverse document frequency (tf-idf) rescaling scheme?",
        "ground_truth": "It rewards words that appear frequently within a single query document, but penalizes them if they appear across many documents in the wider corpus, identifying unique, descriptive terms."
    },
    {
        "question": "What are n-grams in natural language processing?",
        "ground_truth": "n-grams are overlapping sequences of tokens appearing adjacent to each other. Sequences of length two are bigrams, length three are trigrams, and length n are n-grams."
    },
    {
        "question": "What is the core benefit of adding bigrams or trigrams to a bag-of-words text representation model?",
        "ground_truth": "It captures local contextual relationships that reverse or modify word meanings, such as understanding the difference between 'bad, not good' and 'good, not bad'."
    },
    {
        "question": "What is the difference between stemming and lemmatization text normalization techniques?",
        "ground_truth": "Stemming applies basic rule-based heuristics to aggressively chop suffixes off words to extract a stem. Lemmatization utilizes a verified morphological dictionary and sentence context to revert words to their standardized base form (lemma)."
    },
    {
        "question": "What does Latent Dirichlet Allocation (LDA) discover when applied to an unlabelled text corpus?",
        "ground_truth": "LDA is an unsupervised topic modeling algorithm that discovers groups of words that frequently co-occur together (topics), representing each document as a proportional mixture of those topics."
    },

    # ==========================================
    # CH. 8 WRAPPING UP (~5 QAs)
    # ==========================================
    {
        "question": "What critical financial/logistical analysis should always be performed before building a machine learning system?",
        "ground_truth": "Calculate the expected business impact by asking 'What if I built the perfect model?' to ensure the financial or logistical savings justify the time and resource investment of developing an algorithm."
    },
    {
        "question": "What is the purpose of implementing a 'human-in-the-loop' workflow architecture?",
        "ground_truth": "Rerouting highly complex, uncertain predictions to human experts while letting the machine learning system safely automate the high volume of simple, clear-cut cases."
    },
    {
        "question": "What does the Google paper 'Machine Learning: The High Interest Credit Card of Technical Debt' warn practitioners about?",
        "ground_truth": "It highlights that deploying complex machine learning components creates hidden downstream costs, maintenance burdens, and structural dependencies, warning that pipeline simplicity should beat unneeded complexity."
    },
    {
        "question": "What is the difference between offline evaluation and online live testing?",
        "ground_truth": "Offline evaluation evaluates models retrospectively on historical datasets using metrics like accuracy or AUC. Online testing uses live user studies (such as A/B testing) to measure the real-world behavioral changes of users interacting with a live service."
    },
    {
        "question": "What basic scikit-learn classes should you inherit from to build a custom compatible transformer class?",
        "ground_truth": "You should inherit from `BaseEstimator` and `TransformerMixin`, and then explicitly implement the required internal `fit` and `transform` methods."
    }
]