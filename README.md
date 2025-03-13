





COUNTERFACTUAL EXPLANATIONS











DISCUSSION POINTS
	What are Counterfactual Explanations and why are they important?
	What are the key methods for generating Counterfactual Explanations?
	What fundamental requirements should Counterfactual Explanations fulfil?
	What is DICE (Diverse Counterfactual Explanations)?
	What are the available methods in DICE for generating Counterfactuals?
	What unique features or special functionalities does DICE offer?
	How to choose the most suitable method for generating Counterfactual Explanations?




















What are Counterfactual Explanations and why are they important?________________________________________

A counterfactual explanation for a model’s prediction is a set of minimal changes to an input instance that would result in a different predicted outcome. It is widely used in fairness, interpretability, and decision-making contexts.
For example, if a loan application is denied, a counterfactual explanation might state:
"If your income had been $50,000 instead of $40,000, your loan would have been approved."
Instead of explaining why a decision was made, counterfactuals explain what changes are needed to alter the decision.




What are the key methods for generating Counterfactual Explanations?________________________________________

Counterfactual explanations can be generated using various techniques, each suited to different model types and data structures. The primary methods include:

1.  Optimization-Based Methods
	Frame counterfactual search as an optimization problem to minimize feature changes while achieving the desired outcome.
	Example methods: DiCE, CEM
2.  Prototype & Example-Based Methods
	Identify representative data points (prototypes) or find nearest data points that match the desired outcome.
	Example methods: ProtoDash, Nearest Neighbour Counterfactuals
3.  Decision Boundary Methods
	Move data points across the model's decision boundary with minimal feature changes.
	Example methods: MACE, FACE
4.  Heuristic & Rule-Based Methods
	Use predefined rules, constraints, or iterative search for simpler yet interpretable counterfactuals.
	Example methods: Growing Spheres, Anchors
5.  Model-Specific Methods
	Tailored for specific model types, such as using path analysis for tree models or gradient-based methods for neural networks.
	Example methods: TreePath, Gradient-Based Counterfactuals




What fundamental requirements should Counterfactual Explanations fulfil?________________________________________

To ensure counterfactual explanations are meaningful, they should meet the following key criteria:

Actionable: The suggested changes should be practical (e.g., “increase income” is actionable, but “change race” is not).
Sparse: Only a few features should be modified.
Proximate: The modified instance should be close to the original.
Diverse: Providing multiple counterfactual paths to change the outcome is useful.




What is DICE (Diverse Counterfactual Explanations)?________________________________________

DiCE is a model-agnostic method that generates multiple, diverse counterfactuals by formulating counterfactual search as an optimization problem. Along with balancing all the key objectives:
	Proximity: Ensures the counterfactual is close to the original instance.
	Diversity: Produces multiple distinct counterfactuals instead of just one, providing more actionable choices.
	Validity: Ensures the generated counterfactual successfully changes the model’s prediction to the desired class.
	Sparsity: Limits the number of features that change for simpler and more interpretable results.
	Feasibility: Enforces constraints to ensure the suggested changes are realistic.

How DiCE can be implemented?________________________________________

DiCE getting started - DiCE_getting_started_feasible.html
DiCE model agnostic CFs  - DiCE_model_agnostic_CFs.html




What are the available methods in DiCE for generating Counterfactuals?________________________________________

1. Gradient-Based Search: This method leverages gradient descent to iteratively adjust feature values toward a desired prediction. It is particularly effective for differentiable models such as neural networks and logistic regression. However, it is less suitable for non-differentiable models or data with categorical features that exhibit discrete changes.

Gradient-Based Search is an optimization-driven approach that leverages gradient descent to iteratively modify feature values to achieve the desired model prediction. This method is particularly effective for differentiable models like neural networks and logistic regression, where gradients can be computed directly. The search starts with the original data point and adjusts feature values step-by-step in the direction that minimizes the prediction error relative to the target outcome.

The objective of Gradient-Based Search can be represented by the following loss function:
 

	C(x): The set of generated counterfactuals that minimize the combined objective.
	yloss(f(ci, y): The prediction loss — a measure of how close the model’s predicted output for counterfactual ci is to the desired target y.
	dist(ci,x): The proximity term — ensuring counterfactuals remain close to the original instance xxx to maintain realistic changes.
	dpp_diversity(c1,…,ck) — The diversity term — encourages the generated counterfactuals to differ from each other, improving the range of possible actionable insights.
	λ1 and λ2: Regularization parameters that balance the trade-off between proximity and diversity.


where,
yloss(f(ci), y)  =  hinдe_yloss  =  max(0,1 - z * logit(f (c)))
dist_cont(c, x) = 1/dcont  ∑_(p=1)^dcont▒|c^p  – c^p |/MADp  for continuous features where dcont is the number of continuous variables and MADp is the median absolute deviation for the p-th continuous variable.
and
dist_cat(c, x) = 1/dcat ∑_(p=1)^dcat▒〖I(c^p  ≠〖 x〗^p   )〗 for categorical features where dcat is the number of categorical variables.
dist(ci,x) = ℓ1-distance (optionally weighted by a user-provided custom weight for each feature).
dpp_diversity(c1,…,ck)  =  det(K) where Ki,j = 1/(1+ dist(ci ,cj))

Gradient-Based Search is efficient in continuous data spaces but struggles with categorical features that exhibit discrete jumps, as gradient updates may suggest non-feasible changes.
Average Time Complexity for K Counterfactual: O(K*I*d) — where I is the number of gradient steps (iterations) and d is the number of features.


2. KD-Tree Search: This method efficiently identifies nearest neighbours in the feature space, making it well-suited for tree-based models and datasets with low to moderate dimensionality. Its performance deteriorates in very high-dimensional spaces due to increased computational complexity.

K-D Tree (also called as K-Dimensional Tree) is a binary search tree where data in each node is a K-Dimensional point in space.
Let us number the planes as 0, 1, 2, … (K – 1). A point (node) at depth D will have A aligned plane where A is calculated as: A = D mod K
We can understand the creation of a K-D tree with an example of 2-D Tree.
Consider following points in a 2-D plane: (3, 6), (17, 15), (13, 15), (6, 12), (9, 1), (2, 7), (10, 19)
	Insert (3, 6): Since tree is empty, make it the root node.
	Insert (17, 15): Compare it with root node point. Since root node is X-aligned, the X-coordinate value will be compared to determine if it lies in the right subtree or in the left subtree. This point will be Y-aligned.
	Insert (13, 15): X-value of this point is greater than X-value of point in root node. So, this will lie in the right subtree of (3, 6). Again, Compare Y-value of this point with the Y-value of point (17, 15) (Why?). Since, they are equal, this point will lie in the right subtree of (17, 15). This point will be X-aligned.
	Insert (6, 12): X-value of this point is greater than X-value of point in root node. So, this will lie in the right subtree of (3, 6). Again, Compare Y-value of this point with the Y-value of point (17, 15) (Why?). Since, 12 < 15, this point will lie in the left subtree of (17, 15). This point will be X-aligned.
	Insert (9, 1): Similarly, this point will lie in the right of (6, 12).
	Insert (2, 7): Similarly, this point will lie in the left of (3, 6).
	Insert (10, 19): Similarly, this point will lie in the left of (13, 15).
                     
Search in a K-D Tree for nearest point, works on the principal of search in Binary Search Tree.
Average Time Complexity for K Counterfactual: O(K*log(N)*F) – where N is number of data points and F is number of features.
	

3. Genetic Algorithm: This method evolves counterfactual candidates through iterative mutation and crossover processes, making it effective for exploring complex, non-linear decision boundaries. While robust, it can be computationally intensive and requires careful parameter tuning to ensure optimal performance.

Genetic Algorithm mimics the process of natural selection to generate counterfactual explanations by evolving a population of candidate solutions over multiple generations. Each generation undergoes mutation (random feature alterations) and crossover (combining features from different candidates) to enhance diversity and improve solution quality. Fitness functions evaluate candidates based on proximity to the desired outcome while maintaining feature constraints. This method is particularly effective for exploring complex, non-linear decision boundaries in black-box models. However, its performance heavily depends on parameter tuning, such as mutation rates, population size, and the number of generations, and can be computationally intensive for large datasets or high-dimensional feature spaces.

Average Time Complexity for K Counterfactual: O(K*G*P*d) — where G is the number of generations, P is the population size, and d is the number of features.


4. Randomized Search: This method explores the feature space by randomly sampling candidate instances until valid counterfactuals are identified. It is flexible for use with black-box models but may become inefficient in high-dimensional data where extensive sampling is required.

In Randomized Search, candidate counterfactuals are generated by randomly sampling feature values across the input space and evaluating their outcomes against the desired prediction. This method does not rely on model internals, making it highly flexible for black-box models. The sampling process continues iteratively until sufficient valid counterfactuals are identified. While effective in simpler data settings, its efficiency decreases significantly in high-dimensional feature spaces, as the probability of randomly locating meaningful counterfactuals diminishes, often requiring extensive sampling for reliable results.

Average Time Complexity for K Counterfactual: O(K*S*d) — where S is the number of sampled points and d is the number of features.




What unique features or special functionalities does DICE offer?________________________________________

DiCE offers several unique features and special functionalities that enhance its flexibility and applicability in model explainability:
	Proximity and Diversity Tradeoff:
	DiCE allows users to control the balance between proximity and diversity using attributes proximity weight and diversity weight parameters.
	Increasing proximity weight ensures closer instances, while higher diversity weight promotes varied counterfactuals.
	Default values for proximity weight and diversity weight are 0.5 and 1 respectively.
	Feature Weighting:
	DiCE provides the functionality that can be used to assign different weights to features based on their significance or likelihood of change.
	Feature Importance (Local & Global):
	DiCE can integrate local feature importance to focus on key influential features for specific instances.
	It also supports global feature importance analysis, helping identify overall feature impact across the dataset.
	DiCE Values with Pre-trained Models:
	DiCE efficiently works with pre-trained models as well, allowing users to generate counterfactual explanations without retraining the model.
	This makes DiCE flexible and applicable for both real-time and batch predictions in production environments.
	Method KD Trees can’t be use in this case, as it requires training data as well.





How to choose the most suitable method for generating Counterfactual Explanations? ________________________________________

Model Type	Recommended Method

	Reason
Linear Models (e.g., Logistic Regression, Linear Regression)	Gradient-Based Search	Exploits the model's differentiable nature for efficient counterfactual search.
Tree-Based Models (e.g., Decision Tree, Random Forest, XGBoost)	KD-Tree Search or Genetic Algorithm	KD-Tree efficiently identifies nearest neighbours; Genetic Algorithm excels in exploring non-linear spaces.
Neural Networks (e.g., 
Gradient based model, MLP, CNN, RNN, Transformer models)	Gradient-Based Search 	Gradient methods efficiently compute meaningful counterfactuals.

Support Vector Machines (SVMs)	Gradient-Based Search or Genetic Algorithm	Gradient-based methods work well with kernel-based models; Genetic Algorithms improve robustness in non-linear data.
Black-Box Models 	Randomized Sampling or Genetic Algorithm
	Suitable for models without gradient access; provides flexibility in exploring the feature space.

Note: 
	Gradient-Based Search is best suited for models that rely on gradient-based optimization techniques.
	KD-Tree Search is less effective for models trained on large datasets or pre-trained models as it requires training data for tree generation.
	DiCE works with all sklearn, TensorFlow and PyTorch based models.

