# Cost-Effective LLM Routing Using Text Classification with TensorFlow
![image](https://github.com/user-attachments/assets/996472f9-49a4-4542-b5e2-bc2eb9b590e0)

This project focuses on text classification using TensorFlow, where we explore multiple neural network architectures to classify text inputs effectively. The dataset used for this task has an imbalanced class distribution, which posed additional challenges for model performance.

# What is an LLM Router (LLM Classifier) and why do we need it?
![image](https://github.com/user-attachments/assets/e3b8aa01-8eed-4186-b6cc-96ee804cc75e)

An LLM Router, also known as an LLM Classifier, acts as a traffic controller for incoming requests to a system of Large Language Models (LLMs), which may include a diverse set of both proprietary and open-source models. It analyzes the intent and context of a user's query and directs it to the most appropriate LLM or other specialized tool best suited to handle that specific task. We need LLM Routers because relying on a single, general-purpose LLM for all tasks can be inefficient and yield suboptimal results. Furthermore, the rise of open-source LLMs creates a vast landscape of specialized models, and a router helps manage and leverage this diversity effectively. By intelligently routing requests, we can utilize the strengths of different LLMs, including open-source options, improve response quality, optimize resource utilization, and potentially reduce overall costs by selecting the most cost-effective model for each task. This enables building more complex and efficient applications by chaining together different LLMs and tools, regardless of their source.

## Project Overview
This project aims to build a robust and cost-effective text classification system by leveraging both proprietary and open-source Large Language Models (LLMs). I acknowledge that while proprietary models like GPT-4-Turbo offer superior performance, their high cost can be prohibitive. Conversely, open-source LLMs like Mixtral-8x7b-instruct-v0.1 deliver near-comparable quality at a significantly lower cost (approximately 25 times cheaper).

To optimize both performance and cost-efficiency, I propose implementing an LLM Router (Classifier). This router will intelligently analyze incoming text queries and route them to the most appropriate LLM based on the complexity and topic of the query.

My preliminary analysis suggests that open-source models excel at handling simple queries where answers can be readily extracted from provided context (e.g., "what is num_samples in tune?"). However, they struggle with more complex queries involving reasoning, numerical operations, or code examples (e.g., "if I am inside of a anyscale cluster how do I get my cluster-env-build-id").

The LLM Router will be trained to identify these nuances in queries and direct them accordingly. This approach allows me to harness the strengths of each LLM, maximizing performance while minimizing costs. Furthermore, I will address the inherent imbalance in the dataset by employing techniques like class weighting and architectural experimentation during the classifier training process. This ensures the router's effectiveness in accurately classifying and routing queries for optimal system performance.

To tackle this problem, I built and compared four different neural network architectures:

**DNN (Deep Neural Network)**
**1-Layer Bidirectional LSTM**
**2-Layer Bidirectional LSTM**
**CNN (Convolutional Neural Network)**

By experimenting with these architectures, we aimed to find the model that best balances complexity and performance for our text classification task.

## Dataset
This project utilizes a dataset curated from the Ray project's llm-applications repository (https://github.com/ray-project/llm-applications). To facilitate the development of an LLM Router, the dataset creators hand-annotated a dataset of 1.8k queries based on the suitability of different LLMs for handling them.

### Annotation Strategy
The annotation strategy prioritizes cost-efficiency. The creators adopted a default routing approach where queries are directed to an open-source LLM (label=1) unless they explicitly require the advanced capabilities of a proprietary model like GPT-4 (label=0). This ensures that the more expensive proprietary model is only utilized when absolutely necessary.

### Dataset Characteristics
**Source:** The dataset is derived from real-world user interactions with the Ray framework and its associated tools.
**Format:** The data is structured as a set of question-answer pairs, where each question represents a user query and the corresponding label (0 or 1) indicates the appropriate LLM for handling that query.

**Examples:**

Proprietary LLM (0): {'question': 'if I am inside of a anyscale cluster how do I get my cluster-env-build-id', 'target': 0}
Open-source LLM (1): {'question': 'what is num_samples in tune?', 'target': 1}

**Imbalance:** The dataset is expected to exhibit an imbalance in the distribution of classes, with a potential bias towards queries better suited for open-source LLMs (label=1). This imbalance will be addressed through techniques like class weighting during model training.

**Evaluation:** The performance of the trained LLM Router will be evaluated on a separate test dataset, where the effectiveness of each LLM for a given query has been assessed using an evaluator. This allows for a robust and unbiased assessment of the router's accuracy in directing queries to the appropriate LLM.

## Exploratory Data Analysis (EDA)
Before building our LLM Router, we performed an exploratory data analysis (EDA) to gain a better understanding of the dataset's characteristics. Here are some key observations:

### Class Distribution:
The dataset contains 1801 samples and exhibits a significant class imbalance:
Label 1 (Open-source LLM): 79.5%
Label 0 (Proprietary LLM): 20.5%
This imbalance needs to be addressed during model training to prevent bias towards the majority class.

### Text Length Analysis:
The average text length is 82 words, with a standard deviation of 167 words.
The text length distribution is heavily right-skewed, with most queries being relatively short.
There's a noticeable difference in text length distributions between the two classes, with queries routed to the proprietary LLM (label 0) tending to be longer. This aligns with our expectation that more complex queries might require the advanced capabilities of proprietary models.

### Word Frequencies and Bigrams:
The word cloud highlights terms frequently used in the dataset, mainly related to the "Ray" framework ("ray," "cluster," "tune," "use," "train," "model").
The top bigrams further emphasize common phrases within user queries, providing insights into typical usage patterns and potential areas of difficulty for open-source LLMs.

### Correlation Analysis:
There's a small negative correlation (-0.24) between text length and the target label. This suggests that while longer queries tend to be routed to the proprietary LLM, length alone is not a perfect predictor, and other factors likely contribute to the classification.

### Key Takeaways:
The EDA confirms the dataset's imbalance, requiring strategies like class weighting during model training.
Text length is a potential indicator of query complexity, but further analysis is needed to identify other distinguishing features.
Analyzing word frequencies and bigrams provides valuable insights into the domain-specific language and potential challenges for open-source LLMs, which can guide feature engineering efforts.

## Models Overview
### DNN (Deep Neural Network):

A fully connected neural network.
This model serves as a baseline with a relatively simple structure.

### 1-Layer Bidirectional LSTM:

A single-layer LSTM with bidirectional connections.
This architecture allows the model to capture information from both past and future contexts in the sequence.

### 2-Layer Bidirectional LSTM:

A deeper variant with two bidirectional LSTM layers.
This model captures more complex patterns in the text by stacking layers of LSTMs.
CNN (Convolutional Neural Network):

### A convolutional network applied to the text data.
This architecture is designed to capture local patterns in the text, such as phrases or n-grams.

## Results and Evaluation
This section presents the results of our experiments with four different neural network architectures for building the LLM Router: a Deep Neural Network (DNN), a 1-layer bidirectional LSTM, a 2-layer bidirectional LSTM, and a Convolutional Neural Network (CNN). Each model was trained for 30 epochs with a batch size of 32, using the Adam optimizer and a learning rate of 0.001. We used accuracy as the primary metric for evaluating model performance due to its interpretability in this context. However, we also report precision, recall, F1-score, and AUC to provide a more comprehensive view of each model's strengths and weaknesses.

### DNN:
![image](https://github.com/user-attachments/assets/9f9b186b-9d5b-4b2f-9e1c-3c7d1d0e4bc9)

The DNN model achieved an accuracy of 88.5%, indicating a reasonably good ability to classify queries. However, the training curves (accuracy and loss) show signs of overfitting, with the validation loss increasing towards the end of training. The model's precision (99.4%) is remarkably high, suggesting a strong capability to correctly identify queries suited for the proprietary LLM. However, this comes at the expense of recall (8.9%), indicating that it frequently misclassifies queries that should ideally be routed to the open-source LLM. The resulting F1-score is 0.9375, and the AUC is 0.5872.

### 1-layer Bidirectional LSTM:
![image](https://github.com/user-attachments/assets/54c6726e-332e-470d-897b-51747fd67ac3)

This model yielded an accuracy of 80.4%, demonstrating reasonable performance in distinguishing between queries suited for different LLMs. While not as high as the DNN, this model's training curves exhibit less overfitting, with both training and validation accuracy steadily increasing. The precision is 89.5%, and the recall is 68.4%, indicating a more balanced performance in correctly classifying both types of queries. The F1-score is 0.7873, and the AUC is 0.7072.

### 2-layer Bidirectional LSTM:
![image](https://github.com/user-attachments/assets/7410a220-5292-4b60-b4c7-a17b3fefc14f)

Increasing the LSTM layers to two resulted in an accuracy of 79.9%, slightly lower than the 1-layer LSTM. The training curves, however, still suggest better generalization than the DNN. This model achieved a precision of 89.5% and a recall of 68.8%, showing a similar balance to the 1-layer LSTM. The F1-score is 0.7873, and the AUC is 0.74.

### CNN:
![image](https://github.com/user-attachments/assets/8ecbe73a-6e8f-4100-9ead-5f40a271112a)

The CNN model achieved the highest accuracy among all architectures at 88.9%. The training curves, however, show signs of potential overfitting, similar to the DNN. Despite this, the CNN demonstrated a good balance between precision (99.6%) and recall (81.9%), suggesting its effectiveness in correctly classifying both types of queries. The F1 score is 0.8165 and AUC is 0.82, which are the highest among the four architectures, indicating good discriminative power.

### Overall Analysis:
While the DNN and CNN models achieved higher accuracies, their training curves suggest a tendency towards overfitting the training data. This indicates that these models might not generalize as well to unseen data compared to the LSTM-based models. The 1-layer and 2-layer bidirectional LSTMs provided a more balanced performance with good generalization capabilities, indicated by the smoother training curves.
The choice of the best model depends on the specific priorities of the application. If maximizing accuracy on the provided data is the primary concern, the CNN model might be preferred. However, if generalization and a balance between precision and recall are more important, the 1-layer or 2-layer LSTMs would be better choices. Further experimentation with techniques like regularization and hyperparameter tuning could potentially improve the performance and generalization ability of all models.

## Key Challenges

**1) Imbalanced Dataset:** The dataset has significant class imbalance, which required us to apply techniques like class weighting to ensure that the models do not become biased toward the majority class.
**2) Model Selection:** Choosing the right architecture required balancing between model complexity and performance. Deeper models performed better in capturing complex patterns but took longer to train.

## Future Work

While the current project has explored several deep learning architectures and achieved promising results, there are a few areas that could be investigated further to enhance the performance and applicability of the text classification model:

**Hybrid Model Architectures:** Combining different neural network layers, such as CNNs and LSTMs, could potentially capture more diverse features and improve the overall classification accuracy.

**Transformer-based Models:** Exploring the use of pre-trained transformer-based models, such as BERT or RoBERTa, could leverage their powerful contextual understanding and transfer learning capabilities to further boost the model's performance.

**Explainability and Interpretability:** Incorporating techniques like attention mechanisms or saliency maps could help provide insights into the model's decision-making process, making the predictions more interpretable and transparent.

**Domain-specific Embeddings:** Investigating the use of domain-specific word embeddings, either pre-trained or learned during the training process, could potentially improve the model's understanding of the specialized vocabulary and context within the Routing Dataset.

**Multi-label Classification:** Extending the current binary classification task to a multi-label setting, where a question can be associated with multiple intent labels, could better reflect the real-world complexity of user inquiries.

Exploring these future directions, the project can continue to enhance the text classification capabilities, leading to more accurate and robust systems that can better serve the needs of users.

## Conclusion
This project demonstrates the successful implementation of an LLM Router using TensorFlow, enabling the cost-effective utilization of both proprietary and open-source LLMs for text classification tasks. By experimenting with various neural network architectures and analyzing their performance on a real-world dataset, we gained valuable insights into the challenges and opportunities associated with building such systems. The project's findings pave the way for developing more sophisticated and efficient LLM routing mechanisms, ultimately improving the accessibility, affordability, and performance of LLM-powered applications.
