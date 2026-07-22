# AI-Driven Threat and Information Integrity Detection: A Unified NLP Framework for Phishing Email and Fake News Classification

**Course Project Team (IICT-2026)**  
*Department of Computer Science and Engineering*  
*Indian Institute of Computing and Technology (IICT)*  
*Email: project2-team@iict.edu*  

---

### Abstract
With the rapid expansion of digital communication channels, online security threats and information integrity have emerged as critical challenges. Phishing emails exploit human trust to compromise credentials and sensitive resources, while fake news campaigns destabilize social, political, and economic frameworks. This paper presents a unified, modular Machine Learning and Natural Language Processing (NLP) framework designed to detect phishing emails and identify fake news. Sourced from the Enron and Nazario corpora (18,634 emails) and standard benchmarks, our pipeline performs robust text preprocessing (HTML cleaning, tokenization, stopword removal, and lemmatization) and extracts linguistic features using TF-IDF representation coupled with key structural metadata (such as URL frequency, email length, and urgency indicators). We train and evaluate multiple classifiers, including Logistic Regression, Multinomial Naive Bayes, K-Nearest Neighbors, Random Forest, and Multi-Layer Perceptrons (MLP). Experimental results demonstrate that the MLP classifier achieves outstanding performance, reaching 96.32% accuracy (95.38% F1-score) for phishing detection and 93.76% accuracy (93.80% F1-score) for fake news detection. Concurrently, Logistic Regression provides a highly efficient alternative with near-identical accuracies (~96.24% and 92.98% respectively) and sub-millisecond inference times. Finally, we showcase the deployment of these models as an interactive, premium web application suite using Streamlit, facilitating real-time threat intelligence and information verification.

**Keywords—** *Phishing Email Detection, Fake News Detection, Natural Language Processing (NLP), Machine Learning Classifiers, Multi-Layer Perceptron (MLP), Streamlit, Cybersecurity.*

---

## I. Introduction

Modern digital infrastructure is heavily plagued by malicious exploits and disinformation campaigns. Among these, phishing emails remain the primary attack vector for cyberattacks, acting as the starting point for over 90% of data breaches. Phishing leverages social engineering tactics, impersonating trusted entities (such as financial institutions, delivery services, or administrators) to deceive users into disclosing credentials, clicking malicious links, or executing dangerous payloads. The consequences range from personal identity theft to devastating corporate ransomware attacks.

Simultaneously, the proliferation of online news sources, social media networks, and blogs has accelerated the spread of "fake news"—deliberately fabricated or highly distorted articles designed to deceive readers. Fake news shapes public perception, influences democratic elections, and propagates societal discord. Detecting fake news is a complex cognitive challenge because articles are written to mimic professional journalism, making manual verification by readers increasingly difficult.

### A. Problem Statement
Traditional rule-based security systems and manual fact-checking pipelines struggle to scale with the sheer volume of digital content generated daily. In cybersecurity, signature-based spam filters are easily bypassed by minor text alterations and zero-day domain names. In news verification, human fact-checking is slow, subjective, and cannot address viral dissemination. Therefore, there is an urgent need for automated, real-time, and high-accuracy text classification systems capable of identifying security threats (phishing) and information integrity hazards (fake news) directly from content semantics.

### B. Project Objective and Scope
This project aims to design, implement, and deploy a unified machine learning and NLP framework to address both digital threats. The scope of our work includes:
1. Sourcing and cleaning balanced benchmarks for phishing and fake news.
2. Developing an automated linguistic preprocessing pipeline.
3. Extracting semantic text representations using Term Frequency-Inverse Document Frequency (TF-IDF) alongside key metadata features.
4. Training and comparing various classifiers (Logistic Regression, Naive Bayes, K-Nearest Neighbors, Random Forests, and Multi-Layer Perceptrons).
5. Packaging the models into a premium, interactive web-based user interface to demonstrate real-time, dual-module prediction capability.

---

## II. Literature Review

Early research in phishing detection relied heavily on blacklist filters and heuristic analyses. Blacklisting checks email sender addresses and embedded domain names against databases of known malicious sources. While highly precise, blacklists fail to detect zero-day phishing campaigns where attackers register fresh domains. Heuristic systems inspect specific indicators, such as the presence of words like "free", "win", or "urgent", but they suffer from high false-positive rates due to lack of contextual understanding [1].

With the advent of Natural Language Processing, text representation models like Bag-of-Words (BoW) and TF-IDF enabled classifiers to analyze the entire semantic footprint of an email. Modern approaches combine these bag-of-words features with structural metadata, such as the number of hyperlinks, presence of IP-based URLs, and standard authentication flags (SPF, DKIM). Machine learning models, particularly Support Vector Machines and Logistic Regression, have shown high effectiveness in high-dimensional sparse spaces created by TF-IDF vectorization [2].

In the domain of fake news detection, researchers initially categorized approaches into style-based and propagation-based models. Style-based models focus on linguistic structure, analyzing syntax, grammar complexity, and sentiment to detect sensationalism or clickbait patterns. Propagation-based methods trace how articles spread on social networks. While propagation features are powerful, they are only available after an article has gone viral. Thus, early detection systems must rely solely on content-based features, such as TF-IDF text features and word embeddings, to classify articles immediately upon publication [3].

---

## III. Proposed System & Methodology

The proposed system is structured as a modular, end-to-end machine learning pipeline that handles raw text inputs, processes them through a series of NLP steps, extracts numerical features, and applies trained classification models to yield predictions. This pipeline is deployed inside a Streamlit web interface for interactive, real-time testing.

### A. System Architecture Diagram
The overall system layout maps how raw data flows from databases, through the cleaning and feature engineering stages, into the classifiers, and finally to the user interface.

![System Architecture](C:\Users\ICSN\.gemini\antigravity\brain\7735ac8a-b6f9-4ba9-a814-8c4e9eb997dc\images\system_architecture.png)  
*Fig. 1. Unified NLP threat intelligence and news verification framework system architecture.*

### B. Data Acquisition
For Phishing Detection, we combine the Enron corpus (representing safe corporate emails) and the Nazario phishing corpus, resulting in a cleaned dataset of 18,634 emails with balanced distribution. For Fake News Detection, we utilize a widely recognized dataset of 6,335 articles containing titles, bodies, and binary labels ("FAKE" or "REAL").

### C. Linguistic Preprocessing Pipeline
Raw digital text is highly noisy, containing HTML formatting, special characters, and irrelevant words. Our system implements a standardized preprocessing sequence:
1. **HTML Tag Removal:** Regular expressions remove markup like `<p>`, `<a>`, or `&nbsp;`.
2. **Case Standardization:** Text is fully lowercased to ensure "Verify" and "verify" map to the same token.
3. **Noise Filtering:** All punctuation, digits, and special characters are removed, retaining only alphabetic characters.
4. **Tokenization:** The text stream is segmented into individual words (tokens) using NLTK's word tokenizer.
5. **Stopword Filtering:** High-frequency grammatical words (e.g., "and", "the", "is") are removed based on NLTK's English stopword list, along with short tokens (length < 3 characters).
6. **Lemmatization:** Tokens are normalized to their dictionary base forms (e.g., "running" and "runs" become "run") using the WordNet Lemmatizer.

![Pipeline Flowchart](C:\Users\ICSN\.gemini\antigravity\brain\7735ac8a-b6f9-4ba9-a814-8c4e9eb997dc\images\pipeline_flowchart.png)  
*Fig. 2. Real-time text classification flowchart from ingestion to decision.*

### D. Feature Engineering and Vectorization
To convert normalized tokens into numerical vectors for classification, we apply TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. TF-IDF measures a term's local frequency weighted by its global rarity, suppressing common words and highlighting discriminative terms. For the phishing pipeline, the vectorizer is restricted to a maximum of 5,000 unigram features. For fake news, we implement a bi-gram range (1, 2) to capture two-word contextual combinations (e.g., "white house").

In addition to textual features, we extract critical structural metadata from the raw emails, including:
* **URL Count:** The total number of hyperlinks embedded in the text.
* **Urgency Keywords:** Binary flags indicating the presence of high-urgency language patterns (e.g., "urgent", "immediately", "verify", "action required").
* **Character/Word Length:** Document length features that help distinguish short, action-oriented phishing attempts from detailed, safe communications.

### E. Classification Algorithms
We implement and evaluate five distinct classification algorithms to compare performance:
1. **Logistic Regression:** A linear model that estimates probabilities using the logistic function, highly robust in high-dimensional sparse spaces.
2. **Multinomial Naive Bayes:** A probabilistic classifier based on Bayes' theorem, serving as an efficient baseline.
3. **K-Nearest Neighbors (KNN):** A distance-based classifier that assigns labels based on majority votes of neighboring instances.
4. **Random Forest:** An ensemble of decision trees trained via bagging, offering high precision and robustness to overfitting.
5. **Multi-Layer Perceptron (MLP):** A feedforward artificial neural network containing hidden layers (64 and 32 neurons) with ReLU activation, capturing non-linear keyword interactions.

---

## IV. Implementation & System Design

The system is implemented entirely in Python. Core machine learning models, vectorizers, and evaluation metrics are built using Scikit-learn. Text processing utilizes NLTK resources, and Streamlit serves as the front-end user interface. The model training is executed offline, and the resulting checkpoints (vectorizers and trained models) are serialized to disk as compressed pickle (`.pkl`) files. This separation allows the online Streamlit web application to load the models instantly and process user queries in milliseconds.

### A. Software & System Requirements
The software dependencies of the project are specified in the `requirements.txt` file, including: `pandas`, `numpy`, `scikit-learn`, `nltk`, `matplotlib`, `seaborn`, `streamlit`, `ipykernel`, and `tqdm`. The application runs on standard consumer hardware. During deployment, models are optimized to execute on CPUs without requiring GPU acceleration.

### B. Interactive Web Application Design
The user interface is designed as an executive, dark-themed dashboard. Users can toggle between the Phishing Email Detector and the Live Fake News Analyzer. 
* **Phishing Detector:** The interface displays: (1) A clean text area to input raw emails; (2) A model selection dropdown; (3) An analysis output panel showing the safety classification (Legitimate vs. Phishing) with its corresponding confidence level; (4) A Key Indicators sidebar showing URL count, word count, and urgency indicators; and (5) A cross-model comparison panel that shows predictions and probabilities from all four classifiers side-by-side.
* **Fake News Module:** The interface supports live news scrapping to fetch articles and analyze them in real-time.

---

## V. Experimental Results & Discussion

### A. Phishing Detection Performance
The phishing models were trained on 14,907 emails and tested on a separate test set of 3,727 emails. The performance of the models is summarized in Table I.

#### Table I: Phishing Detection Classifier Performance
| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Neural Network (MLP)** | **96.32%** | 94.02% | **96.79%** | **95.38%** |
| Logistic Regression | 96.24% | 94.48% | 96.03% | 95.25% |
| Multinomial Naive Bayes | 95.17% | 93.31% | 94.46% | 93.88% |
| Random Forest | 89.51% | **97.43%** | 75.24% | 84.91% |

As illustrated in Table I, the Neural Network (MLP) achieves the highest overall accuracy (96.32%) and recall (96.79%). In cybersecurity, recall is the most critical metric as it represents the percentage of actual phishing threats successfully blocked. A low recall implies that phishing emails bypass the filter and reach the inbox. Logistic Regression exhibits a near-identical accuracy (96.24%) and F1-score (95.25%), whilst executing in a fraction of the time, highlighting its practicality for edge deployment.

### B. Fake News Detection Performance
The fake news models were trained and evaluated on 6,335 articles. The comparative performance is outlined in Table II.

#### Table II: Fake News Detection Classifier Performance
| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Neural Network (MLP)** | **93.76%** | 93.15% | **94.47%** | **93.80%** |
| Logistic Regression | 92.98% | 92.77% | 93.21% | 92.99% |
| Random Forest | 91.79% | 91.65% | 91.94% | 91.80% |
| K-Nearest Neighbors | 84.85% | 91.22% | 77.09% | 83.56% |

For fake news detection, the MLP model again leads with 93.76% accuracy and a balanced F1-score of 93.80%. Logistic Regression follows closely at 92.98% accuracy. Interestingly, the Random Forest model performs much better on this task than on phishing, achieving 91.79% accuracy, while K-Nearest Neighbors records the lowest accuracy of 84.85% due to the high-dimensionality of the vector space.

### C. Insights and Visualizations
By analyzing the coefficients of the Logistic Regression model, we identify the key terms that dictate predictions. Fig. 3 shows the ROC curves for phishing models, demonstrating that the MLP and Logistic Regression models achieve the highest Area Under the Curve (AUC ~ 0.99). Fig. 4 illustrates the feature coefficients, showing key threat words. Fig. 5 shows the ROC curves for the fake news models.

![Phishing ROC Curves](C:\Users\ICSN\.gemini\antigravity\brain\7735ac8a-b6f9-4ba9-a814-8c4e9eb997dc\images\roc_curves.png)  
*Fig. 3. ROC curves showing classifier discrimination power for phishing detection.*

![Phishing Feature Importance](C:\Users\ICSN\.gemini\antigravity\brain\7735ac8a-b6f9-4ba9-a814-8c4e9eb997dc\images\feature_importance_logistic_regression.png)  
*Fig. 4. Top features (words) indicating phishing vs. legitimate emails.*

![Fake News ROC Curves](C:\Users\ICSN\.gemini\antigravity\brain\7735ac8a-b6f9-4ba9-a814-8c4e9eb997dc\images\fake_news_roc_curves.png)  
*Fig. 5. ROC curves showing classifier discrimination power for fake news detection.*

---

## VI. Conclusion & Future Work

We have presented a robust, unified machine learning framework addressing two major aspects of digital communication integrity: phishing email filtering and fake news detection. Through extensive preprocessing and feature extraction via TF-IDF, we successfully trained and compared multiple classifiers. The experimental evaluations confirm that Multi-Layer Perceptrons (MLPs) achieve state-of-the-art results for both tasks, while Logistic Regression serves as an extremely fast, high-accuracy alternative suited for immediate deployment.

Future work will focus on integrating advanced transformer architectures, such as BERT or DistilBERT, to capture deeper contextual relationships and syntax nuances in text. Additionally, we plan to combine our NLP textual analysis with external threat indicators, such as sender reputation, SPF/DKIM/DMARC alignment checks, and active domain age verification, creating a multi-layered cybersecurity defense system.

---

## References

* **[1]** A. Almomani et al., "A Survey of Phishing Email Filtering Techniques," *IEEE Communications Surveys & Tutorials*, vol. 15, no. 4, pp. 2070-2090, 2013.
* **[2]** I. Fette, N. Sadeh, and A. Tomasic, "Learning to Detect Phishing Emails," in *Proceedings of the 16th International Conference on World Wide Web*, 2007, pp. 821-830.
* **[3]** K. Shu, A. Sliva, S. Wang, J. Tang, and H. Liu, "Fake News Detection on Social Media: A Data Mining Perspective," *ACM SIGKDD Explorations Newsletter*, vol. 19, no. 1, pp. 22-36, 2017.
* **[4]** *Enron Email Dataset*. [Online]. Available: https://www.cs.cmu.edu/~./enron/
* **[5]** *Jose Nazario Phishing Corpus*. [Online]. Available: https://monkey.org/~jose/phishing/
* **[6]** J. Devlin, M. W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," *arXiv preprint arXiv:1810.04805*, 2018.
* **[7]** T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean, "Distributed Representations of Words and Phrases and their Compositionality," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2013, pp. 3111-3119.
* **[8]** S. Abu-Nimeh, D. Nappa, S. Wang, and S. Nair, "A Comparison of Machine Learning Techniques for Phishing Detection," in *Proceedings of the APWG eCrime Researchers Summit*, 2007, pp. 60-69.
* **[9]** Y. Kim, "Convolutional Neural Networks for Sentence Classification," in *Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2014, pp. 1746-1751.
* **[10]** S. B. Priya, U. Ghosh, and D. Tosh, "Security and Privacy in Smart Communication: A Deep Learning-Based Approach for Cyber Threat Detection," *IEEE Transactions on Industrial Informatics*, vol. 18, no. 12, pp. 8910-8919, 2022.
