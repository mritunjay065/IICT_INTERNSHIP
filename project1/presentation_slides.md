# Presentation Slides: AI-Powered Fake News Detection
**Course/Program:** Summer Internship Program in AI & ML 2026  
**Project Title:** Fake News Classification Using Custom ML Pipeline  

---

## Slide 1: Title Slide
- **Title:** AI-Powered Fake News Detection Using Text Classification from Scratch
- **Objective:** Design, build, and benchmark an end-to-end ML classification pipeline without pre-built libraries.
- **Presenter:** Student Intern

---

## Slide 2: Problem Statement & Significance
- **Problem:** Scale of digital information makes manual fact-checking impossible.
- **Goal:** Classify news articles automatically as Real (Label 0) or Fake (Label 1).
- **Impact:** Mitigate spread of misinformation on social networks and search systems.

---

## Slide 3: 30-Day Workflow Accomplishments
- **Week 1:** Manual preprocessing, tokenization, stopword cleaning, punctuation stripping.
- **Week 2:** Explored dataset distributions (EDA) and custom TF-IDF feature extraction.
- **Week 3:** Built KNN, Logistic Regression, Random Forest, and a MLP Neural Network from scratch in NumPy.
- **Week 4:** Evaluated models, generated performance charts, and compiled IEEE project report.

---

## Slide 4: Methodology - Preprocessing & Feature Extraction
- **Text Cleaning:** Manual lowercase conversion, non-alphanumeric filtering via regex, and tokenization.
- **Stopwords:** Hardcoded filtering list of common articles, prepositions, and pronouns.
- **TF-IDF from Scratch:** Term-frequency paired with smoothed inverse document frequency ($\text{IDF}(t) = \ln(\frac{1+N}{1+\text{DF}}) + 1$) and L2 unit-norm regularization.

---

## Slide 5: Algorithms Built From Scratch (NumPy)
1. **KNN:** Cosine distance similarity majority vote ($k=5$).
2. **Logistic Regression:** Sigmoid activation, binary cross-entropy loss, and gradient descent optimization.
3. **Random Forest:** Ensemble of Gini-impurity-based decision trees with bootstrap sample bagging.
4. **Simple Neural Network:** 1-hidden-layer MLP with ReLU & Sigmoid activations trained using backpropagation.

---

## Slide 6: Results & Benchmarks
- All models achieved **100% classification accuracy** on test set (240 samples).
- Tested against **scikit-learn** versions; custom models matched baseline benchmarks exactly.
- **Test Confusion Matrix:**
  - True Real: 122 | False Fake: 0
  - False Real: 0 | True Fake: 118

---

## Slide 7: Discussion: Parametric vs. Non-Parametric
- **Parametric (LogReg, MLP):** Summarizes data through set parameters. Fast inference, low memory usage.
- **Non-Parametric (KNN):** Requires keeping all training data in memory. Slower prediction phase.
- **Linguistic Separation:** The synthetic dataset is perfectly separable, leading to the observed 100% accuracy.

---

## Slide 8: Future Scope & Conclusion
- **Conclusion:** Custom pipeline validates mathematical theory behind classification algorithms.
- **Limitations:** Tested on synthetic corpus; real-world articles require deeper semantic handling.
- **Next Steps:**
  - Evaluate on real-world benchmark datasets.
  - Implement advanced optimizers (Adam) and embeddings (BERT).
