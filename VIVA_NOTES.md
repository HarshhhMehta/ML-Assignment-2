# Machine Learning Assignment 2: Comprehensive Viva & Defense Master Guide
> **Private Student Preparation Notes â€” Strictly for Oral Exam & Viva Defense**  
> *(Kept in `.gitignore`; not committed or uploaded to submission artifacts).*

---

## Table of Contents
1. [Rapid-Fire 30-Second Cheatsheet](#1-rapid-fire-30-second-cheatsheet)
2. [Dataset & Experimental Protocol](#2-dataset--experimental-protocol)
3. [Question 1: Decision Trees & Bagging Ensembles](#3-question-1-decision-trees--bagging-ensembles)
4. [Question 2: Multi-Class SAMME Boosting](#4-question-2-multi-class-samme-boosting)
5. [Question 3: Empirical Bias-Variance Decomposition](#5-question-3-empirical-bias-variance-decomposition)
6. [Tricky / Trap Questions That Professors Love to Ask](#6-tricky--trap-questions-that-professors-love-to-ask)
7. [Exact Numbers & Key Statistics Reference Table](#7-exact-numbers--key-statistics-reference-table)

---

## 1. Rapid-Fire 30-Second Cheatsheet

| Question | 1-Sentence Punchline Answer |
| :--- | :--- |
| **Why did Bagging beat Boosting?** | In high-D ($4096$), small-N ($240$) regime, Bagging suppresses high tree variance without distorting data; Boosting overconcentrates weight on noisy outlier faces ($23\times$ weight spike). |
| **Why does Bagging survive occlusion?** | Greedy single trees split on central eyes/nose; Bagging forces diverse subspace roots that use forehead, jawline, and hairline to outvote the damaged center. |
| **Why does SAMME have $+\ln(K-1)$?** | It compensates for $K$-class random guessing error ($1 - 1/K$ instead of $1/2$), ensuring $\alpha_t > 0$ whenever accuracy exceeds random chance. |
| **What fraction of samples are distinct in bootstrap?** | Exactly $1 - (1-1/N)^N \xrightarrow{N\to\infty} 1 - e^{-1} \approx 63.21\%$. For $N=240$, theoretical is $151.89$ ($63.29\%$), and empirical was exactly $151.89$. |
| **Does Bagging reduce Bias?** | No. Bagging averages identically distributed models, leaving expectation (bias) almost unchanged while slashing variance by $\frac{1}{B} + \frac{B-1}{B}\rho$. |
| **Can Bagging overfit as $B \to \infty$?** | Never. Breiman proved Bagging variance monotonically converges to a positive asymptote as $B \to \infty$. It cannot overfit by adding more trees. |
| **Can Boosting overfit as $T \to \infty$?** | Yes. Sequential exponential loss minimization will eventually fit noise, margin outliers, and label ambiguities. |

---

## 2. Dataset & Experimental Protocol

### Q: Describe the Olivetti Faces dataset and its characteristics.
- **Identities ($K$):** 40 distinct subjects.
- **Images per subject:** 10 images each (varying lighting, facial expressions, open/closed eyes, glasses).
- **Total samples:** 400 images.
- **Resolution:** $64 \times 64$ grayscale pixels, flattened into $D = 4096$ continuous features normalized to $[0, 1]$.
- **Challenge:** Severe small-sample problem ($N_{\text{train}} = 240 \ll D = 4096$).

### Q: How was the dataset split and how is data leakage prevented?
- **Stratified split without replacement:**
  - **Train:** 6 images per person ($240$ images, $60\%$).
  - **Validation:** 2 images per person ($80$ images, $20\%$).
  - **Test:** 2 images per person ($80$ images, $20\%$).
- **Seed generation:** Seed `26013` deterministically derived from roll number `M26CSA013`.
- **Leakage check:** Asserted that `train_idx \cap val_idx = \emptyset`, `train_idx \cap test_idx = \emptyset`, and `val_idx \cap test_idx = \emptyset`. All feature normalization and hyperparameters were learned solely on the train split.

### Q: What is the center occlusion protocol and why $16 \times 16$?
- A center bounding box $[24:40, 24:40]$ ($16 \times 16 = 256$ pixels out of $4096$, i.e., $6.25\%$ of the face area) is set to zero (`0.0`).
- This specifically covers the bridge of the nose and inner eye regionsâ€”the most discriminative features for facial recognitionâ€”to evaluate how models degrade under localized failure (simulating masks, sunglasses, or sensor dropout).

---

## 3. Question 1: Decision Trees & Bagging Ensembles

### Q: What were the hyperparameter search results for a standalone decision tree?
- Evaluated grid: `max_depth` $\in \{2, 4, 8, 16, \text{None}\}$ and `min_samples_leaf` $\in \{1, 2, 4\}$.
- **Shallow trees ($d=2, 4$):** Suffer from severe underfitting. At $d=2$, only $3$ leaves form (can only separate $\le 3$ classes out of 40), yielding only $7.5\%$ train acc and $5.0\%$ test acc.
- **Deep trees ($d=16$):** Train acc $42.5\%$, test acc $25.0\%$.
- **Unbounded tree ($d=\text{None}$, `min_samples_leaf`$=1$):** Achieves **$100\%$ train acc**, but drops to **$46.25\%$ test acc** (46 leaves, $\sim 5.2$ samples/leaf).
- **Conclusion:** Standalone decision trees suffer from extreme variance in high dimensions. They memorize the 240 training instances but cannot generalize well to unseen test angles.

### Q: Derive the theoretical expected number of unique bootstrap samples.
- Suppose we draw $N$ samples uniformly with replacement from a dataset of size $N$.
- The probability that a specific sample $i$ is NOT chosen in a single draw is $1 - \frac{1}{N}$.
- The probability that sample $i$ is NOT chosen in all $N$ independent draws is:
  $$P(\text{sample } i \text{ omitted}) = \left(1 - \frac{1}{N}\right)^N$$
- Let $I_i$ be an indicator variable that sample $i$ appears at least once:
  $$\mathbb{E}[I_i] = 1 - \left(1 - \frac{1}{N}\right)^N$$
- By linearity of expectation, the expected number of distinct samples $U$ is:
  $$\mathbb{E}[U] = \sum_{i=1}^N \mathbb{E}[I_i] = N \left[1 - \left(1 - \frac{1}{N}\right)^N\right]$$
- **Asymptotics:** As $N \to \infty$, $(1 - 1/N)^N \to e^{-1} \approx 0.367879$.
  $$\lim_{N \to \infty} \frac{\mathbb{E}[U]}{N} = 1 - e^{-1} \approx 63.212\%$$
- **Exact evaluation for $N = 240$:**
  $$\mathbb{E}[U] = 240 \times \left[1 - \left(\frac{239}{240}\right)^{240}\right] = 240 \times (1 - 0.367140) = \mathbf{151.886} \approx 151.89$$
  $$\text{Asymptotic: } 240 \times (1 - e^{-1}) = \mathbf{151.709} \approx 151.71$$
  $$\text{Empirical across } B=200 \text{ resamples}: \mathbf{151.89} \text{ (exact match!)}$$

### Q: What is Out-Of-Bag (OOB) error and how does it behave as $B$ grows?
- For each tree $b$, roughly $36.8\%$ of samples are not selected (out-of-bag).
- For every training instance $x_i$, an ensemble prediction is formed using *only* the subset of trees for which $x_i$ was out-of-bag.
- OOB accuracy acts as a built-in cross-validation score without requiring a separate validation set.
- **Empirical progression across $B$:**
  - $B=1$: OOB Acc $= 21.98\%$ (many samples have 0 OOB trees).
  - $B=10$: OOB Acc $= 41.60\%$.
  - $B=50$: OOB Acc $= 72.08\%$.
  - $B=200$: OOB Acc $= \mathbf{81.25\%}$.
- As $B$ increases, the probability that a sample has zero OOB trees decays as $(1 - e^{-1})^B \to 0$, stabilizing the OOB estimate.

### Q: What is pairwise tree diversity (disagreement) and the vote margin?
- **Disagreement Rate:** The fraction of test samples on which two trees $t_j$ and $t_k$ predict different classes:
  $$\text{Disagreement} = \frac{1}{\binom{B}{2} N_{\text{test}}} \sum_{j < k} \sum_{i=1}^{N_{\text{test}}} \mathbb{I}(\hat{y}_j(x_i) \neq \hat{y}_k(x_i))$$
  In our experiments, disagreement remained consistently around **$42.1\% - 43.0\%$** across all $B \ge 5$. This proves the trees remain highly diverse and do not collapse into redundant clones.
- **Vote Margin:** The difference between the top-1 vote fraction and the runner-up vote fraction:
  $$\text{Margin}(x) = \frac{V_{(1)}(x) - V_{(2)}(x)}{B}$$
  As $B$ grew from 5 to 200, the margin expanded from **$0.6025 \to 0.7057$**, indicating that ensemble consensus becomes significantly more decisive.

### Q: Why is Bagging so robust to center occlusion ($88.75\% \to 80.00\%$)?
- A single decision tree uses greedy top-down recursive partitioning. In facial datasets, central nose/eye pixels have the highest mutual information and are chosen at the root and depth-1 levels. When occluded, the single tree immediately breaks.
- In Bagging ($B=200$), sample perturbation via bootstrapping creates diversity in root splits. Many trees are forced to split on peripheral features (jawlines, hair boundaries, cheekbones, chin contours).
- During inference on occluded faces, the damaged trees voting for wrong classes are easily overwhelmed by the majority of peripheral-feature trees.

---

## 4. Question 2: Multi-Class SAMME Boosting

### Q: What is SAMME and how does its formula differ from binary AdaBoost?
- SAMME stands for **Stagewise Additive Modeling using a Multi-class Exponential loss** (Zhu, Hastie, Rosset, Tibshirani, 2009).
- While binary AdaBoost solves 2-class problems with $\alpha_t = \frac{1}{2}\ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)$, SAMME generalizes to $K > 2$ classes:
  $$\alpha_t = \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right) + \ln(K - 1)$$
- Sample weights update as:
  $$w_i^{(t+1)} = w_i^{(t)} \cdot \exp\left(\alpha_t \cdot \mathbb{I}(y_i \neq c_t(x_i))\right)$$
  normalized such that $\sum_{i=1}^N w_i^{(t+1)} = 1$.

### Q: Why does SAMME add $+\ln(K - 1)$? What is the breakdown condition?
- In binary classification ($K=2$), random guessing has error $\epsilon = 0.5$. When $\epsilon < 0.5$, $\frac{1-\epsilon}{\epsilon} > 1$, making $\ln(\frac{1-\epsilon}{\epsilon}) > 0$.
- In $K$-class classification, random guessing gives error $\epsilon_{\text{random}} = 1 - \frac{1}{K}$.
  - For $K=40$, $\epsilon_{\text{random}} = 1 - 1/40 = 39/40 = 0.975$ ($97.5\%$).
- If we did not add $\ln(K-1)$, any learner with error $> 0.5$ (e.g., $60\%$ error, which is amazing compared to random guess $97.5\%$) would receive a *negative* $\alpha_t$!
- **Proof that $\alpha_t > 0 \iff \epsilon_t < 1 - 1/K$:**
  $$\alpha_t > 0 \iff \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right) + \ln(K - 1) > 0$$
  $$\iff \frac{1 - \epsilon_t}{\epsilon_t} (K - 1) > 1 \iff (1 - \epsilon_t)(K - 1) > \epsilon_t$$
  $$\iff K - 1 - K\epsilon_t + \epsilon_t > \epsilon_t \iff K - 1 > K\epsilon_t \iff \epsilon_t < \frac{K - 1}{K} = 1 - \frac{1}{K}$$
- **Breakdown condition:** If $\epsilon_t \ge 1 - 1/K$, the weak learner is no better than random guessing. $\alpha_t \le 0$, and boosting must terminate immediately.

### Q: Walk through the 3-class toy example calculation step-by-step.
- Dataset: $N = 6$ points, $K = 3$ classes ($y = [0, 0, 1, 1, 2, 2]$).
- **Initial weights:** $w_i^{(1)} = 1/6 \approx 0.1667$ for all $i$.
- **Weak learner predictions:** $[0, 0, 1, 1, 1, 1]$ (misclassifies the two Class 2 samples).
- **Misclassified mask:** $[0, 0, 0, 0, 1, 1]$.
- **Weighted error:**
  $$\epsilon = \frac{0.1667 \times 0 + \dots + 0.1667 \times 1 + 0.1667 \times 1}{1.0} = \frac{2}{6} = 0.3333$$
  Check threshold: $\epsilon = 0.3333 < 1 - 1/3 = 0.6667$ (valid!).
- **Learner weight $\alpha$:**
  $$\alpha = \ln\left(\frac{1 - 0.3333}{0.3333}\right) + \ln(3 - 1) = \ln(2) + \ln(2) = 2\ln(2) \approx \mathbf{1.3863}$$
- **Weight update factor for misclassified points:**
  $$\exp(\alpha) = \exp(1.3863) = 4.0$$
- **Unnormalized weights:**
  - Correct points (4 points): $0.1667 \times e^0 = 0.1667$
  - Incorrect points (2 points): $0.1667 \times 4.0 = 0.6667$
  - Sum of unnormalized weights: $4(0.1667) + 2(0.6667) = 0.6667 + 1.3333 = 2.0$
- **Normalized updated weights $w^{(2)}$:**
  - Correct points: $0.1667 / 2.0 = \mathbf{0.0833}$
  - Incorrect points: $0.6667 / 2.0 = \mathbf{0.3333}$
  - Total sum = $4(0.0833) + 2(0.3333) = \mathbf{1.0000}$.

### Q: Why did depth-1 decision stumps fail completely on the Olivetti dataset?
- A stump ($d=1$) has exactly 1 decision node and 2 leaf nodes.
- A model with 2 leaves can predict at most 2 distinct class labels.
- With $K=40$ classes, assigning 240 samples into only 2 classes leaves at least 38 classes completely unrepresented in any single round.
- Even after 100 rounds of boosting, depth-1 achieved only **$13.75\%$ test accuracy**.
- Boosting requires base learners with enough capacity to partition the multi-class space ($d \ge 4$).

### Q: Why did Bagging ($88.75\%$) crush Boosting ($51.25\%$) on Olivetti?
1. **Outlier Vulnerability:** Boosting exponentially punishes mistakes. Olivetti contains faces with tilted heads, extreme shadow angles, or glasses. These are hard to separate in high dimensions. Boosting assigned massive weight to these outliers (Sample #199, Class 33 grew from initial weight $0.00417 \to 0.09542$, a **$23\times$ inflation**!). The weak learners spent all capacity fitting label noise and atypical images.
2. **Sequential Dependency vs. Independent Averaging:** In high-dimensional small-sample datasets ($D=4096, N=240$), the empirical distribution easily distorts. Boosting destroys the data distribution by concentrating $35\%$ of total probability mass on just 10 faces. Bagging preserves uniform sampling and reduces variance without bias distortion.
3. **Training Time:** Bagging 100 trees took $122.2\text{s}$ (or can be done in parallel across cores in $1.68\text{s}$), whereas sequential boosting took $35.75\text{s}$ strictly serial.

---

## 5. Question 3: Empirical Bias-Variance Decomposition

### Q: What is the Bias-Variance formulation used in this study?
- Under multi-class squared probability error loss:
  $$\text{Loss}(y, \hat{P}) = \frac{1}{2} \sum_{k=1}^K (e_{y, k} - \hat{P}_k)^2$$
  where $e_y$ is the one-hot encoded ground truth vector and $\hat{P}$ is the model's predicted probability distribution.
- Over $R = 30$ independent stratified refits:
  - **Mean prediction vector:** $\bar{P}(x) = \frac{1}{R} \sum_{r=1}^R P_r(x)$
  - **Empirical $\text{Bias}^2$:**
    $$\text{Bias}^2 = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \|\bar{P}(x_i) - e_{y_i}\|_2^2$$
  - **Empirical $\text{Variance}$:**
    $$\text{Variance} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \left[ \frac{1}{R} \sum_{r=1}^R \|P_r(x_i) - \bar{P}(x_i)\|_2^2 \right]$$
  - **Total Loss:** $\text{Expected Loss} = \text{Bias}^2 + \text{Variance}$ (ignoring irreducible Bayes error).

### Q: How did you implement the stratified resampling protocol?
- $R = 30$ resampled training sets were constructed.
- To prevent class imbalance during resampling, sampling was **stratified within each identity**: exactly 6 samples were drawn *with replacement* from the 6 available training images of each of the 40 identities ($6 \times 40 = 240$ samples).
- Every model was refit 30 times on each independent resample and evaluated on the fixed 80-sample test set.

### Q: What is the effect of base tree depth on Bias$^2$ and Variance?
From our exact $R=30$ empirical decomposition:

| Tree Depth | Single Tree $\text{Bias}^2$ | Single Tree Variance | Bagging(100) Variance | Variance Removed | Single Tree Acc | Bagging Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$d=4$ (Shallow)** | $0.8692$ (High) | $0.0706$ (Low) | $0.0131$ | **$81.40\%$** | $9.54\%$ | $26.17\%$ |
| **$d=16$ (Medium)** | $0.6655$ | $0.2290$ | $0.0210$ | **$90.82\%$** | $26.75\%$ | $61.04\%$ |
| **$d=\text{None}$ (Unpruned)** | **$0.3984$** (Lowest) | **$0.6491$** (Highest) | **$0.0255$** | **$96.07\%$** | $47.63\%$ | **$78.67\%$** |

### Q: What is the core insight from the depth experiments?
1. **Unbounded trees have minimal bias:** Deep/unconstrained trees have the representational capacity to fit complex 4096-dimensional decision boundaries, yielding the lowest bias ($0.3984$).
2. **Bagging is a pure variance-reduction engine:** Bagging removes **$96.07\%$** of the variance of unpruned trees ($0.6491 \to 0.0255$), while barely touching bias ($0.3984 \to 0.5107$).
3. **Why shallow trees fail even with Bagging:** Bagging *cannot* reduce bias. If the base tree has high bias ($0.8692$ at $d=4$), Bagging 100 shallow trees still has high bias ($0.8959$) and poor accuracy ($26.17\%$). Thus, **Bagging requires complex, unpruned base estimators**.

---

## 6. Tricky / Trap Questions That Professors Love to Ask

### Q: "Why did you use Bagging instead of Random Forest?"
- **Answer:** Random Forest introduces an additional layer of randomness: at each split node, it selects a random subset of $m = \sqrt{D}$ features (`max_features='sqrt'`).
- In this assignment, Question 1 specifically mandated standard **Bootstrap Aggregation (Bagging)** on full-feature unconstrained decision trees (`max_features=None`, $D=4096$).
- Pure Bagging demonstrates the exact theoretical variance reduction $(1/B + \frac{B-1}{B}\rho)$ arising solely from sample resampling, without confounding it with random subspace projection.

### Q: "Why can't Bagging overfit if we increase $B$ to 10,000?"
- **Answer:** Let the ensemble prediction be $\bar{f}(x) = \frac{1}{B} \sum_{b=1}^B f_b(x)$.
  The variance of the average of $B$ pairwise-correlated random variables (correlation $\rho$, variance $\sigma^2$) is:
  $$\text{Var}(\bar{f}(x)) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
- As $B \to \infty$, $\frac{1-\rho}{B}\sigma^2 \to 0$.
- The variance approaches the irreducible positive asymptote $\rho \sigma^2$.
- Because the expected loss monotonically decreases to a floor as $B \to \infty$, **adding more trees never increases test error or overfits**. It simply wastes CPU cycles.

### Q: "Why does Boosting decrease Bias while Bagging decreases Variance?"
- **Answer:**
  - **Bagging** takes unpruned trees (low bias, high variance) and averages them independently. Averaging independent estimators reduces variance by law of large numbers while maintaining $\mathbb{E}[\bar{f}] = \mathbb{E}[f]$ (bias unchanged).
  - **Boosting** starts with high-bias, low-variance weak estimators (e.g., $d=4$). Each subsequent round fits a learner to the *pseudo-residuals / misclassifications* of the existing committee, creating increasingly non-linear composite boundaries that drive bias down.

### Q: "Why did SAMME test accuracy slightly increase on Masked data ($46.25\% \to 51.25\%$)? Is Boosting better at occlusion?"
- **Answer:** No, that is an artifact of prediction instability near chance level.
- When inspecting prediction confidence, the clean test confidence was only **$0.1208$** and masked was **$0.1068$** (barely above random guessing $1/40 = 0.025$).
- On a small test set of 80 images, a difference of 4 images ($37/80$ vs $41/80$) easily fluctuates when the model is guessing with low margin. In contrast, Bagging maintained high confidence and dropped only from $88.75\% \to 80.00\%$.

### Q: "What splitting criterion did the decision trees use: Gini or Entropy?"
- **Answer:** CART uses Gini Impurity:
  $$I_G(p) = 1 - \sum_{k=1}^K p_k^2$$
  Gini is computationally faster than Entropy (avoids logarithmic operations $\sum p_k \log_2 p_k$) while producing virtually identical split topologies in practice.

---

## 7. Exact Numbers & Key Statistics Reference Table

Memorize these exact figures before entering the viva:

| Metric | Single Tree ($d=\text{None}$) | Bagging ($B=200$) | SAMME Boosting ($d=4, T=100$) |
| :--- | :---: | :---: | :---: |
| **Clean Test Accuracy** | $46.25\%$ ($37/80$) | **$88.75\%$** ($71/80$) | $46.25\%$ ($37/80$) |
| **Masked Test Accuracy** | $46.25\%$ | **$80.00\%$** ($64/80$) | $51.25\%$ |
| **Occlusion Drop** | $0.0\%$ (was already poor) | **$-8.75\%$** (super robust) | $+5.0\%$ (noise artifact) |
| **Test Confidence (Margin)** | $1.000$ (overconfident) | $0.3957$ (well-calibrated) | $0.1208$ (near random guess) |
| **OOB Accuracy** | N/A ($1$ tree) | **$81.25\%$** | N/A |
| **Tree Diversity (Disagreement)**| $0.0\%$ | **$42.11\%$** | N/A |
| **Empirical $\text{Bias}^2$ ($R=30$)** | $0.3984$ | $0.5111$ | $0.8850$ |
| **Empirical $\text{Variance}$ ($R=30$)**| $0.6491$ | **$0.0216$** | $0.0173$ |
| **Total Loss ($\text{Bias}^2 + \text{Var}$)** | $1.0475$ | **$0.5328$** | $0.9023$ |
| **Variance Removed by Bagging** | Baseline | **$96.07\%$** (unbounded) | N/A |
| **Training Time ($100$ learners)** | $0.08\text{s}$ | $122.2\text{s}$ (serial) / $1.68\text{s}$ (par) | $35.75\text{s}$ (serial) |

### Key Dataset & Bootstrap Statistics
- **Total Images:** 400 ($64 \times 64$ pixels = 4096 features).
- **Split Breakdown:** 240 Train, 80 Validation, 80 Test.
- **Occlusion Region:** Center $[24:40, 24:40]$ = $16 \times 16 = 256$ pixels ($6.25\%$).
- **Theoretical Unique Bootstrap Samples:** $240(1 - (239/240)^{240}) = \mathbf{151.89}$ ($63.29\%$).
- **Asymptotic Limit:** $240(1 - 1/e) = \mathbf{151.71}$ ($63.21\%$).
- **Empirical Unique Bootstrap Samples:** **$151.89$** ($63.29\%$, exact match).
- **Hardest Boosting Sample:** Image **#199 (Class 33)**; initial weight $= 0.00417 \to$ peak weight $= \mathbf{0.09542}$ ($23\times$ increase); correctly classified in only $13/100$ weak rounds.

