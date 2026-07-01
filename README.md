Day 1:
- Downloaded Credit Score Classification dataset from Kaggle
- Understand the dataset
- Create a project folder
- Identify target column (Credit Score)

The dataset represents 8 months of financial behaviour records for 12,500 unique bank customers, tracked monthly to classify them into one of three credit score brackets: Good, Standard, or Poor.

Attribute           Detail
Total Rows          100,000
Total Columns        28
Unique Customers    12,500 (each appears ~8 times)
Time Period         Month 1 → Month 8
Target Variable     Credit_Score (Good / Standard / Poor)
Missing Values      0 (this is a pre-cleaned mirror)
Data Types          8 numeric, 20 categorical

Key feature groups:

Identity: Customer_ID, Month, Age, Occupation
Income: Annual_Income, Monthly_Inhand_Salary
Debt behaviour: Outstanding_Debt, Num_of_Loan, Num_of_Delayed_Payment, Delay_from_due_date
Credit profile: Num_Bank_Accounts, Num_Credit_Card, Credit_Mix, Credit_History_Age
Behaviour: Payment_of_Min_Amount, Payment_Behaviour, Total_EMI_per_month

Target class imbalance: Standard 53.2% · Poor 29.0% · Good 17.8% — the dataset is skewed; "Good" is a minority class.

 Top 5 Surprising Insights

① The "Minimum Payment Trap" is the single most alarming behavioural signal.
Customers who pay only the minimum amount on their credit cards are 39.6% Poor vs only 3.7% Good — a 10× gap. Counterintuitively, customers labelled "No" (pays more than minimum) show 38.3% Good and only 13.4% Poor. Most people assume paying the minimum is "responsible" — the data says it's a red flag.

② Interest rate is the #1 numerical predictor (r = −0.48), beating income.
The average interest rate for "Good" customers is 7.66% vs 20.19% for "Poor" — a 2.6× difference. This suggests interest rate isn't just a symptom but acts as a proxy for the lender's risk assessment of that customer, making it a leading indicator embedded in the data itself.

③ Number of bank accounts hurts your score — more is worse.
Good customers average 3.4 accounts vs Poor customers averaging 6.5. This is counterintuitive; many assume more accounts = more financial activity = better. The reality is that excessive account-opening signals credit-seeking behaviour.

④ Credit History Age separates the top from the bottom decisively.
Good-score customers have an average credit history of 284 months (~23.7 years) vs 170 months (~14.2 years) for Poor. This 9-year gap suggests that young or new-to-credit customers are systematically disadvantaged and need dedicated products.

⑤ Credit Mix quality is a near-binary classifier.
Among customers with a "Good" credit mix (diverse, healthy portfolio), 48.9% are in the Good tier. Among those with a "Bad" mix, 60.1% are in the Poor tier. This single categorical variable would be an extremely powerful feature in any ML model.

Correlation Rankings (Spearman, with Credit Score)
Rank Feature                     Correlation Direction
1   Interest Rate               −0.482      Higher rate → worse score
2   Days Overdue                −0.438      More delays → worse score
3   Outstanding Debt            −0.434      More debt → worse score
4   Credit History Age          +0.397      Longer history → better score
5   Num Bank Accounts           −0.387      More accounts → worse score
6   Num Credit Cards            −0.386      More cards → worse score
7   Num of Delayed Payments     −0.365      More delays → worse score
8   Num of Loans                −0.354      More loans → worse score
9   Monthly Balance             +0.220      Higher balance → better score
10  Annual Income               +0.207      Higher income → better score

NOTE:Credit_Utilization_Ratio has nearly zero correlation (+0.042), meaning raw utilization ratio is a surprisingly weak standalone predictor here — the type and mix of credit matters far more.

train.csv
     │
     ▼
Exploratory Data Analysis (EDA)
     │
     ▼
Data Preprocessing
     │
     ▼
Feature Engineering
     │
     ▼
Train/Test Split (using train.csv)
     │
     ├── Training Set (80%)
     └── Validation/Test Set (20%)
     │
     ▼
Train Machine Learning Model
     │
     ▼
Evaluate Model
     │
     ▼
Save Best Model
     │
     ▼
Use test.csv for Final Predictions (optional)

Day 2: Exploratory data analysis-Distributions, correlations, class imbalance
- Check distribution of credit scores
- Find important features
- Create graphs

Why train.csv only??
train.csv contains both the features and the target column (Credit_Score).You can analyze the relationship between features and the target.test.csv only contain the inputs.

Day 3:  Data preprocessing
- Remove unnecessary columns
- Handle missing values
- Fix data types
- Remove duplicates
- Handling outliers

Load Dataset
      ↓
Create Copy
      ↓
Understand Every Column
      ↓
Remove Irrelevant Columns
      ↓
Clean Numerical Columns
      ↓
Clean Categorical Columns
      ↓
Handle Missing Values
      ↓
Remove Duplicates
      ↓
Fix Data Types
      ↓
Remove Special Characters
      ↓
Fix Category Labels
      ↓
Detect Impossible Values
      ↓
Detect Outliers
      ↓
Treat Outliers
      ↓
Final Validation
      ↓
Save Cleaned Dataset

Day 4: 
1.> 3 Feature engineering 
- Convert text columns into numbers.
- Create new features
- encoding
- scaling

Cleaned Dataset
      ↓
1. Check Data Types
      ↓
2. Convert Target Variable
      ↓
3. Create New Features
      ↓
4. Encode Categorical Features
      ↓
5. Scale Numerical Features
      ↓
6. Feature Selection
      ↓
Final Dataset Ready for ML

2.> Train Models

1. Train-Test Split
        ↓
2. Logistic Regression (Baseline, interpretable)
        ↓
3. Decision Tree (Explainable splits)
        ↓
4. Random Forest (High accuracy, robust)
        ↓
5. XGBoost (Optional)
        ↓
6. Compare All Models
        ↓
7. Hyperparameter Tuning (Best Model)
        ↓
8. Save Final Model

3.>Model evaluation
Precision, recall, F1, ROC-AUC, confusion matrix

4.> Hyperparameter tuning
Randomized search, cross-validation, threshold selection

Day 5:Build web application
- Setup Streamlit
- Configure the page: Add: Page title, Page icon, Wide layout
- Sidebar: Project Name, Developer Name, Model Used, Dataset Information
- Connect Model
- Input Form
- Improve UI
- Recommendation engine added
- Analytics Dashboard

Structure of app.py:
1. Import libraries
        ↓
2. Configure Streamlit page
        ↓
3. Load model & feature names
        ↓
4. Create UI (number inputs, select boxes)
        ↓
5. Wait for user to click Predict
        ↓
6. Compute engineered features
        ↓
7. Create DataFrame
        ↓
8. Apply preprocessing (get_dummies, reindex)
        ↓
9. Predict
        ↓
10. Display result