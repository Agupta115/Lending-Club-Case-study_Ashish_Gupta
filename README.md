# Lending-Club-Case-study_Ashish_Gupta
    # importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows',1000)

# Read data from File
loan= pd.read_csv(r"C:\Users\gexce\OneDrive\Case study\loan.csv")

# check data 
loan.head()

## Data Pre-processing

#Drop the columns having only null values.
loan.dropna(axis = 1, how = 'all', inplace = True)
loan.drop(['revol_bal','total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries','collection_recovery_fee', 'last_pymnt_d','last_pymnt_amnt','next_pymnt_d', 'last_credit_pull_d','delinq_2yrs','out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv','title','application_type','policy_code','collections_12_mths_ex_med','initial_list_status','pymnt_plan','delinq_amnt','chargeoff_within_12_mths','acc_now_delinq',"member_id","emp_title","zip_code","tax_liens",'url','desc','mths_since_last_delinq','mths_since_last_record'],axis=1, inplace=True)
loan.shape

# Removing duplicate rows in the loanframe
loan_data = loan.drop_duplicates()
# Shape of the dataframe after removing duplicate rows
print(loan.shape)

# No duplicate rows found in the dataframe

##### Removing rows with loan status as Current(i.e loan is currently ongoing), as analysis is required on only defaulted or completed loans

loan = loan[loan.loan_status != "Current"]
loan.loan_status.unique()

100 * loan.isnull().mean()

#Identifying the mode value and replacing the null values with mode for categorical variables(i.e emp_length,pub_rec_bankruptcies) and droping the rows with na values in revol_util column
print('Mode : ' + loan.emp_length.mode())
# loan.emp_length.value_counts()

loan.emp_length.fillna(loan.emp_length.mode()[0],inplace=True)
loan.dropna(axis = 0, subset = ['revol_util'] , inplace = True)
loan.pub_rec_bankruptcies.fillna(loan.pub_rec_bankruptcies.mode()[0],inplace=True)

###### Creating new columns and correcting datatypes of columns

#Removing % from int_rate or revol_util column and converting them from string to integer
loan['int_rate'] = pd.to_numeric(loan['int_rate'].str.strip('%'))
loan['revol_util']= pd.to_numeric(loan['revol_util'].str.strip('%'))

#Converting issue_d into date format
loan['issue_d'] = pd.to_datetime(loan['issue_d'] , format='%b-%d')
loan['issue_d'] = loan['issue_d'].apply(lambda x: x.replace(year=2024))

# Deriving new column as issue_month and issue_year from issue_d column
loan['issue_month'] = loan['issue_d'].dt.strftime('%b')
loan['issue_week']= loan['issue_d'].dt.weekday.astype(object)

# Removing months from term column values
loan['term']=loan['term'].apply(lambda x: int(x.replace(' months',''))).astype(int)

# Removing special chartacters and words from values of emp_length column 
loan['emp_length']=loan['emp_length'].apply(lambda x: x.replace('years','').replace('+','').replace('< 1','0.5').replace('year','')).astype(float)


###### Outliers identification in the data

numerical_col= ['loan_amnt','funded_amnt', 'funded_amnt_inv', 'int_rate', 'installment','annual_inc','dti','revol_util']
categorical_col= ['term','grade', 'sub_grade','emp_length', 'home_ownership','verification_status','loan_status','purpose', 'addr_state','inq_last_6mths','open_acc','pub_rec','total_acc','pub_rec_bankruptcies','issue_month','issue_week']
extra_col=['id','issue_d','earliest_cr_line']

len(numerical_col+categorical_col+extra_col)

# Finding outliers for each numerical column
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

# Loop through numerical columns and create a boxplot for each
for i, col in enumerate(numerical_col):
    sns.boxplot(x=loan[col], ax=axes[i])
    axes[i].set_ylabel('count')
    axes[i].set_title(f'{col} Boxplot')

# Hide any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.show()

#finding quartiles for each numerical column
for i in numerical_col:
    print(i + ' at different quartiles')
    print(loan[i].quantile([0.01,0.1,0.5,0.10,0.50,0.75,0.90,0.95,0.96,0.97,0.98,0.99,1.0]))

#Removing values with annual income greater than equal to 0.99 quartile as there is an exponential increase in annual income around the 99th percentile. Therefore, it is advisable to exclude values beyond the 99th percentile.
annual_inc_99_per = loan['annual_inc'].quantile(0.99)
loan = loan[loan.annual_inc <= annual_inc_99_per]

# check size of data

loan.shape

# now we will do analysis 
# first analysis will be based on Univariate analysis
# check loan status first
print(100*  (loan['loan_status'].value_counts()/loan['loan_status'].value_counts().sum()))

As per above analysis below are the observations
Observation : 
- 85.40 percent of people have fully paid the loan whereas approx. 
- 14.60 percent of people defaulted.

will check loan status fully paid vs charged off


plt.figure(figsize=(6,3))
ax = sns.countplot(x='loan_status', data=loan)
for p in ax.patches:
    height = int(p.get_height()) if p.get_height().is_integer() else p.get_height()
    ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.title('Count of Members with Loan Status Paid/ Charged Off')
plt.ylabel('loan_status count')
plt.show()

let us check term loan

print(100*  (loan['term'].value_counts()/loan['term'].value_counts().sum()))
plt.figure(figsize=(6,3))
ax = sns.countplot(x='term', data=loan)
for p in ax.patches:
    height = int(p.get_height()) if p.get_height().is_integer() else p.get_height()
    ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.title('Count of Members with Term of Loan')
plt.ylabel('term count')
plt.show()

As per above analysis below are the observations

Observation: 
    More then 75% loans are taken fore duration of 36 months in comparison to 60 months term plan

Let's analysis on grade

print(100*  (loan['grade'].value_counts()/loan['grade'].value_counts().sum()))
plt.figure(figsize=(6,4))
sorted_order = sorted(loan['grade'].unique())
ax = sns.countplot(x = 'grade', data = loan, order=sorted_order)
for p in ax.patches:
    height = int(p.get_height()) if p.get_height().is_integer() else p.get_height()
    ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.title('Grade distribution plot ')
plt.ylabel('Grade count')
plt.show()

Obervations - Findings : More then 50% borrowers belong to group A and B in comparizon to another groups

lets do analysis on sub grade


print(100*  (loan['sub_grade'].value_counts()/loan['sub_grade'].value_counts().sum()))
plt.figure(figsize=(14,4))
sorted_order = sorted(loan['sub_grade'].unique())
ax = sns.countplot(x = 'sub_grade', data = loan, order=loan['sub_grade'].value_counts().index)
for p in ax.patches:
    height = int(p.get_height()) if p.get_height().is_integer() else p.get_height()
    ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.title('Sub_grade distribution plot ')
plt.ylabel('Sub_grade count')
plt.show()

Observation: The majority of loans within grades A and B are predominantly acquired through subgrades 4 and 5, with each grade encompassing five subgrades.

Lets do anlysis on employee


print(100*  (loan['emp_length'].value_counts()/loan['emp_length'].value_counts().sum()))
plt.figure(figsize=(10,3))
sns.countplot(x = 'emp_length', data = loan, order=sorted(loan.emp_length))
plt.title('Employee years of experience')
plt.ylabel('emp_length count')
plt.show()

Observartion: Around 1/4th of loans are taken by people with more then 10 years of experience

lets do analysis on loan amount

bins = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000]
labels = ['0-5k', '5k-10k', '10k-15k', '15k-20k', '20k-25k', '25k-30k', '30k-35k']
loan['loan_amnt_bins'] = pd.cut(loan['loan_amnt'], bins=bins, labels=labels)
# Count the number of loans in each bin
loan_amnt_bin_counts = loan['loan_amnt_bins'].value_counts().sort_index()
bin_counts_df = loan_amnt_bin_counts.reset_index()
bin_counts_df.columns = ['Loan Amount Bin', 'Number of Loans']

fig = px.bar(bin_counts_df, x='Loan Amount Bin', y='Number of Loans',
             title='Number of Loans per Loan Amount Bin',
             labels={'Loan Amount Bin': 'Loan Amount Bin', 'Number of Loans': 'Number of Loans'})
fig.show()
print(100 * (loan['loan_amnt_bins'].value_counts()/loan['loan_amnt_bins'].value_counts().sum()))

Higher amuunt of loan has taken in the amount bracket of 5k-10k

import matplotlib.dates as mdates
loan_1 = loan.copy()

loan_1['issue_d'] = pd.to_datetime(loan_1['issue_d'])
loan_1.set_index('issue_d', inplace=True) 

monthly_loan_amnt = loan_1['loan_amnt'].resample('M').sum()  
sorted_monthly_loan_amnt = monthly_loan_amnt.sort_index()

# Plotting the trend of loan amounts
plt.figure(figsize=(8, 3))
plt.plot(sorted_monthly_loan_amnt, marker='o')
plt.xticks(sorted_monthly_loan_amnt.index, [date.strftime('%b%y') for date in sorted_monthly_loan_amnt.index])
plt.title('Trend of Loan Amount')
plt.xlabel('Date')
plt.ylabel('Loan Amount')
plt.grid(True)
plt.show()

Observation: 
 - Majority of the loans taken in the month of December
 - Around 1/3rd of loans are taken between 5k to 10k 

now let's check interest rates

# Distribution of interest rate
plt.figure(figsize=(5,3))
sns.distplot(sorted(loan.int_rate),kde=True,bins=25)
plt.xlabel('Interest Rate')
plt.ylabel('Density')
plt.title('Distribution of Interest Rate')
plt.show()

Observations:
Majority of loans are taken between 10-15%, with decreasing flow after 15%.

#### Ownership

print(100*  (loan['home_ownership'].value_counts()/loan['home_ownership'].value_counts().sum()))
plt.figure(figsize=(10,3))
sns.countplot(x = 'home_ownership', data = loan, order=sorted(loan.home_ownership))
plt.title('home_ownership')
plt.ylabel('home_ownership count')
plt.show()

Observation: 
 - Just 7% of loan recipients are homeowners.
 - Over 90% of loan recipients reside in mortgaged or rented accommodations.

Lets check verificaiton status 

print(100*  (loan['verification_status'].value_counts()/loan['verification_status'].value_counts().sum()))
plt.figure(figsize=(10,3))
sns.countplot(x = 'verification_status', data = loan, order=sorted(loan.verification_status))
plt.title('verification_status')
plt.ylabel('verification_status count')
plt.show()

Observation: 
Over 50% of individuals have undergone either verification or source verification, with 43% people not verified

#### lets check loan purpose

print(100*  (loan['purpose'].value_counts()/loan['purpose'].value_counts().sum()))
plt.figure(figsize=(22,5))
sns.countplot(x = 'purpose', data = loan, order=sorted(loan.purpose))
plt.title('purpose')
plt.ylabel('purpose count')
plt.show()

Observation: More then 45% of loans are taken for debt consolidation 

