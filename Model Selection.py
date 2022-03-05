df_4 = pd.read_csv("diabetes.csv")
df_4.head()

## Identify variables and outcome
df_4_x = df_4.iloc[:,:8]
df_4_y = df_4.iloc[:,8]

## Test for Multicollinearity
corr = df_4_x.corr(method='spearman')

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

fig, ax = plt.subplots(figsize=(10, 8))

cmap = sns.diverging_palette(220, 10, as_cmap=True, sep=100)

sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, linewidths=.5)

fig.suptitle('Correlation matrix of features', fontsize=15)
ax.text(0.77, 0.2, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
         transform=ax.transAxes, color='grey', alpha=0.5)

fig.tight_layout()
### Multicollinearity is acceptable

## Built Logit Model
logit_model_full=sm.Logit(df_4_y,df_4_x)
result=logit_model_full.fit()
print(result.summary2())

df_4["Outcome"].value_counts()

def get_result(x_df4): 
    i = 0
    while i < 8:
        logit_df4 = sm.Logit(df_4_y,x_df4).fit()
        logit_pvalue = logit_df4.pvalues
        drop_cols = logit_pvalue[logit_pvalue > 0.05].index.to_list()
        if len(drop_cols) == 0:
            break
        else:
            x_df4 = x_df4.drop(drop_cols,axis=1,inplace=True)
        i += 1
    return (logit_df4)

logit_df4 = get_result(df_4_x)
logit_df4.summary2()

## Using SMOTE to fix the imbalance
os = SMOTE(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(df_4_x, df_4_y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_resample(X_train, y_train)

os_data_X = pd.DataFrame(os_data_X,columns=columns )
os_data_y= pd.DataFrame(os_data_y.to_list(),columns=['y'])

print("length of oversampled data is ",len(os_data_X))
print("Number of people without diabetes in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of people with diabetes",len(os_data_y[os_data_y['y']==1]))
print("Proportion of people without diabetes in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of people with diabetes in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))

## Model Fitting
X_train = os_data_X
X_test = df_4_x
y_train = os_data_y
y_test = df_4_y

logit_df4_final = LogisticRegression()
logit_df4_final.fit(X_train, y_train)

y_pred = logit_df4_final.predict(X_test)

print('AUC score of logistic regression classifier on test set: {:.2f}'.format(roc_auc_score(y_test, y_pred)))

confusion_matrix = confusion_matrix(y_test, y_pred)

confusion_matrix

print(classification_report(y_test, y_pred))

logit_roc_auc = roc_auc_score(y_test, y_pred)

fpr, tpr, thresholds = roc_curve(y_test, logit_df4_final.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')

plt.show()


