
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
from sklearn.utils import resample 
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing 
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import plot_confusion_matrix 
from sklearn.decomposition import PCA 
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, brier_score_loss


df = pd.read_csv (r'C:/Users/65904/Desktop/Machine Learning/Datasets/heart_2020_cleaned.csv')

#EDA
df.dtypes
df.describe()
df['AgeCategory'].unique()


print(df.isna().sum()) #blanks out of 1718 rows

#Replacing 1, 0s
df.HeartDisease.replace(('Yes', 'No'), (1, 0), inplace=True)
df.Smoking.replace(('Yes', 'No'), (1, 0), inplace=True)
df.AlcoholDrinking.replace(('Yes', 'No'), (1, 0), inplace=True)
df.Stroke.replace(('Yes', 'No'), (1, 0), inplace=True)
df.DiffWalking.replace(('Yes', 'No'), (1, 0), inplace=True)
df.Sex.replace(('Female', 'Male'), (1, 0), inplace=True)
df.PhysicalActivity.replace(('Yes', 'No'), (1, 0), inplace=True)
df.Asthma.replace(('Yes', 'No'), (1, 0), inplace=True)
df.KidneyDisease.replace(('Yes', 'No'), (1, 0), inplace=True)
df.SkinCancer.replace(('Yes', 'No'), (1, 0), inplace=True)
df.head(5)

# df['XXX'].replace(' ', '_', regex=True, inplace=True)

# Resampling of data for large datasets
df_no_HD = df[df['HeartDisease'] == 0]
df_HD = df[df['HeartDisease'] == 1]

df_no_HD_downsampled = resample(df_no_HD,
                                  replace=False,
                                  n_samples=1000,
                                  random_state=42)
len(df_no_HD_downsampled)


df_HD_downsampled = resample(df_HD,
                                  replace=False,
                                  n_samples=1000,
                                  random_state=42)
len(df_HD_downsampled)
df_downsample = pd.concat([df_no_HD_downsampled, df_HD_downsampled])
len(df_downsample)


#X,y Split
X = df_downsample.drop('HeartDisease', axis=1).copy() 
X.head() 
y = df_downsample['HeartDisease'].copy()
y.head()

#One Hot Encoding
X_encoded = pd.get_dummies(X, columns=['AgeCategory',
                                       'Race',
                                       'GenHealth',
                                       'Diabetic'
                                       ])
X_encoded.head()
X_encoded.dtypes

# X,y train test split + Stratification 
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42, stratify = y)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_train_df = pd.DataFrame(X_train_scaled, columns = X_train.columns, index = X_train.index)

#standardizing the out-of-sample data
X_test_scaled = scaler.transform(X_test)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns = X_test.columns, index = X_test.index)


# y_test_scaled = scaler.transform(y_test)
# y_test_scaled_df = pd.DataFrame(y_test_scaled, columns = y_test.columns, index = y_test.index)


# # Build A Preliminary Support Vector Machine
clf_svm = SVC(random_state=42)
clf_svm.fit(X_train_scaled, y_train)

plot_confusion_matrix(clf_svm, 
                      X_test_scaled, 
                      y_test,
                      display_labels=["Does not have HD", "Has HD"])


# Using  `GridSearchCV()`. 
param_grid = [
  {'C': [0.5, 1, 10, 100], # NOTE: Values for C must be > 0
   'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001], 
   'kernel': ['rbf']},
]


optimal_params = GridSearchCV(
        SVC(), 
        param_grid,
        cv=5,
        scoring='roc_auc', 
        verbose=0 
    )

optimal_params.fit(X_train_scaled, y_train)
print(optimal_params.best_params_)

clf_svm = SVC(random_state=42, C=100, gamma= 0.0001)
clf_svm.fit(X_train_scaled, y_train)


plot_confusion_matrix(clf_svm, 
                      X_test_scaled, 
                      y_test, 
                      display_labels=["Does not have HD", "Has HD"])



num_features = np.size(X_train_scaled, axis=1)
param_grid = [
  {'C': [1, 10, 100, 1000], 
   'gamma': [1/num_features, 1, 0.1, 0.01, 0.001, 0.0001], 
   'kernel': ['rbf']},
]

optimal_params = GridSearchCV(
        SVC(), 
        param_grid,
        cv=5,
        scoring='roc_auc', 
        verbose=0 
    )

optimal_params.fit(X_train_scaled, y_train)
print(optimal_params.best_params_)


clf_svm = SVC(random_state=42, C=1000, gamma=0.0001)
clf_svm.fit(X_train_scaled, y_train)

plot_confusion_matrix(clf_svm, 
                      X_test_scaled, 
                      y_test, 
                      display_labels=["Does not have HD", "Has HD"])



len(df_downsample.columns)



pca = PCA() # NOTE: By default, PCA() centers the data, but does not scale it.
X_train_pca = pca.fit_transform(X_train_scaled)


per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = [str(x) for x in range(1, len(per_var)+1)]
 
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Components')
plt.title('Screen Plot')
plt.show()


# The screen plot shows that the first principal component, PC1, accounts for a relatively large amount of variation in the raw data, 
# and this means that it will be good candidate for the x-axis in the 2-dimensional graph. 
# However, PC2 is not much different from PC3 or PC4, which doesn't bode well for dimension reduction. 
# Now we will draw the PCA graph. 

train_pc1_coords = X_train_pca[:, 0] 
train_pc2_coords = X_train_pca[:, 1]


## NOTE:
## pc1 contains the x-axis coordinates of the data after PCA
## pc2 contains the y-axis coordinates of the data after PCA

## Now center and scale the PCs...
pca_train_scaled = preprocessing.scale(np.column_stack((train_pc1_coords, train_pc2_coords)))



## Now we optimize the SVM fit to the x and y-axis coordinates
## of the data after PCA dimension reduction...
num_features = np.size(pca_train_scaled, axis=1)
param_grid = [
  {'C': [1, 10, 100, 1000], 
   'gamma': [1/num_features, 1, 0.1, 0.01, 0.001, 0.0001], 
   'kernel': ['rbf']},
]

optimal_params = GridSearchCV(
        SVC(), 
        param_grid,
        cv=5,
        scoring='roc_auc', 
        verbose=0 
    )

optimal_params.fit(pca_train_scaled, y_train)
print(optimal_params.best_params_)

#Calculating Accuracy and Recall
clf_svm = SVC( kernel='rbf', random_state=42, C=1000, gamma=0.001, probability=True)
classifier = clf_svm.fit(X_train_df, y_train)


predicted = classifier.predict(X_test_scaled_df) 
prob_default = classifier.predict_proba(X_test_scaled_df)
prob_default = [x[1] for x in prob_default] 

print("accuracy:", accuracy_score(y_test, predicted))
print("balanced_accuracy:", balanced_accuracy_score(y_test, predicted))
print("recall:", recall_score(y_test, predicted))
print("brier_score_loss:", brier_score_loss(y_test, prob_default))


#Plotting the SVM Chart 
clf_svm = SVC( kernel='rbf', random_state=42, C=1000, gamma=0.01)
classifier = clf_svm.fit(pca_train_scaled, y_train)

X_test_pca = pca.transform(X_test_scaled)
test_pc1_coords = X_test_pca[:, 0] 
test_pc2_coords = X_test_pca[:, 1]


x_min = test_pc1_coords.min() - 1
x_max = test_pc1_coords.max() + 1

y_min = test_pc2_coords.min() - 1
y_max = test_pc2_coords.max() + 1

xx, yy = np.meshgrid(np.arange(start=x_min, stop=x_max, step=0.1),
                     np.arange(start=y_min, stop=y_max, step=0.1))


Z = clf_svm.predict(np.column_stack((xx.ravel(), yy.ravel())))

Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10,10))

ax.contourf(xx, yy, Z, alpha=0.1)

## now create custom colors for the actual data points
cmap = colors.ListedColormap(['#e41a1c', '#4daf4a'])


scatter = ax.scatter(test_pc1_coords, test_pc2_coords, c=y_test, 
               cmap=cmap, 
               s=100, 
               edgecolors='k', ## 'k' = black
               alpha=0.7)

## now create a legend
legend = ax.legend(scatter.legend_elements()[0], 
                   scatter.legend_elements()[1],
                    loc="upper right")
legend.get_texts()[0].set_text("No HD")
legend.get_texts()[1].set_text("Yes HD")

## now add axis labels and titles
ax.set_ylabel('PC2')
ax.set_xlabel('PC1')
ax.set_title('Decison surface using the PCA transformed/projected features')
## plt.savefig('svm.png')
plt.show()


