import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from lab_utils_multi import load_house_data
from lab_utils_common import dlc

# Settings properties
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')

# Importing our train data
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']

# Scaling our data
scaler = StandardScaler();
X_norm = scaler.fit_transform(X_train);

# Running the learning algorithm to find the perfect match
sgdr = SGDRegressor(max_iter=1000);
sgdr.fit(X_norm, y_train);

# Our trained constants
b_norm = sgdr.intercept_
w_norm = sgdr.coef_

# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)

# plot predictions and targets vs original features
fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],y_pred_sgd,color=dlc["dlorange"], label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()