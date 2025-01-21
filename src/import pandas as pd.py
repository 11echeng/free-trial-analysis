import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
data = pd.read_csv('log_data.csv')

# Create dummy variables for categorical columns
data_with_dummies = pd.get_dummies(data, columns=['ATL_OR_DR', 'CAMPAIGN_TYPE', 'CHANNEL'], drop_first=True)

# Create the model
X = data_with_dummies.drop('LOG_FREE_TRIALS', axis=1)
y = data_with_dummies['LOG_FREE_TRIALS']

# Add constant
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Print summary
print(model.summary())

# Calculate VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVariance Inflation Factors:")
print(vif_data)

# Create diagnostic plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Residuals vs Fitted
ax1.scatter(model.fittedvalues, model.resid)
ax1.set_xlabel('Fitted values')
ax1.set_ylabel('Residuals')
ax1.set_title('Residuals vs Fitted')
ax1.axhline(y=0, color='r', linestyle=':')

# QQ Plot
sm.graphics.qqplot(model.resid, dist='norm', line='45', fit=True, ax=ax2)
ax2.set_title('Q-Q plot')

# Scale-Location
ax3.scatter(model.fittedvalues, np.sqrt(np.abs(model.resid)))
ax3.set_xlabel('Fitted values')
ax3.set_ylabel('Sqrt(|Residuals|)')
ax3.set_title('Scale-Location')

# Residuals vs Leverage
sm.graphics.influence_plot(model, ax=ax4)
ax4.set_title('Residuals vs Leverage')

plt.tight_layout()