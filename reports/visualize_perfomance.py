import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_excel(r".\lemon_grading\reports\results\output.xlsx")

# plot violin. 'Scenario' is according to x axis,
# 'LMP' is y axis, data is your dataframe. ax - is axes instance
sns.displot(data=df, x="test_acc", kind="kde")
plt.show()

sns.displot(data=df, x="test_acc", hue="unfreeze_top_num_layers", kind="kde")
plt.show()

sns.displot(data=df, x="test_acc", hue="dropout_rate", kind="kde")
plt.show()
a = 9
