import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

data = pd.read_csv("error_fold1.csv")
df = pd.DataFrame({'err': data['error']}, index=data['err_id'])

plot = df.plot.line()
fig = plot.get_figure()
fig.savefig('plot_1.png')