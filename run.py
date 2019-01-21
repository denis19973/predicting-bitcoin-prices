import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

models = [
        SVR(kernel='linear', C=1e3, gamma=0.1),
        SVR(kernel='poly', C=1e3, degree=2, gamma=0.1),
        SVR(kernel='rbf', C=1e3, gamma=0.1),
	]

colors = ['green', 'red', 'blue']

def get_data(filename):
	dates = []
	prices = []
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		for row in csvFileReader:
			date_str = row[0]
			dates.append(date_str[:date_str.rfind('-')])
			prices.append(float(row[4]))
	return dates, prices


def fit_and_plot_model(model, X, y, color='black', label=''):
	model.fit(X, y)
	plt.plot(X, model.predict(X), color=color, label=label)

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('SVM Regression')

dates, prices = get_data('BTC-USD.csv')
int_dates = [i for i in range(1, len(dates) + 1)]
int_dates = np.reshape(int_dates, (len(dates), 1))
plt.scatter(int_dates, prices, color='black', label='Initial Data')
used_colors = []
for model in models:
	color = list(filter(lambda c: c not in used_colors, colors))[0]
	used_colors.append(color)
	fit_and_plot_model(
		    model, 
		    int_dates, 
		    prices, 
		    color=color, 
		    label=str(model.kernel),
		)

plt.legend()
plt.show()
