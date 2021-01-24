import dataset
import tensorflow as tf
import numpy as np
import csv
import operator
import matplotlib.pyplot as plt

TRAINING_SET_FRACTION = 0.90


def main(argv):
	data_to_predict = dataset.Dataset('data/booki.csv')
	data = dataset.Dataset('data/bookEN.csv')

	train_results_len = int(TRAINING_SET_FRACTION * len(data.processed_results))
	train_results = data.processed_results[:train_results_len]
	test_results = data.processed_results[train_results_len:]

	dp = data_to_predict.processed_results
	def get_result(x):
		r = ['H', 'D', 'A']
		index = np.argmax(x)
		return r[index]
	
	def map_results(results):
		features = {}

		for result in results:
			for key in result.keys():
				if key not in features:
					features[key] = []

				features[key].append(result[key])

		for key in features.keys():
			features[key] = np.array(features[key])

		return features, features['result'], features['hTeam'], features['aTeam']

	train_features, train_labels, hTeams, aTeams = map_results(train_results)
	test_features, test_labels, hTeams, aTeams = map_results(test_results)

	dp_features, dp_labels, dp_hTeams, dp_aTeams = map_results(dp)
	
	dp_input_fn = tf.estimator.inputs.numpy_input_fn(
		x=dp_features,
		#y=dp_labels,
		num_epochs=1,
		shuffle=False
	)
	
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x=train_features,
		y=train_labels,
		batch_size=500,
		num_epochs=None,
		shuffle=False
	)

	test_input_fn = tf.estimator.inputs.numpy_input_fn(
		x=test_features,
		y=test_labels,
		num_epochs=1,
		shuffle=False
	)

	feature_columns = []

	# data from last 10 games
	for mode in ['home', 'away']:
		feature_columns = feature_columns + [
			
			tf.feature_column.numeric_column(key='{}-wins-home'.format(mode)), # wins
			tf.feature_column.numeric_column(key='{}-wins-away'.format(mode)), # wins
			tf.feature_column.numeric_column(key='{}-losses-home'.format(mode)), # wins
			tf.feature_column.numeric_column(key='{}-losses-away'.format(mode)), # wins
			tf.feature_column.numeric_column(key='{}-draws-home'.format(mode)), # wins
			tf.feature_column.numeric_column(key='{}-draws-away'.format(mode)), # wins
			#tf.feature_column.numeric_column(key='{}-wins'.format(mode)), # wins
			#tf.feature_column.numeric_column(key='{}-draws'.format(mode)), # losses
			#tf.feature_column.numeric_column(key='{}-losses'.format(mode)), # draws
			tf.feature_column.numeric_column(key='{}-goals'.format(mode)), # goals scored
			tf.feature_column.numeric_column(key='{}-opposition-goals'.format(mode)), # goals suffered
			tf.feature_column.numeric_column(key='{}-shots'.format(mode)),
			tf.feature_column.numeric_column(key='{}-shots-on-target'.format(mode)),
			tf.feature_column.numeric_column(key='{}-opposition-shots'.format(mode)),
			tf.feature_column.numeric_column(key='{}-opposition-shots-on-target'.format(mode)),
		]

	model = tf.estimator.DNNClassifier(
		model_dir='model/',
		hidden_units=[19, 12], #[19] (input * 2/3 + output) | if[12,8] first layer 12 neurons second 8
		feature_columns=feature_columns,
		n_classes=3,
		label_vocabulary=['H', 'D', 'A'],
		activation_fn = tf.nn.relu, 
		optimizer=tf.train.ProximalAdagradOptimizer(
			learning_rate=0.01,
			l1_regularization_strength=0.001
	))

	#with open('data.csv', 'w') as f:
    	#	for key in train_features.keys():
        #		f.write("%s,%s\n"%(key,train_features[key]))
	
	print('train nr data: {} | test nr data: {}'.format(len(train_labels), len(test_labels)))
	with open('training-log.csv', 'w') as stream:
		csvwriter = csv.writer(stream)

		epochs = 50
		eval_data = []
		for x in range(1, epochs):

			train = model.train(input_fn=train_input_fn, steps=100)
			evaluation_result = model.evaluate(input_fn=test_input_fn)
			eval_data.append(evaluation_result)


		accuracy = []
		loss = []
		for x in eval_data:
			accuracy.append(x['accuracy'])
			loss.append(x['loss'])

		fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
		fig.suptitle('England Training Metrics')

		axes[0].set_ylabel("Loss", fontsize=14)
		axes[0].plot(loss)

		axes[1].set_ylabel("Accuracy", fontsize=14)
		axes[1].set_xlabel("Epoch", fontsize=14)
		axes[1].plot(accuracy)
		plt.show()	

		labels = []
		home_features = []
		away_features = []

		for x in test_labels:
			if x == 'H': labels.append('red')
			elif x == 'D': labels.append('black')
			elif x == 'A': labels.append('blue')

		for x in range(0, len(test_features['result'])):
			home_features.append(test_features['home-wins-home'][x]+test_features['home-wins-away'][x]+test_features['home-losses-home'][x]+test_features['home-losses-away'][x]+test_features['home-draws-home'][x]+test_features['home-draws-away'][x]+test_features['home-shots'][x]+test_features['home-opposition-shots'][x]+test_features['home-goals'][x]+test_features['home-opposition-goals'][x]+test_features['home-shots-on-target'][x]+test_features['home-opposition-shots-on-target'][x])

			away_features.append(test_features['away-wins-home'][x]+test_features['away-wins-away'][x]+test_features['away-losses-home'][x]+test_features['away-losses-away'][x]+test_features['away-draws-home'][x]+test_features['away-draws-away'][x]+test_features['away-shots'][x]+test_features['away-opposition-shots'][x]+test_features['away-goals'][x]+test_features['away-opposition-goals'][x]+test_features['away-shots-on-target'][x]+test_features['away-opposition-shots-on-target'][x])

		print(np.array(home_features))
		print(np.array(away_features))
		print(np.array(labels))

		plt.scatter(home_features, away_features, c=labels, cmap='viridis')
		plt.xlabel("home features")
		plt.ylabel("away features")
		plt.show()

		predictions = list(model.predict(input_fn=test_input_fn))
		
		acc = 0
		i = 0
		for x in predictions:
			pred = get_result(np.array(x['probabilities']))
			if test_labels[i] == pred: acc += 1
			print("Game: {} vs {} | actual winner: {} - predicted winner: {} | prediction: {}".format(hTeams[i], aTeams[i], test_labels[i], pred, x['probabilities']))
			i += 1

		print('Accuracy: {} | correctly perdicted: {} | nr of data: {}'.format(acc/len(test_labels), acc, len(test_labels)))
	
		pred_data = list(model.predict(input_fn=dp_input_fn))
		
		i = 0
		acc = 0
		for x in pred_data:
			pred = get_result(np.array(x['probabilities']))
			if dp_labels[i] == pred: acc += 1
			print("Game: {} vs {} | actual winner: {} - predicted winner: {} | probabilities: {}".format(dp_hTeams[i], dp_aTeams[i], dp_labels[i], pred, x['probabilities']))
			i += 1

		print('Accuracy: {} | correctly perdicted: {} | nr of data: {}'.format(acc/len(dp_labels), acc, len(dp_labels)))	

		csvwriter.writerow([evaluation_result['accuracy'], evaluation_result['average_loss']])


if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main=main)
