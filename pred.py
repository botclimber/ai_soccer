import dataset
import tensorflow as tf
import numpy as np
import csv

def main(argv):
	
	data = dataset.Dataset('data/book.csv')	
	test_results = data.processed_results[3430:]
	
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

	test_features, test_labels, hTeams, aTeams = map_results(test_results)
	
	data = tf.estimator.inputs.numpy_input_fn(
		x=test_features,
		#y=test_labels,
		num_epochs=1,
		shuffle=False
	)

	feature_columns = []

	for mode in ['home', 'away']:
		feature_columns = feature_columns + [
			tf.feature_column.numeric_column(key='{}-wins'.format(mode)),
			tf.feature_column.numeric_column(key='{}-draws'.format(mode)),
			tf.feature_column.numeric_column(key='{}-losses'.format(mode)),
			tf.feature_column.numeric_column(key='{}-goals'.format(mode)),
			tf.feature_column.numeric_column(key='{}-opposition-goals'.format(mode)),
			tf.feature_column.numeric_column(key='{}-shots'.format(mode)),
			tf.feature_column.numeric_column(key='{}-shots-on-target'.format(mode)),
			tf.feature_column.numeric_column(key='{}-opposition-shots'.format(mode)),
			tf.feature_column.numeric_column(key='{}-opposition-shots-on-target'.format(mode)),
		]

	model = tf.estimator.DNNClassifier(
		model_dir='model/',
		hidden_units=[15],
		feature_columns=feature_columns,
		n_classes=3,
		label_vocabulary=['H', 'D', 'A'],
	)
	
	predictions = list(model.predict(input_fn=data))
	
	i = 0	
	for x in predictions:
		print("Game: {} vs {} | winner: {} | prediction: {}".format(hTeams[i], aTeams[i], test_labels[i], x['probabilities']))
		i += 1		



if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main=main)
