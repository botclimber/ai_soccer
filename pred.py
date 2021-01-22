import dataset
import tensorflow as tf
import numpy as np
import csv
import operator

def main(argv):

	data = dataset.Dataset('predict_data/dataToPred.csv')	
	predict_data = data.processed_results
	
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

	data_features, data_labels, hTeams, aTeams = map_results(predict_data)
	
	data = tf.estimator.inputs.numpy_input_fn(
		x=data_features,
		#y=test_labels,
		num_epochs=1,
		shuffle=False
	)

	feature_columns = []

	for mode in ['home', 'away']:
		feature_columns = feature_columns + [
			tf.feature_column.numeric_column(key='{}-wins-home'.format(mode)), # wins
			tf.feature_column.numeric_column(key='{}-wins-away'.format(mode)), # wins
			tf.feature_column.numeric_column(key='{}-losses-home'.format(mode)), # wins
			tf.feature_column.numeric_column(key='{}-losses-away'.format(mode)), # wins
			tf.feature_column.numeric_column(key='{}-draws-home'.format(mode)), # wins
			tf.feature_column.numeric_column(key='{}-draws-away'.format(mode)), # wins
			#tf.feature_column.numeric_column(key='{}-wins'.format(mode)),
			#tf.feature_column.numeric_column(key='{}-draws'.format(mode)),
			#tf.feature_column.numeric_column(key='{}-losses'.format(mode)),
			tf.feature_column.numeric_column(key='{}-goals'.format(mode)),
			tf.feature_column.numeric_column(key='{}-opposition-goals'.format(mode)),
			tf.feature_column.numeric_column(key='{}-shots'.format(mode)),
			tf.feature_column.numeric_column(key='{}-shots-on-target'.format(mode)),
			tf.feature_column.numeric_column(key='{}-opposition-shots'.format(mode)),
			tf.feature_column.numeric_column(key='{}-opposition-shots-on-target'.format(mode)),
		]

	model = tf.estimator.DNNClassifier(
		model_dir='model/',
		hidden_units=[19],
		feature_columns=feature_columns,
		n_classes=3,
		label_vocabulary=['H', 'D', 'A'],
	)
	
	predictions = list(model.predict(input_fn=data))
	
	i = 0	
	acc = 0
	for x in predictions:
		if data_labels[i] == get_result(np.array(x['probabilities'])): acc += 1
		print("Game: {} vs {} | winner: {} | prediction: {}".format(hTeams[i], aTeams[i], data_labels[i], x['probabilities']))
		i += 1		

	print('Accuracy: {} | correctly perdicted: {} | nr of data: {}'.format(acc/len(data_labels), acc, len(data_labels)))

if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main=main)
