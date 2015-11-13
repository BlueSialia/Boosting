import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Normalize;
import weka.filters.unsupervised.instance.RemoveRange;

import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

class Main {

	private static final Random randomGenerator = new Random();
	private static Instances data;
	private static Instances train;
	private static Instances sample;
	private static Instances test;
	private static int testStart;
	private static double[] classPredicted;
	private static double[] wk; //Synchronized with data
	private static double[] accuracies;

	public static void main(String[] args) {
		try {
			data = new Instances(new FileReader("res/iris.arff"));
			data.setClassIndex(data.numAttributes() - 1);
		} catch (IOException e) {
			System.err.println("Error reading file.");
			e.printStackTrace();
		}

		//Starting classification
		try {
			// Preprocess
			data.randomize(randomGenerator);
			data = filterNormalize(data);

			sample = new Instances(data, 0);

			crossValidation(10);

			double media = 0;
			for (double accuracy : accuracies) {
				media += accuracy;
//				System.out.println("accuracy = " + accuracies[i]);
			}
			System.out.println("Accuracy media = " + media/accuracies.length);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void crossValidation(int folds) {
		double foldSize = data.numInstances() / 10;
		accuracies = new double[folds];

		for (int i = 0; i < folds; i++) {
			KNN[] classifiers = new KNN[2];

			wk = new double[data.numInstances()];
			Arrays.fill(wk, 1.0 / data.numInstances());

			if (i == 0) {
				train = filterRemoveRange(data, "first-" + String.valueOf((int) foldSize));
				test = filterRemoveRange(data, String.valueOf((int) foldSize + 1) + "-last");
				testStart = 0;
				int[] starts = {(int) foldSize + 1};
				int[] ends = {data.numInstances()};
				rouletteWheel(starts, ends);
			} else if (i == folds - 1) {
				train = filterRemoveRange(data, String.valueOf(data.numInstances() - (int) foldSize + 1) + "-last");
				test = filterRemoveRange(data, "first-" + String.valueOf(data.numInstances() - (int) foldSize));
				testStart = data.numInstances() - (int) foldSize;
				int[] starts = {0};
				int[] ends = {data.numInstances() - (int) foldSize};
				rouletteWheel(starts, ends);
			} else {
				train = filterRemoveRange(data, String.valueOf((int) (foldSize * i) + 1) + "-" + String.valueOf((int) (foldSize * (i + 1))));
				test = filterRemoveRange(data, "first-" + String.valueOf((int) (foldSize * i)) + "," + String.valueOf((int) (foldSize * (i + 1)) + 1) + "-last");
				testStart = (int) (foldSize * i);
				int[] starts = {0, (int) (foldSize * (i + 1)) + 1};
				int[] ends = {(int) (foldSize * i) + 1, data.numInstances()};
				rouletteWheel(starts, ends);
			}
			train.setClassIndex(train.numAttributes() - 1);
			test.setClassIndex(test.numAttributes() - 1);

			// For each classifier
			for (int k = 0; k < classifiers.length; k++) {

				classifiers[k] = new KNN(sample, data, 2, 2);

				if (classifiers[k].weightedError == 0 || classifiers[k].weightedError >= 0.5) {
					Arrays.fill(wk, 1.0 / data.numInstances());
					k--;
					System.out.println("k--");
				} else {
					classifiers[k].Bk = classifiers[k].weightedError / (1 - classifiers[k].weightedError);
					double[] newWK = new double[wk.length];
					for (int j = 0; j < wk.length; j++) {
						double numerator = wk[j] * Math.pow(classifiers[k].Bk, 1 - classifiers[k].I[j]);
						double denominator = 0;
						for (int n = 0; n < data.numInstances(); n++) {
							denominator += wk[n] * Math.pow(classifiers[k].Bk, 1 - classifiers[k].I[j]);
						}
						newWK[j] = numerator / denominator;
					}
					wk = newWK;
				}
			} // End for each classifier

			// Check the support for each class in each instance and assign to it the class with max support.
			double[] supports = new double[data.numClasses()];
			Arrays.fill(supports, 0);

			classPredicted = new double[test.numInstances()];
			for (int x = 0; x < test.numInstances(); x++) {
				for (int t = 0; t < test.numClasses(); t++) {
					double accumulator = 0;
					for (KNN classifier : classifiers) {
						if (classifier.classPredicted[x + testStart] == t)
							accumulator += Math.log(1.0 / classifier.Bk);
					}
					supports[t] = accumulator;
				}
				classPredicted[x] = indexMax(supports);
//				System.out.println("Predice: " + classPredicted[x] + "\tEs: " + test.instance(x).classValue() + "\t\tEl array es: (" + supports[v2] + ", " + supports[1] + ", " + supports[2] + ")");
			}

			int correctClassified = 0;
			for (int n = 0; n < test.numInstances(); n++) {
				if (test.instance(n).classValue() == classPredicted[n]) correctClassified++;
			}
			accuracies[i] = ((double)correctClassified) / test.numInstances();
//			System.out.println("accuracy = " + accuracies[i]);
		}
	}

	private static void rouletteWheel(int[] starts, int[] ends) {
		sample.delete();
		double sumWK = 0;
		double[] auxWK = new double[train.numInstances()];
		int indexWK = 0;
		int auxCheckEnd = 0;
		for (int i = starts[0]; i < ends[ends.length-1]; i++) {
			if (i==ends[auxCheckEnd]) {
				auxCheckEnd++; // If error put break? But shouldn't fail?
				i = starts[auxCheckEnd];
			}
			sumWK += wk[i];
			auxWK[indexWK] = wk[i];
			indexWK++;
		}

		for (int i = 0; i < train.numInstances(); i++) {
			double randomNumber = randomGenerator.nextDouble() * sumWK;
			int j = 0;
			double accumulation = auxWK[j];
			while (accumulation < randomNumber) {
				j++;
				accumulation += auxWK[j];
			}
			sample.add(train.instance(j));
		}
	}

	private static class KNN {

		private double weightedError;
		private double Bk;
		private double[] classPredicted;
		private int[] I;

		private KNN(Instances train, Instances test, int k, int m) {

			weightedError = 0;
			classPredicted = new double[test.numInstances()];
			I = new int[test.numInstances()];

			for (int testIndex = 0; testIndex < test.numInstances(); testIndex++) {
				Instance testInstance = test.instance(testIndex);
				double distances[] = new double[k];
				Arrays.fill(distances, Double.MAX_VALUE);
				double nearestNeighbors[] = new double[k];
				Arrays.fill(nearestNeighbors, Double.MAX_VALUE);
				for (int trainIndex = 0; trainIndex < train.numInstances(); trainIndex++) {
					Instance trainInstance = train.instance(trainIndex);
					double distanceBetweenInstances = distance(testInstance, trainInstance, m);
					if (distanceBetweenInstances < distances[k - 1])
						insertOrdered(distanceBetweenInstances, distances, trainInstance.classValue(), nearestNeighbors);
				}

				double[] classValues = new double[testInstance.numClasses()];
				for (double nearestNeighbor : nearestNeighbors) {
					classValues[(int) nearestNeighbor]++;
				}
				classPredicted[testIndex] = indexMax(classValues);
				I[testIndex] = (classPredicted[testIndex] == testInstance.classValue()) ? 0 : 1;
				weightedError += wk[testIndex]*I[testIndex];
//				System.out.println("Predice: " + classPredicted[testIndex] + "\tEs: " + testInstance.classValue());
			}
//			int aux = v2;
//			for (int i = v2; i < I.length; i++) {
//				aux += I[i];
//			}
//			System.out.println("El KNN falla: " + aux);
		}

		private double distance(Instance i1, Instance i2, int m) {
			double distance = 0;

			for (int index = 0; index < i1.numAttributes() - 1; index++) {
				if (i1.attribute(index).isNumeric()) {
					distance += Math.pow(Math.abs(i1.value(index) - i2.value(index)), m);
				} else if (i1.attribute(index).isNominal()) {
					//If we found a nominal attribute we set the distance to v2 when they are equal, otherwise the distance is set to 1
					if (i1.value(index) != i2.value(index))
						distance += 1;
				}
			}
			return Math.pow(distance, 1.0 / m);
		}

		private void insertOrdered(double distanceBetweenInstances, double[] distances, double classValue, double[] nearestNeighbors) {
			for (int i = 0; i < distances.length; i++) {
				if (distanceBetweenInstances <= distances[i]) {
					for (int j = distances.length-2; j >= i ; j--) {
						distances[j + 1] = distances[j];
						nearestNeighbors[j + 1] = nearestNeighbors[j];
					}
					distances[i] = distanceBetweenInstances;
					nearestNeighbors[i] = classValue;
					break;
				}
			}
		}
	}

	private static Instances filterNormalize(Instances data) {
		Normalize normalize = new Normalize();
		try {
			normalize.setInputFormat(data);
			data = Filter.useFilter(data, normalize);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return data;
	}

	private static Instances filterRemoveRange(Instances data, String range) {
		RemoveRange remRange = new RemoveRange();
		try {
			remRange.setInputFormat(data);
			remRange.setInstancesIndices(range);
			data = Filter.useFilter(data, remRange);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return data;
	}

	public static double indexMax(double[] array) {
		double max = 0;
		int maxIndex = 0;
		for (int i = 0; i < array.length; i++) {
			if (array[i] > max) {
				max = array[i];
				maxIndex = i;
			}
		}
		return maxIndex;
	}

}