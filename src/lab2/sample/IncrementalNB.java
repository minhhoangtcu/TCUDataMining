
package lab2.sample;
/**
 
* "Training a incremental Naive Bayes classifier "
 *
 * Build and use a classifier.
 *
 * @author http://bostjankaluza.net
 * 
 * modified and expanded by Antonio Sanchez  2017
 */

import java.io.File;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;

public class IncrementalNB {

	public static void main(String args[]) throws Exception {
		// incremental Naive Bayes classifier
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File("data/bankSmall.arff"));
		Instances dataStructure = loader.getStructure();
		dataStructure.setClassIndex(dataStructure.numAttributes() - 1);
		NaiveBayesUpdateable nb = new NaiveBayesUpdateable();
		nb.buildClassifier(dataStructure);

		Instance current;
		while ((current = loader.getNextInstance(dataStructure)) != null) {
			System.out.println("instance " + current);
			nb.updateClassifier(current);
		}

		// specify data source
		DataSource source = new DataSource("data/bankSmalTest.arff");
		// load the data
		Instances data = source.getDataSet();
		data.setClassIndex(data.numAttributes() - 1);

		// Printing out each prediction
		for (int i = 0; i < data.numInstances(); i++) {
			double pred = nb.classifyInstance(data.instance(i));
			System.out.print("Instance " + i);
			if (data.instance(i).isMissing(data.numAttributes() - 1))
				System.out.print(", actual: ?");
			else
				System.out.print(", actual: " + data.classAttribute().value((int) data.instance(i).classValue()));
			System.out.print(", predicted: " + data.classAttribute().value((int) pred));
			System.out.println(" with p=" + Double.toString(pred));

		}
		Evaluation eTest = new Evaluation(data);
		eTest.evaluateModel(nb, data);
		// Print the results
		System.out.println(eTest.toSummaryString("Results\n ", true));

	}

}
