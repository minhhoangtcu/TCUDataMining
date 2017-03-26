package lab2.sample;
/**
 * "Selecting attributes (Intermediate)"
 *
 * Select relevant attributes, apply principal component analysis.
 *
 * @author http://bostjankaluza.net
 * 
 * modified and expanded by Antonio Sanchez  2017
 */

import java.util.Random;

import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.ReliefFAttributeEval;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class InfoGainAttribute {

	public static void main(String args[]) throws Exception {

		infoGainExample();
		testInfoGain();

	}

	public static void infoGainExample() throws Exception {
		System.out.println("Info Gain (entropy)  Example");
		DataSource source = new DataSource("data/iris.csv");
		Instances data = source.getDataSet();
		weka.attributeSelection.AttributeSelection attSelect = new weka.attributeSelection.AttributeSelection(); // package
																													// weka.attributeSelection.AttributeSelection!
		InfoGainAttributeEval eval = new InfoGainAttributeEval();

		Ranker search = new Ranker();
		attSelect.setEvaluator(eval);
		attSelect.setSearch(search);
		attSelect.SelectAttributes(data);
		int[] indices = attSelect.selectedAttributes();
		System.out.println(attSelect.toResultsString());
		System.out.println(Utils.arrayToString(indices));

	}

	public static void testInfoGain() throws Exception {

		DataSource source = new DataSource("data/contact-lenses.arff");
		Instances data = source.getDataSet();
		AttributeSelection filter = new AttributeSelection(); // package
																// weka.filters.supervised.attribute!
		InfoGainAttributeEval evalA = new InfoGainAttributeEval();
		Ranker search = new Ranker();
		// Ranker attribute options
		String[] optionsR = new String[2];
		optionsR[0] = "-N";
		optionsR[1] = "-1"; // number of attributes
		search.setOptions(optionsR);
		filter.setEvaluator(evalA);
		filter.setSearch(search);
		filter.setInputFormat(data);
		// generate new data
		Instances newData = Filter.useFilter(data, filter);
		System.out.println(newData.numAttributes());
		newData.setClassIndex(newData.numAttributes() - 1);
		// decision trees options
		String[] options = new String[1];
		options[0] = "-U";
		J48 tree = new J48();
		System.out.println(tree);
		// build classifier
		tree.buildClassifier(newData);
		System.out.println(tree); // output classifier
		// evaluating the classifier model
		Evaluation eval = new Evaluation(newData);
		eval.crossValidateModel(tree, newData, 10, new Random(1));
		// output evaluation
		System.out.println(eval.toSummaryString("Results\n ", false));
		// print the confusion matrix
		System.out.println(eval.toMatrixString());

	}

}