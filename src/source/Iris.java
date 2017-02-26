package source;

import java.io.File;
import java.util.Random;

import javax.swing.JFrame;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

public class Iris {
	
	public static void main(String args[]) throws Exception {

		infoGain("data/iris_fil.arff");
		incrementalNB("data/iris_fil.arff", "data/iris_fil_test.arff");
		incrementalIBk("data/iris_fil.arff", "data/iris_fil_test.arff");
		drawROCCurve("data/iris_fil.arff");
	}

	public static void infoGain(String dataset) throws Exception {
		System.out.println("Info Gain (entropy)  Example");
		DataSource source = new DataSource(dataset);
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
	
	public static void incrementalNB(String dataset, String testDataset) throws Exception {
		// incremental Naive Bayes classifier
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File(dataset));
		Instances dataStructure = loader.getStructure();
		dataStructure.setClassIndex(dataStructure.numAttributes() - 1);
		NaiveBayesUpdateable nb = new NaiveBayesUpdateable();
		nb.buildClassifier(dataStructure);
		
		Instance current;
		while ((current = loader.getNextInstance(dataStructure)) != null) {
//			System.out.println("instance " + current);
			nb.updateClassifier(current);
		}
		
		// specify data source
		DataSource source = new DataSource(testDataset);
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
	
	public static void incrementalIBk(String dataset, String testDataset) throws Exception {
		// incremental instance based classifier
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File(dataset));
		Instances dataStructure = loader.getStructure();
		dataStructure.setClassIndex(dataStructure.numAttributes() - 1);
		IBk ibk = new IBk();
		ibk.buildClassifier(dataStructure);
		Instances inMemoryData = new Instances(dataStructure, 20);
		Instance current;
		int numMem = 0;
		while ((current = loader.getNextInstance(dataStructure)) != null) {
			System.out.println("instance " + current);
			ibk.updateClassifier(current);
			// inMemoryStore for only 200 instances
			if (numMem < 200) {
				inMemoryData.add(current);
				numMem++;
			}
		}

		inMemoryData.setClassIndex(inMemoryData.numAttributes() - 1); // for the data read
		// specify data source to evaluate against for prediction
		DataSource source = new DataSource(testDataset);
		// load the data for prediction
		Instances dataT = source.getDataSet();
		dataT.setClassIndex(dataT.numAttributes() - 1);
		Evaluation eTest = new Evaluation(inMemoryData); // cross validate inMemory instances
		eTest.crossValidateModel(ibk, inMemoryData, 10, new Random(1));
		System.out.println(eTest.toSummaryString("Results Training\n ", false));
		System.out.println("\n Testing Example \n ");
		// Evaluating the predicted values
		
		// Printing out each prediction
		for (int i = 0; i < dataT.numInstances(); i++) {
			double pred = ibk.classifyInstance(dataT.instance(i));
			System.out.print("Instance " + i);
			if (dataT.instance(i).isMissing(dataT.numAttributes() - 1))
				System.out.print(", actual: ?");
			else
				System.out.print(", actual: " + dataT.classAttribute().value((int) dataT.instance(i).classValue()));
			System.out.print(", predicted: " + dataT.classAttribute().value((int) pred));
			System.out.println(" with p=" + Double.toString(pred));

		}

		eTest = new Evaluation(dataT);
		eTest.evaluateModel(ibk, dataT);
		System.out.println(eTest.toSummaryString("Results Test\n ", false));
	}
	
	public static void drawROCCurve(String dataset) throws Exception {
		// load data
		DataSource source = new DataSource(dataset);
		Instances data = source.getDataSet();
		data.setClassIndex(data.numAttributes() - 1);

		// train classifier
		J48 cl = new J48();
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(cl, data, 10, new Random(1));

		// generate curve
		ThresholdCurve tc = new ThresholdCurve();
		int classIndex = 0;
		Instances result = tc.getCurve(eval.predictions(), classIndex);

		// plot curve
		ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
		vmc.setROCString("(Area under ROC = "+ Utils.doubleToString(tc.getROCArea(result), 4) + ")");
		vmc.setName(result.relationName());
		PlotData2D tempd = new PlotData2D(result);
		tempd.setPlotName(result.relationName());
		tempd.addInstanceNumberAttribute();
		// specify which points are connected
		boolean[] cp = new boolean[result.numInstances()];
		for (int n = 1; n < cp.length; n++)
			cp[n] = true;
		tempd.setConnectPoints(cp);
		// add plot
		vmc.addPlot(tempd);

		// display curve		
		JFrame frame = new javax.swing.JFrame("ROC Curve");
		frame.setSize(800, 500);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.getContentPane().add(vmc);
		frame.setVisible(true);
	}

}
