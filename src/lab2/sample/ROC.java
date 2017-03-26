package lab2.sample;
/**
 

* "Testing and evaluating your models (Simple)"
 *
 * Test and estimate model performance.
 * 
 * @author http://bostjankaluza.net
 * modified by Antonio Sanchez
 */

import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;

import java.text.DecimalFormat;
import java.util.Random;

import javax.swing.*;

import weka.core.*;
import weka.classifiers.evaluation.*;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.gui.visualize.*;

public class ROC {
	
	public static void main(String args[]) throws Exception {

		evaluate("data/bankSmall.arff");
		ROCCurve("data/bankSmall.arff");
	}

	public static void evaluate(String dataset) throws Exception {
	    DecimalFormat insF = new DecimalFormat("#,###,###");
	    DecimalFormat perF = new DecimalFormat("#,###,###.##");
	    DecimalFormat kF = new DecimalFormat("#,###,###.###");
		DataSource source = new DataSource(dataset);

		Instances data = source.getDataSet();
		data.setClassIndex(data.numAttributes() - 1);
		J48 classifier = new J48();
		classifier.buildClassifier(data);
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(classifier, data, 10, new Random(1)); // do cross validation
		//eval.evaluateModel(classifier, data);// evaluate the model with full train dataset
		System.out.println("------R E S U L T S ----------\n");
		// Printing out each prediction
		for (int i = 0; i < data.numInstances(); i++) {
			   double pred = classifier.classifyInstance(data.instance(i));
			   if(data.instance(i).isMissing(data.numAttributes() - 1) )
				   System.out.print(i+ ", actual: ?");
			    else System.out.print(i+", actual: " + data.classAttribute().value((int) data.instance(i).classValue()));
			   System.out.print(", predicted: " + data.classAttribute().value((int) pred)); 
			   System.out.println(" with p="+perF.format(pred));
		
		}
		System.out.print("\n     SUMMARY  \n" );
		System.out.print("Correct instances " + insF.format(eval.correct()));
		System.out.println(" with " + perF.format(eval.pctCorrect()) + " %");
		System.out.println("Kappa Value = " + kF.format(eval.kappa())+ "\n");
		//System.out.println(eval.toSummaryString("Results\n ", false));  // output summary report

		;
		System.out.println(eval.toMatrixString());

	}

	public static void ROCCurve(String dataset) throws Exception {

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