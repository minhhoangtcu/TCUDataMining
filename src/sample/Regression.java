
package dmv;
/**
 * "Regression models (Simple)"
 *
 * Build a model for numerical predictions.
 *
 * @author http://bostjankaluza.net
 */
import java.io.BufferedReader;
import java.io.FileReader;

import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.functions.GaussianProcesses;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.REPTree;

public class Regression{

	public static void main(String args[]) throws Exception{
		
		//load data
		Instances data = new Instances(new BufferedReader(new FileReader("dataset/house.arff")));
		data.setClassIndex(data.numAttributes() - 1);
		
		//build model
		//ZeroR gpModel = new ZeroR();
		//LinearRegression gpModel = new LinearRegression();
		//REPTree gpModel = new REPTree();
		//SMOreg gpModel = new SMOreg();
		//MultilayerPerceptron gpModel = new MultilayerPerceptron();
		
		GaussianProcesses gpModel = new GaussianProcesses();
		gpModel.buildClassifier(data); //the last instance with missing class is not used
		System.out.println(gpModel);
		
		
		//classify the last instance
		Instance myHouse = data.lastInstance();
		double price = gpModel.classifyInstance(myHouse);
		System.out.println("My house ("+myHouse+"): "+price);
	}

}
