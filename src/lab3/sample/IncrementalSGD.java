
package dmv;
/**
 
* "Training an incremental SGD classifier "
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
import weka.classifiers.functions.SGD;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;

//
public class IncrementalSGD {

	public static void main(String args[]) throws Exception{
    	//incremental instance based classifier
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File("dataset/tipsOK.arff"));
		Instances dataStructure = loader.getStructure();
		dataStructure.setClassIndex(dataStructure.numAttributes() - 1);
		SGD sgd = new SGD();
		sgd.buildClassifier(dataStructure);
		Instances inMemoryData = new Instances(dataStructure,20);
		Instance current; int numMem=0;
		int numMemF = 0;
		
		while ((current = loader.getNextInstance(dataStructure)) != null)
	        {//System.out.println("instance " + current);  
			sgd.updateClassifier(current);
			// inMemoryStore  for only 10% instances
			   if (numMemF>10 )
			   { inMemoryData.add(current); numMemF = 0;
			                                     numMem++; } 
			   
			      else  numMemF++; 
			}
		   System.out.println("instances in Memory " + numMem);  
		        inMemoryData.setClassIndex(inMemoryData.numAttributes() - 1);  // for the data read
				// specify data source to evaluate against for prediction
				DataSource source = new DataSource("dataset/tipsOK.arff");
				// load the data for prediction
				Instances dataT = source.getDataSet();
				dataT.setClassIndex(dataT.numAttributes() - 1);
				Evaluation eTest = new Evaluation(inMemoryData);  // cross validate inMemory instances
				eTest.crossValidateModel(sgd, inMemoryData, 10, new Random(1));
				//eTest.evaluateModel(sgd, inMemoryData);
				System.out.println(eTest.toSummaryString("Results\n ", false));
				// print the confusion matrix
				System.out.println(eTest.toMatrixString());
				//System.out.println(eTest.toSummaryString("Results Training\n ", false));
				System.out.println("\n Testing Example \n ");
				// Evaluating the predicted values
				// Printing out each prediction
				/*for (int i = 0; i < dataT.numInstances(); i++) {
					   double pred = sgd.classifyInstance(dataT.instance(i));
					   System.out.print("Instance " + i);
					   if(dataT.instance(i).isMissing(dataT.numAttributes() - 1) )
						   System.out.print(", actual: ?");
					    else System.out.print(", actual: " + dataT.classAttribute().value((int) dataT.instance(i).classValue()));
					   System.out.print(", predicted: " + dataT.classAttribute().value((int) pred)); 
					   System.out.println(" with p="+Double.toString(pred));
				
				}
				*/
				
			   // eTest = new Evaluation(dataT);	
				//eTest.evaluateModel(sgd, dataT);
				//System.out.println(eTest.toSummaryString("Results Test\n ", false));
			
				
				
	}

}
