
package sample;
/**
 
* "Training an incremental IBK classifier "
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
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;


public class IncrementalIBK {

	public static void main(String args[]) throws Exception{
    	//incremental instance based classifier
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File("data/bankSmall.arff"));
		Instances dataStructure = loader.getStructure();
		dataStructure.setClassIndex(dataStructure.numAttributes() - 1);
		IBk ibk = new IBk();
		ibk.buildClassifier(dataStructure);
		Instances inMemoryData = new Instances(dataStructure,20);
		Instance current; int numMem=0;
		while ((current = loader.getNextInstance(dataStructure)) != null)
	        {System.out.println("instance " + current);  
			ibk.updateClassifier(current);
			// inMemoryStore  for only 200 instances
			   if (numMem < 200)  { inMemoryData.add(current);  numMem++; } 
			}
		
		        inMemoryData.setClassIndex(inMemoryData.numAttributes() - 1);  // for the data read
				// specify data source to evaluate against for prediction
				DataSource source = new DataSource("data/bankSmalTest.arff");
				// load the data for prediction
				Instances dataT = source.getDataSet();
				dataT.setClassIndex(dataT.numAttributes() - 1);
				Evaluation eTest = new Evaluation(inMemoryData);  // cross validate inMemory instances
				eTest.crossValidateModel(ibk, inMemoryData, 10, new Random(1));
				System.out.println(eTest.toSummaryString("Results Training\n ", false));
				System.out.println("\n Testing Example \n ");
				// Evaluating the predicted values
				// Printing out each prediction
				for (int i = 0; i < dataT.numInstances(); i++) {
					   double pred = ibk.classifyInstance(dataT.instance(i));
					   System.out.print("Instance " + i);
					   if(dataT.instance(i).isMissing(dataT.numAttributes() - 1) )
						   System.out.print(", actual: ?");
					    else System.out.print(", actual: " + dataT.classAttribute().value((int) dataT.instance(i).classValue()));
					   System.out.print(", predicted: " + dataT.classAttribute().value((int) pred)); 
					   System.out.println(" with p="+Double.toString(pred));
				
				}
				
				
			    eTest = new Evaluation(dataT);	
				eTest.evaluateModel(ibk, dataT);
				System.out.println(eTest.toSummaryString("Results Test\n ", false));
			
				
				
	}

}
