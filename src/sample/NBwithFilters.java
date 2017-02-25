package sample;
/**

 * "Training a classifier (Simple)"
 *
 * Build and use a classifier.
 *
 * @author http://bostjankaluza.net
 * modified and expanded by Antonio Sanchez  2017
 */

import java.io.File;
import java.io.IOException;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NominalToString;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToNominal;

public class NBwithFilters {
    //  a silly example of three filters
	public static void main(String args[]) throws Exception{
		// specify data source
		DataSource source = new DataSource("data/weather.arff");
		// load the data 
		Instances data = source.getDataSet();
		// output data
		System.out.println(data+"\n: "+data.numInstances()+" instances loaded without filters.");
		// construct the filters 
		NumericToNominal fil1 = new NumericToNominal();  
		NominalToString fil2 = new NominalToString(); 
		StringToNominal fil3 = new StringToNominal(); 
		// options for the filter only on attribute 2
		String[] optionsF = new String[2];
		optionsF[0] = "-R";                                    // "range"
		optionsF[1] = "2";                                     // only attribute 2
	                    
		fil1.setOptions(optionsF);                           // set options filter 1
		fil1.setInputFormat(data);                          // inform filter about dataset **AFTER** setting options
		Instances filData1 = Filter.useFilter(data, fil1); // apply filter 1
		fil2.setOptions(optionsF);                           // set options filter 2
		fil2.setInputFormat(filData1);       
		Instances filData2 = Filter.useFilter(filData1, fil2);   // apply filter 2
		fil3.setOptions(optionsF);                           // set options filter 3
		fil3.setInputFormat(filData2);       
		Instances filData3 = Filter.useFilter(filData2, fil3);   // apply filter 3
		System.out.println(filData3+"\n: "+filData3.numInstances()+" instances loaded with filter.");
		
		// use NB with dataset after filtered three times
		// Select the knowledge class
		filData3.setClassIndex(filData3.numAttributes() - 1);

		// decision trees options
			String[] options = new String[1];
			options[0] = "-U";
			NaiveBayes bs = new NaiveBayes();
			//bs.setOptions(options);
			
		
		//build classifier
		bs.buildClassifier(filData3);	
		
		System.out.println("After building classifier\n" + bs);  // output classifier
		// Printing out each prediction
		for (int i = 0; i < filData3.numInstances(); i++) {
			   double pred = bs.classifyInstance(filData3.instance(i));
			   System.out.print("Instance " + i);
			   System.out.print(", actual: " + filData3.classAttribute().value((int) filData3.instance(i).classValue()));
			   System.out.print(", predicted: " + filData3.classAttribute().value((int) pred)); 
			   System.out.println(" with p="+Double.toString(pred));
		
		}
		// evaluating the classifier model 
		Evaluation eval = new Evaluation(filData3);
		eval.crossValidateModel(bs, filData3, 10, new Random(1));
		// output evaluation
		System.out.println(eval.toSummaryString("Results\n ", false));
		// print the confusion matrix
		System.out.println(eval.toMatrixString());
		// save data set batch or incremental
		// saveDataset(data, false);  
	}
			
			
			public static void saveDataset(Instances dataset, boolean batchSave) throws IOException{
				ArffSaver saver = new ArffSaver();
				if(batchSave){
					System.out.println(dataset+ " instances saved batch.");
					saver.setInstances(dataset);
					saver.setFile(new File("data/testWB.arff"));
					saver.writeBatch();
				}
				else{ System.out.println(dataset+ " instances saved incremental.");
					saver.setRetrieval(ArffSaver.INCREMENTAL);
					saver.setInstances(dataset);
					saver.setFile(new File("data/testWIN.arff"));
					for(int i=0; i < dataset.numInstances(); i++){
						saver.writeIncremental(dataset.instance(i));
					}
					saver.writeIncremental(null);
				}
			}
			
}
