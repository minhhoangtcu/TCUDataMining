package lab3.source;
/**
 * "Clustering (Simple)"
 *
 * Build, evaluate, and use clusters. Note, the code below requires 
 * lots of processing, which may cause your computer to hang for a while.
 *
 * @author http://bostjankaluza.net
 * modifed by Antonio Sanchez
 */

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Cobweb;
import weka.clusterers.EM;
import weka.clusterers.SimpleKMeans;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.ClassificationViaClustering;

public class Clustering {

	public static void main(String args[]) throws Exception {

		// load data
		Instances data = new Instances(new BufferedReader(new FileReader("data/titanic.arff")));

		// new instance of clusterer
		// SimpleKMeans model = new SimpleKMeans(); // Simple K means cluster
		EM model = new EM(); // Expectation Maximizer
		// build the clusterer
		model.buildClusterer(data);
		// System.out.println(model);
		// classifyByCluster();
		clusterClassify(); // classify a new entry
		// incrementalCluster();

		evaluate();

	}

	public static void clusterClassify() throws Exception {
		DecimalFormat cF = new DecimalFormat("#0.00#");
		// load data
		Instances data = new Instances(new BufferedReader(new FileReader("data/titanic.arff")));
		Instance inst = data.instance(0); // select instance 0 only
		data.delete(0); // eliminate instance 0 not to be used in the model

		// new instance of clusterer
		SimpleKMeans model = new SimpleKMeans(); // Simple K means cluster
//		EM model = new EM();		// EM Cluster options

		String[] options = new String[2];
		options[0] = "-N";
		options[1] = "3"; // three clusters

		model.setOptions(options);
		// build the clusterer
		model.buildClusterer(data);
		// System.out.println(model);

		int cls = model.clusterInstance(inst);
		System.out.println("this Instance belongs to Cluster num " + cls);

		double[] dist = model.distributionForInstance(inst);
		System.out.println("Ralative distances to the various Clusters ");
		for (int i = 0; i < dist.length; i++)
			System.out.println("Cluster " + i + ".\t" + cF.format(dist[i]));

	}

	public static void classifyByCluster() throws Exception {
		DecimalFormat cF = new DecimalFormat("#0.00#");
		// load data
		Instances data = new Instances(new BufferedReader(new FileReader("data/titanic.arff")));
		// SimpleKMeans model = new SimpleKMeans(); // Simple K means cluster
		ClassificationViaClustering model = new ClassificationViaClustering();

		// Cluster options
		data.setClassIndex(data.numAttributes() - 1); // for the data read
		String[] options = new String[2];
		options[0] = "-W";
		// options[1] = "weka.clusterers.SimpleKMeans"; // Simple K
		// clusterdriver 4
		options[1] = "weka.clusterers.EM"; // EM clusters
		// options[1] = "weka.clusterers.Cobweb"; // cobweb clusters
		model.setOptions(options);
		// build the classifier
		model.buildClassifier(data);
		Evaluation eTest = new Evaluation(data); // cross validate inMemory instances
		eTest.crossValidateModel(model, data, 10, new Random(1));
		// eTest.evaluateModel(model, data);
		System.out.println(eTest.toSummaryString("Results Training\n ", false));
		// print the confusion matrix
		System.out.println(eTest.toMatrixString());
	}

	public static void incrementalCluster() throws Exception {
		// load data
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File("data/titanic.arff"));
		Instances data = loader.getStructure();

		// train Cobweb
		Cobweb model = new Cobweb(); // the Cobweb allows for incremental construction yet very slow
		model.buildClusterer(data);
		Instance current;
		while ((current = loader.getNextInstance(data)) != null)
			model.updateClusterer(current);
		model.updateFinished();
		System.out.println(model);
		ClusterEvaluation eval = new ClusterEvaluation();
		// eval.setClusterer(model); // the cluster to evaluate
		eval.evaluateClusterer(data);
		System.out.println("# of clusters: " + eval.getNumClusters()); // output # of clusters

	}

	public static void evaluate() throws Exception {
		DecimalFormat cF = new DecimalFormat("#0.00#");
		Instances data = new Instances(new BufferedReader(new FileReader("data/titanic.arff")));

		EM model = new EM();
		// Cobweb model = new Cobweb(); // the Cobweb allows for incremental
		// construction yet very slow
		double logLikelyhood = ClusterEvaluation.crossValidateModel(model, data, 10, new Random(1));
		System.out.println(cF.format(logLikelyhood));

		ClusterEvaluation eval = new ClusterEvaluation();
		model.buildClusterer(data); // build clusterer
		eval.setClusterer(model); // the cluster to evaluate
		eval.evaluateClusterer(data); // data to evaluate the clusterer on

		System.out.println("# of clusters: " + eval.getNumClusters()); // output # of clusters

	}
}
