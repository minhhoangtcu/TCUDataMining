package lab3.source;

/**
 * "Recommendation system (Advanced)"
 *  A simple example of a recommender based on collaborative filtering 
 * Implement a recommendation system based on collaborative filtering.
 * http://en.wikipedia.org/wiki/Collaborative_filtering
 * basic ideas taken from http://csci.viu.ca/%7Ebarskym/
 * @author http://bostjankaluza.net
 * modifed by Antonio Sanchez
 */

import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.neighboursearch.LinearNNSearch;

import java.text.DecimalFormat;
import java.util.*;

public class Recommender {
	public static boolean verbose = false;
	public static void main(String[] args) throws Exception {
		DecimalFormat pF = new DecimalFormat("#0.00");
		
		// read learning dataset
		DataSource source = new DataSource("data/movieRatings.arff");
		Instances dataset = source.getDataSet();
		
		// read user data
		source = new DataSource("data/user.arff");
		Instances userRating = source.getDataSet();
		Instance userData = userRating.firstInstance();

		LinearNNSearch kNN = new LinearNNSearch(dataset);
		Instances neighbors = null;
		double[] distances = null;
       // consider 5 close neighbours
		try {
			neighbors = kNN.kNearestNeighbours(userData, 5);
			distances = kNN.getDistances();
		} catch (Exception e) {
			if (verbose) System.out.println("Neighbors could not be found.");
			return;
		}

		double[] similarities = new double[distances.length];
		for (int i = 0; i < distances.length; i++) {
			similarities[i] = 1.0 / distances[i];
			if (verbose) System.out.println(" percentage similarity for neighbour " + i + " is " +  pF.format(similarities[i]*100) + "%");
			if (verbose) System.out.println(" distance from neighbour " + i + " is " +  pF.format(distances[i]) );
		}

		Map<String, List<Integer>> recommendations = new HashMap<String, List<Integer>>();
		for(int i = 0; i < neighbors.numInstances(); i++){
			Instance currNeighbor = neighbors.instance(i);

			for (int j = 0; j < currNeighbor.numAttributes(); j++) {
				// item is not ranked by the user, but is ranked by neighbors 
				if (userData.value(j) < 1) {
					// retrieve the name of the movie
					String attrName = userData.attribute(j).name();
					List<Integer> lst = new ArrayList<Integer>();
					if (recommendations.containsKey(attrName)) {
						lst = recommendations.get(attrName);
					}
					
					lst.add((int)currNeighbor.value(j));
					recommendations.put(attrName, lst);
				}
			}

		}

		List<RecommendationRecord> finalRanks = new ArrayList<RecommendationRecord>();

		Iterator<String> it = recommendations.keySet().iterator();
		while (it.hasNext()) {
			String atrName = it.next();
			double totalImpact = 0;
			double weightedSum = 0;
			List<Integer> ranks = recommendations.get(atrName);
			for (int i = 0; i < ranks.size(); i++) {
				int val = ranks.get(i);
				totalImpact += similarities[i];
				weightedSum += (double) similarities[i] * val;
				if (verbose) System.out.println( i +  " TI "+ pF.format(totalImpact) + " WS " + pF.format(weightedSum));
				
			}
			RecommendationRecord rec = new RecommendationRecord();
			rec.attributeName = atrName;
			// note that if movie has not been rated the score would be 0
			rec.score = weightedSum / totalImpact;
			if (verbose) System.out.println( rec.attributeName  +  " score is  "+ pF.format(rec.score) );
			
			finalRanks.add(rec);
		}
		Collections.sort(finalRanks);

		
		double sum =0;
		
		
		// the higher the value the better choice 
		System.out.println(" First choice " + finalRanks.get(0) );
		System.out.println(" Second choice " + finalRanks.get(1));
		System.out.println(" Third choice " + finalRanks.get(2));
		System.out.println(" Fourth choice " + finalRanks.get(3));
		System.out.println(" Fifth choice " + finalRanks.get(4));
	}

	static class RecommendationRecord implements Comparable<RecommendationRecord> {
		public double score;
		public String attributeName;
		DecimalFormat cF = new DecimalFormat("#0.0#");
		public int compareTo(RecommendationRecord other) {
			if (this.score > other.score)
				return -1;
			if (this.score < other.score)
				return 1;
			return 0;
		}
		
		public String toString() {
			return " Movie recommended " + attributeName + ": " + cF.format(score);
		}
	}
}
