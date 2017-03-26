package dmv;

/**
 * "Associations rules (Simple)"
 *
 * Find requent patterns.
 *
 * @author http://bostjankaluza.net
 * modifed by Antonio Sanchez
 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.text.DecimalFormat;

import weka.core.Instances;
import weka.associations.Apriori;
import weka.classifiers.trees.J48;

public class AssociationRules{

	public static void main(String args[]) throws Exception{
		DecimalFormat insF = new DecimalFormat("#,###,###");
	    DecimalFormat perF = new DecimalFormat("#,###,###.##");
	    DecimalFormat kF = new DecimalFormat("#,###,###.###");
		//load data
		Instances data = new Instances(new BufferedReader(new FileReader("data/contactLenses.arff")));
		
		//build model
		Apriori model = new Apriori();
		// Apriori options
					String[] options = new String[4];
					options[0] = "-N";
					options[1] = "20";  //   default 10 
					options[2] = "-T";
					options[3] = "2";  // by leverage  default 9 confidence
					// <0=confidence | 1=lift | 2=leverage | 3=Conviction>
					 
					model.setOptions(options);
		model.buildAssociations(data); 
		System.out.println(model);
		
	}

}
