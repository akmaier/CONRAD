package edu.stanford.rsl.tutorial.modelObserver;

import java.util.Arrays;

import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;

/**
 * Given two sets of classifier scores X and Y, the ROC class can be used to compute and show ROC statistics
 * like ROC curve, false positive fraction / specificity, and sensitivity
 * 
 * @author Frank Schebesch
 */

public class ROC {

	int NC1;		// number of samples of first class (signal class)
	int NC2;		// number of samples of second class (non-signal class)
	int N;			// number of samples of both classes
	int Nthresh;	// maximum number of threshold steps (doubles not excluded)

	double[] C1scores;	// scores of first class
	double[] C2scores;	// scores of second class
	double[] allScores; // concatenation of all class scores

	double[] fpf;	// array of false positive fractions for given thresholds on the ROC statistic
	double[] sens;	// array of sensitivities for given thresholds on the ROC statistic

	boolean flipScoreOrientation;
	
	/**
	 * Constructor
	 * @param X : scores of first class (signal class)
	 * @param Y : scores of second class  (non-signal class)
	 * @param flipScoreOrientation : indicate of score logic is upside down (depends on score statistic)
	 */
	public ROC(double[] X, double[] Y, boolean flipScoreOrientation) {

		this.flipScoreOrientation = flipScoreOrientation;
		
		NC1 = X.length;
		NC2 = Y.length;
		N = NC1 + NC2;
		Nthresh = N + 2;
		
		C1scores = X.clone();
		C2scores = Y.clone();
		Arrays.sort(C1scores);
		Arrays.sort(C2scores);
		
		allScores = new double[N];
		System.arraycopy(C1scores, 0, allScores, 0, NC1);
		System.arraycopy(C2scores, 0, allScores, NC1, NC2);
	}
	
	/**
	 * Short constructor
	 * @param X : scores of first class (signal class)
	 * @param Y : scores of second class  (non-signal class)
	 */
	public ROC(double[] X, double[] Y) {
		
		this(X,Y,false);
	}
	
	/**
	 * Example on how to use the ROC class 
	 * @param args : not used
	 */
	public static void main(String[] args) {
		
//		double[] X = new double[]{0,0,1,1,2,3,4,5,5,5,6,8};
//		double[] Y = new double[]{3,3,5,6,7,7,8,8,8,9,9,10,10};
		
		// some arbitrary class scores
		double[] X = new double[] { -7, -6, -6, -5, -5, -5, -5, -3, -1, -1, 0, 2, 5 };
		double[] Y = new double[] { -6, -2, 0, 1, 3, 5, 6, 7, 7, 8, 9, 11, 15};
				
		// create the ROC object
		ROC roc = new ROC(X, Y);

		// evaluate sensitivity and false positive fraction (=1-specificity)
		roc.computeFractions();

		// get the respective fractions
		double[] fpf_ = roc.getFPF();
		double[] sens_ = roc.getSensitivity();
		
		// show ROC curve
		VisualizationUtil.createPlot(fpf_, sens_, "ROC", "fpf", "sens").show();		
		roc.show(); // alternative method
	}

	/**
	 * Computation of false positive fraction and sensitivity
	 */
	public void computeFractions() {

		int[] TP = new int[Nthresh];
		int[] FN = new int[Nthresh];
		int[] FP = new int[Nthresh];
		int[] TN = new int[Nthresh];

		// vector of scores
		SimpleVector vals = new SimpleVector(N);
		for (int i = 0; i < N; ++i) {
			vals.setElementValue(i, allScores[i]);
		}

		// generate vector of thresholds
		SimpleVector t = new SimpleVector(Nthresh);

		double[] tmp = allScores.clone();
		Arrays.sort(tmp);
		t.setElementValue(0, vals.min() - 1);
		for (int i = 1; i < N + 1; ++i) {
			t.setElementValue(i, tmp[i - 1]);
		}
		t.setElementValue(N + 1, vals.max() + 1);

		// count number of false/true-positives and negatives
		for (int i = 0; i < Nthresh; ++i) {
			
			TP[i] = 0;
			FN[i] = 0;
			FP[i] = 0;
			TN[i] = 0;
			
			// TODO: check consistency for equal elements 
			for (int j = 0; j < NC1; ++j) {
				
				if (!flipScoreOrientation) {
					
					if (vals.getElement(j) < t.getElement(i))
						TP[i]++;
					else
						FN[i]++;
				}
				else{ // inverse logic
					
					if (vals.getElement(j) > t.getElement(i))
						TP[i]++;
					else
						FN[i]++;					
				}
			}
			for (int j = NC1; j < N; ++j) {
				
				if (!flipScoreOrientation) {
					
					if (vals.getElement(j) < t.getElement(i))
						FP[i]++;
					else
						TN[i]++;
				}
				else{ // inverse logic
					
					if (vals.getElement(j) > t.getElement(i))
						FP[i]++;
					else
						TN[i]++;						
				}
			}
		}

		fpf = new double[Nthresh];
		sens = new double[Nthresh];

		for (int i = 0; i < Nthresh; ++i) {
			fpf[i] = (double) FP[i] / (double) (FP[i] + TN[i]);
			sens[i] = (double) TP[i] / (double) (TP[i] + FN[i]);
		}
	}
	
	/**
	 * Show ROC plot
	 */
	public void show(){
		
		if (fpf!=null && sens !=null)
			VisualizationUtil.createPlot(fpf, sens, "ROC", "false positive fraction", "sensitivity").show();
	}
	
	/**
	 * Getter for fpf values
	 * @return : array of false positive fractions for each threshold
	 */
	public double[] getFPF() {
		
		if(this.fpf==null)
			this.computeFractions();
		
		return this.fpf;
	}
	
	/**
	 * Getter for sensitivity values
	 * @return : array of sensitivities for each threshold
	 */
	public double[] getSensitivity() {
		
		if(this.sens==null)
			this.computeFractions();
		
		return this.sens;
	}
	
	/**
	 * Compute and get specificity values
	 * @return : array of specificities for each threshold
	 */
	public double[] getSpecificity() {
		
		if(this.fpf==null)
			this.computeFractions();
		
		double[] spec = this.fpf.clone();
		
		for (int i = 0; i < spec.length; i++)
			spec[i] = 1.0 - spec[i]; 
		
		return spec;
	}
	
	/**
	 * Getter for class scores
	 * @return : scores of first class
	 */
	public double[] getC1scores() {
		return this.C1scores;
	}
	
	/**
	 * Getter for class scores
	 * @return : scores of second class
	 */
	public double[] getC2scores() {
		return this.C2scores;
	}

	/**
	 * Getter for class scores
	 * @return : scores of both classes
	 */
	public double[] getAllScores() {
		return this.allScores;
	}
	
	/**
	 * Get ratios of sample amounts
	 * @return : array of shares for first and second class (in named order) 
	 */
	public double[] getPrevalenceWeights() {
		return new double[]{ (double) NC1/(double) N, (double) NC2/(double) N };
	}
}

/*
 * Copyright (C) 2010-2016 - Frank Schebesch, Iris Kellermann, Priyal Patel, Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License
 * (GPL).
 */
