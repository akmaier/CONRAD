package edu.stanford.rsl.tutorial.modelObserver;

import java.util.Arrays;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;

/**
 * @author Iris Kellermann
 */

public class ROCold {

	/**
	 * Calculates and displays the ROC curve for the given test images. Calculates and displays the SNR.
	 * @param testImages The object test images.
	 * @param emptyTestImages The test images without objects.
	 * @param Ntest The number of test images in each category.
	 * @param template The template of the observer.
	 * @param channelMatrix The matrix with the channel images.
	 */
	public static void ShowROC(Grid2D[] testImages, Grid2D[] emptyTestImages, int Ntest, SimpleVector template, SimpleMatrix channelMatrix)
	{
		SimpleVector vals = new SimpleVector(2 * Ntest); 
        
		// get observer result values
        for(int i = 0; i < Ntest; ++i) 
        {   	
        	vals.setElementValue(i,Observer.GetResultValue(testImages[i], template, channelMatrix));
            vals.setElementValue(i+Ntest,Observer.GetResultValue(emptyTestImages[i], template, channelMatrix));
        }
        
        // compute SNR
        
        // mean
        double sumObj = 0;
        double sumEmp = 0;
        for(int i = 0; i < Ntest; ++i)
        {
        	sumObj += vals.getElement(i);
        	sumEmp += vals.getElement(i+Ntest);
        }
        
        double meanObj = sumObj / Ntest;
        double meanEmp = sumEmp / Ntest;
        
        // variance
        double sumVarObj = 0;
        double sumVarEmp = 0;
        for(int i = 0; i < Ntest; ++i)
        {
        	sumVarObj += (meanObj - vals.getElement(i)) * (meanObj - vals.getElement(i));
        	sumVarEmp += (meanEmp - vals.getElement(i + Ntest)) * (meanEmp - vals.getElement(i + Ntest));
        }
        
        double varObj = sumVarObj / Ntest;
        double varEmp = sumVarEmp / Ntest;
        
        double SNR = (meanObj - meanEmp) / (Math.sqrt((varObj + varEmp) / 2));
        
        System.out.println(SNR);
        

        SimpleVector t = new SimpleVector(2 * Ntest + 2);
        
        int[] TP = new int[2 * Ntest + 2];
        int[] FP = new int[2 * Ntest + 2];
        int[] TN = new int[2 * Ntest + 2];
        int[] FN = new int[2 * Ntest + 2];
        double[] tmp =  vals.copyAsDoubleArray();
        
        Arrays.sort(tmp);
        
        t.setElementValue(0, vals.min() - 1);
        
        for (int i=1; i < 2 * Ntest + 1; ++i)
        {
            t.setElementValue(i, tmp[i-1]);
        }
        
        t.setElementValue(2 * Ntest + 1,vals.max() + 1);

        // count number of false/true-positives and negatives
        for (int i = 0; i < 2 * Ntest + 2; ++i)
        {
            TP[i] = 0;
            for (int j = 0; j < Ntest; ++j){
                if (vals.getElement(j) < t.getElement(i))
                    TP[i]++;
                else
                    FN[i]++;
                if (vals.getElement(j + Ntest) < t.getElement(i))
                    FP[i]++;
                else
                    TN[i]++;
            }
        }

        double[] fpf = new double[2 * Ntest + 2];
        double[] sens = new double[2 * Ntest + 2];

        for (int i = 0; i < 2 * Ntest + 2; ++i){
            fpf[i] = (double)FP[i]/(double)(FP[i]+TN[i]);
            sens[i] = (double)TP[i]/(double)(TP[i]+FN[i]);
        }

        //create ROC plot
        VisualizationUtil.createPlot(fpf,sens,"ROC","fpf","sens").show();
	}
}

/*
 * Copyright (C) 2010-2014 - Iris Kellermann 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/