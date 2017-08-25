package edu.stanford.rsl.tutorial.modelObserver;

import ij.ImageJ;
import java.util.Arrays;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import edu.stanford.rsl.tutorial.modelObserver.TestModelGenerator;
import edu.stanford.rsl.tutorial.modelObserver.TestModelGenerator.VariationType;

/**
 * @author Iris Kellermann
 */

public class ObserverPipeline {

	public static void main(String args[])
	{
		int imageSize = 200;
		int Ntrain = 20;
		int Ntest = 20;
		int channelCount = 5;
		double gaussValue = 40;
		
		new ImageJ();
		
		//the model
		TestModelGenerator model = new TestModelGenerator(imageSize, imageSize);
		
		//the different object-present images
		Grid2D[] trainModels = model.CreateTestModels(Ntrain, VariationType.SimpleVariation);
		
		//the different background images
		Grid2D[] emptyTrainImages = model.CreateEmptyImages(Ntrain, VariationType.ProjectionWOFilter);
		
		//the channel images
		SimpleMatrix channelMatrix = Channels.CreateChannelMatrix(channelCount, imageSize, gaussValue);
				
		//create template
		SimpleVector template = Observer.CreateTemplate(trainModels, emptyTrainImages, channelMatrix);
			
		//create test data
		Grid2D[] testImages = model.CreateTestModels(Ntest, VariationType.ProjectionPoisson);
        Grid2D[] emptyTestImages = model.CreateEmptyImages(Ntest, VariationType.ProjectionWOFilter);
        
        // create ROC curve
		//ShowROC(testImages, emptyTestImages, Ntest, template, channelMatrix); // use of deprecated method
        (new ObserverPipeline()).computeStats(testImages, emptyTestImages, Ntest, template, channelMatrix); // recommended call for ROC computation
	}

	/**
	 * Successor of the method ShowROC(Grid2D[], Grid2D[], int, SimpleVector, SimpleMatrix)
	 * 
	 * Calculates and displays the ROC curve for the given test images. Calculates and displays the SNR.
	 * @param testImages The object test images.
	 * @param emptyTestImages The test images without objects.
	 * @param Ntest The number of test images in each category.
	 * @param template The template of the observer.
	 * @param channelMatrix The matrix with the channel images.
	 */
	public void computeStats(Grid2D[] testImages, Grid2D[] emptyTestImages, int Ntest, SimpleVector template, SimpleMatrix channelMatrix)
	{
		SimpleVector valsImage = new SimpleVector(2 * Ntest);
		SimpleVector valsEmpty = new SimpleVector(2 * Ntest); 
        
		// get observer result values
        for(int i = 0; i < Ntest; ++i) 
        {   	
        	valsImage.setElementValue(i,Observer.GetResultValue(testImages[i], template, channelMatrix));
        	valsEmpty.setElementValue(i+Ntest,Observer.GetResultValue(emptyTestImages[i], template, channelMatrix));
        }
        
        // compute SNR
        
        // mean
        double sumObj = 0;
        double sumEmp = 0;
        for(int i = 0; i < Ntest; ++i)
        {
        	sumObj += valsImage.getElement(i);
        	sumEmp += valsEmpty.getElement(i+Ntest);
        }
        
        double meanObj = sumObj / Ntest;
        double meanEmp = sumEmp / Ntest;
        
        // variance
        double sumVarObj = 0;
        double sumVarEmp = 0;
        for(int i = 0; i < Ntest; ++i)
        {
        	sumVarObj += (meanObj - valsImage.getElement(i)) * (meanObj - valsImage.getElement(i));
        	sumVarEmp += (meanEmp - valsEmpty.getElement(i + Ntest)) * (meanEmp - valsEmpty.getElement(i + Ntest));
        }
        
        double varObj = sumVarObj / Ntest;
        double varEmp = sumVarEmp / Ntest;
        
        double SNR = (meanObj - meanEmp) / (Math.sqrt((varObj + varEmp) / 2));
        
        System.out.println("SNR="+SNR);
        
        // compute ROC curve (and statistics)
        
		ROC roc = new ROC(valsEmpty.copyAsDoubleArray(), valsImage.copyAsDoubleArray());
		roc.computeFractions();		
		roc.show();
	}
	
	/**
	 * Deprecated method (uses custom ROC implementation):
	 * Please use
	 * computeStats(Grid2D[], Grid2D[], int, SimpleVector, SimpleMatrix)
	 * instead of the ShowROC(...) method in future implementations and adapt existing code.
	 * (You can also use the ROC class which allows a more general access to ROC statistics.)
	 * 
	 * Calculates and displays the ROC curve for the given test images. Calculates and displays the SNR.
	 * @param testImages The object test images.
	 * @param emptyTestImages The test images without objects.
	 * @param Ntest The number of test images in each category.
	 * @param template The template of the observer.
	 * @param channelMatrix The matrix with the channel images.
	 */
	@Deprecated
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
        
        System.out.println("SNR="+SNR);
        
        // compute ROC
		 
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
 * Copyright (C) 2010-2017 - Iris Kellermann, Frank Schebesch
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

