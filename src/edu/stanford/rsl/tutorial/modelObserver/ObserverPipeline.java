package edu.stanford.rsl.tutorial.modelObserver;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.tutorial.modelObserver.Model;
import edu.stanford.rsl.tutorial.modelObserver.Model.VariationType;

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
		Model model = new Model(imageSize, imageSize);
		
		//the different object-present images
		Grid2D[] trainModels = model.CreateTestModels(Ntrain, VariationType.SimpleVariation);
		
		//the different background images
		Grid2D[] emptyTrainImages = model.CreateEmptyImages(Ntrain, VariationType.ProjectionWOFilter);
		
		//the channel images
		SimpleMatrix channelMatrix = Channels.CreateChannelMatrix(channelCount, imageSize, gaussValue);
				
		//create template
		SimpleVector template = Observer.CreateTemplate(trainModels, emptyTrainImages, channelMatrix);
			
		//test and create ROC curve
		Grid2D[] testImages = model.CreateTestModels(Ntest, VariationType.ProjectionPoisson);
        Grid2D[] emptyTestImages = model.CreateEmptyImages(Ntest, VariationType.ProjectionWOFilter);
        
		ROC.ShowROC(testImages, emptyTestImages, Ntest, template, channelMatrix);		
	}
}

/*
 * Copyright (C) 2010-2014 - Iris Kellermann 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

