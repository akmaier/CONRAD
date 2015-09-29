package ProjectIrisKellermann;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import ProjectIrisKellermann.Model;
import ProjectIrisKellermann.Model.VariationType;

/**
 * @author Iris Kellermann
 */

public class ObserverPipeline {

	public static void main(String args[])
	{
		int imageSize = 200;
		int Ntrain = 20;
		int Ntest = 10;
		int channelCount = 10;
		double gaussValue = 50;
		
		//the model
		Model model = new Model(imageSize, imageSize);
		
		//the different object-present images
		Grid2D[] testModels = model.CreateTestModels(Ntrain, VariationType.ProjectionWOFilter);
		
		//the different background images
		Grid2D[] emptyImages = model.CreateEmptyImages(Ntrain, VariationType.ProjectionWOFilter);
				
		//the channel images
		SimpleMatrix channelMatrix = Channels.CreateChannelMatrix(channelCount, imageSize, gaussValue);
		
		
		//create template
		SimpleVector template = Observer.CreateTemplate(testModels, emptyImages, channelMatrix);
	
		
		//test and create ROC curve
		
		Grid2D[] testImages = model.CreateTestModels(Ntest, VariationType.ProjectionWrongPoisson);
        Grid2D[] emptyTestImages = model.CreateEmptyImages(Ntest, VariationType.ProjectionWrongPoisson);  
		
		ROC.ShowROC(testImages, emptyTestImages, Ntest, template, channelMatrix);		
	}
}

/*
 * Copyright (C) 2010-2014 - Iris Kellermann 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

