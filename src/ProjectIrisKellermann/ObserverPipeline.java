package ProjectIrisKellermann;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.InversionType;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import ProjectIrisKellermann.Model;

/**
 * @author Iris Kellermann
 */

public class ObserverPipeline {

	public static void main(String args[])
	{
		int imageSize = 200;
		int imageCount = 5;
		int channelCount = 10;
		
		//the model
		Model model = new Model(imageSize, imageSize);
		
		model.show();
		
		//the different object-present images
		Grid2D[] testModels = model.CreateTestModels(imageCount);
		
		for(int i = 0; i < testModels.length; ++i)
		{
			testModels[i].show();
		}
		
		//the different background images
		Grid2D[] emptyImages = model.CreateEmptyImages(imageCount);
		
		for(int i = 0; i < emptyImages.length; ++i)
		{
			emptyImages[i].show();
		}
		
		//the channel images
		SimpleMatrix channelMatrix = Channels.CreateChannelMatrix(channelCount, imageSize);
		
		//result
		System.out.println(Observer.GetResultValue(model, testModels, emptyImages, channelMatrix));
	}
}


