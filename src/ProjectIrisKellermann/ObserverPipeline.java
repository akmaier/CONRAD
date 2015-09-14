package ProjectIrisKellermann;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.InversionType;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import ProjectIrisKellermann.Model;

public class ObserverPipeline {

	public static void main(String args[])
	{
		int imageSize = 200;
		int imageCount = 10;
		int channelCount = 10;
		
		//the model
		Model model = new Model(imageSize, imageSize);
		
		
		model.show();
		
		//the different object-present images
		Grid2D[] testModels = model.CreateTestModels(imageCount);
		
		//the different background images
		Grid2D[] emptyImages = model.CreateEmptyImages(imageCount);
		
		//the channel images
		SimpleMatrix channelMatrix = Channels.CreateChannelMatrix(channelCount, imageSize);
		
		//result
		System.out.println(Observer.GetResultValue(model, testModels, emptyImages, channelMatrix));
	}
}


