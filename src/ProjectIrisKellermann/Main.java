package ProjectIrisKellermann;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import ProjectIrisKellermann.Model;

public class Main {

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
		Grid2D channelImages = new Grid2D(channelCount, imageSize * imageSize);
				
		for(int x = 0; x < channelCount; ++x)
		{
			Grid2D channelImage = ImageHelper.ConvertImageToArray(Channels.CreateLGChannelImage(imageSize, imageSize, x));
			
			for(int y = 0; y < imageSize; ++y)
			{
				channelImages.putPixelValue(x, y, channelImage.getPixelValue(0, y));
			}
		}
		
		//creation of the template
		Grid2D s_v;
		Grid2D C_v;
		
		Grid2D v;
		
		
		
		
		//new ImageJ();
	}
}


