package ProjectIrisKellermann;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.filtering.LogPoissonNoiseFilteringTool;
import ProjectIrisKellermann.Model;

public class Main {

	public static void main(String args[])
	{
		Model model = new Model(500,500);
		
		// model.show();
		
		/*Grid2D[] testModels = model.CreateTestModels(10);
		
		Grid2D[] emptyImages = model.CreateEmptyImages(10);
		
		for(int i = 0; i < testModels.length; ++i)
		{
			testModels[i].show();
		}
		
		for(int i = 0; i < emptyImages.length; ++i)
		{
			emptyImages[i].show();
		}*/
		
		Grid2D channelImage = Channels.CreateLGChannelImage(500, 500, 9);
		
		channelImage.show();
		
		try
		{
			System.out.println(Channels.MultiplyImages(model, channelImage));
		}
		catch(Exception e)
		{
			System.out.println("Exception");
		}
		
		//new ImageJ();
	}
}


