package ProjectIrisKellermann;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.InversionType;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import ProjectIrisKellermann.Model;

public class Main {

	public static void main(String args[])
	{
		int imageSize = 60;
		int imageCount = 10;
		int channelCount = 10;
		
		//the model
		Model model = new Model(imageSize, imageSize);
		
		
		model.show();
		
		System.out.println("1. ");
				
		//the different object-present images
		Grid2D[] testModels = model.CreateTestModels(imageCount);
		
		System.out.println("2. ");
		
		//the different background images
		Grid2D[] emptyImages = model.CreateEmptyImages(imageCount);
		
		System.out.println("3. ");
		
		//the channel images
		Grid2D channelImages = new Grid2D(channelCount, imageSize * imageSize);
				
		for(int x = 0; x < channelCount; ++x)
		{
			Grid2D channelImage = ImageHelper.ConvertImageToColumn(Channels.CreateLGChannelImage(imageSize, imageSize, x));
			
			for(int y = 0; y < imageSize; ++y)
			{
				channelImages.putPixelValue(x, y, channelImage.getPixelValue(0, y));
			}
		}
		
		try
		{
			System.out.println("4. ");
			
			Grid2D s = Observer.GetMeanDifferenceImage(testModels, emptyImages);
			Grid2D C_g = Observer.GetCovarianceMatrix(testModels, emptyImages);
			
			System.out.println("5. ");
			
			//convert Grid2D images to SimpleMatrix
			SimpleMatrix channelImagesMatrix = ImageHelper.ConvertGrid2DToSimpleMatrix(channelImages);
			SimpleMatrix sMatrix = ImageHelper.ConvertGrid2DToSimpleMatrix(ImageHelper.ConvertImageToColumn(s));
			SimpleMatrix C_gMatrix = ImageHelper.ConvertGrid2DToSimpleMatrix(C_g);
			
			SimpleMatrix modelMatrix = ImageHelper.ConvertGrid2DToSimpleMatrix(ImageHelper.ConvertImageToColumn(model));
			
			System.out.println("6. ");
			
			//calculate s_v, C_v and v
			SimpleMatrix s_vMatrix = SimpleOperators.multiplyMatrixProd(channelImagesMatrix.transposed(), sMatrix);
			SimpleMatrix C_vMatrix = SimpleOperators.multiplyMatrixProd(SimpleOperators.multiplyMatrixProd(channelImagesMatrix.transposed(), C_gMatrix), channelImagesMatrix);
			SimpleMatrix vMatrix = SimpleOperators.multiplyMatrixProd(channelImagesMatrix.transposed(), modelMatrix);
		
			System.out.println("7. ");
			
			//create template
			SimpleMatrix C_vInverted = C_vMatrix.inverse(InversionType.INVERT_QR);
			
			System.out.println("8. ");
			
			SimpleMatrix wMatrix = SimpleOperators.multiplyMatrixProd(s_vMatrix.transposed(), C_vInverted);
			
			SimpleMatrix resultMatrix = SimpleOperators.multiplyMatrixProd(wMatrix, vMatrix);
			
			System.out.println(resultMatrix.getElement(0, 0));
		}
		catch(Exception e)
		{
			System.out.println(e.getMessage());
		}
		
		
		//new ImageJ();
	}
}


