package ProjectIrisKellermann;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;

public class Observer {

	public static Grid2D GetMeanDifferenceImage(Grid2D[] objectImages, Grid2D[] emptyImages)
	{
		try
		{
			Grid2D meanObjectImage = ImageHelper.GetMeanImage(objectImages);
		
			Grid2D meanEmptyImage = ImageHelper.GetMeanImage(emptyImages);
		
			Grid2D resultImage = ImageHelper.SubstractImages(meanObjectImage, meanEmptyImage);
		
			return resultImage;
		}
		catch(Exception e)
		{
			System.out.println(e.getMessage());
		}
		
		return null;
	}

	public static Grid2D GetCovarianceMatrix(Grid2D[] objectImages, Grid2D[] emptyImages)
	{
		Grid2D[] covarianceMatrixesObjects = new Grid2D[objectImages.length];
		Grid2D[] covarianceMatrixesEmpty = new Grid2D[emptyImages.length];
		
		Grid2D[] meanCovarianceImages = new Grid2D[2];
		
		try
		{
			// object images
			Grid2D meanObjectImage = ImageHelper.GetMeanImage(objectImages);
			
			for(int i = 0; i < objectImages.length; ++i)
			{
				Grid2D differenceImageArray = ImageHelper.ConvertImageToArray(ImageHelper.SubstractImages(objectImages[i], meanObjectImage));
				
				covarianceMatrixesObjects[i] = ImageHelper.MatrixMultiplication(differenceImageArray, ImageHelper.Transpose(differenceImageArray));
			}
			
			//get mean of the covariance matrixes
			meanCovarianceImages[0] = ImageHelper.GetMeanImage(covarianceMatrixesObjects);
		
			
			//empty images
			Grid2D meanEmptyImage = ImageHelper.GetMeanImage(emptyImages);
			
			for(int i = 0; i < emptyImages.length; ++i)
			{
				Grid2D differenceImageArray = ImageHelper.ConvertImageToArray(ImageHelper.SubstractImages(emptyImages[i], meanEmptyImage));
				
				covarianceMatrixesEmpty[i] = ImageHelper.MatrixMultiplication(differenceImageArray, ImageHelper.Transpose(differenceImageArray));
			}
			
			//get mean of the covariance matrixes
			meanCovarianceImages[1] = ImageHelper.GetMeanImage(covarianceMatrixesEmpty);
		
			//get mean of the mean images
			Grid2D resultCovarianceImage = ImageHelper.GetMeanImage(meanCovarianceImages);
			
			return resultCovarianceImage;
		}
		catch(Exception e)
		{
			System.out.println(e.getMessage());
		}
		
		return null;
	}

}
