package edu.stanford.rsl.tutorial.modelObserver;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.InversionType;

/**
 * @author Iris Kellermann
 */

public class Observer {

	/**
	 * Calculates the difference of the mean images of the given image arrays.  
	 * @param objectImages The array of images with present object.
	 * @param emptyImages The array of empty images.
	 * @return  The mean difference image.
	 */
	public static SimpleMatrix GetMeanDifferenceImage(SimpleMatrix[] objectImages, SimpleMatrix[] emptyImages)
	{
		try
		{
			SimpleMatrix meanObjectImage = ImageHelper.GetMeanImage(objectImages);
		
			SimpleMatrix meanEmptyImage = ImageHelper.GetMeanImage(emptyImages);
		
			meanObjectImage.subtract(meanEmptyImage);
			
			return meanObjectImage;
		}
		catch(Exception e)
		{
			System.out.println(e.getMessage());
		}
		
		return null;
	}

	/**
	 * Calculates the mean covariance matrix of channelized image arrays.  
	 * @param objectImages The array of images with present object.
	 * @param emptyImages The array of empty images.
	 * @param channelMatrix The matrix containing the channel image vectors.
	 * @return  The result matrix.
	 */
	public static SimpleMatrix GetCovarianceMatrix(SimpleMatrix[] objectImages, SimpleMatrix[] emptyImages, SimpleMatrix channelMatrix)
	{
		SimpleMatrix meanObjectImage = ImageHelper.GetMeanImage(objectImages);
		SimpleMatrix meanEmptyImage = ImageHelper.GetMeanImage(emptyImages);
		
		SimpleMatrix[] MeanCovariance = new SimpleMatrix[2]; 
		MeanCovariance[0] = new SimpleMatrix	(channelMatrix.getCols(), channelMatrix.getCols());
		MeanCovariance[1] = new SimpleMatrix	(channelMatrix.getCols(), channelMatrix.getCols());
				
		for(int i = 0; i < objectImages.length; ++i)
		{
			SimpleVector differenceImage = ImageHelper.ConvertSimpleMatrixToVector(SimpleOperators.subtract(objectImages[i], meanObjectImage));
			SimpleVector channelizedImage = SimpleOperators.multiply(channelMatrix.transposed(), differenceImage);
		
			MeanCovariance[0].add(SimpleOperators.multiplyOuterProd(channelizedImage, channelizedImage));
		}			
		
		MeanCovariance[0].divideBy(objectImages.length);
		
		for(int i = 0; i < emptyImages.length; ++i)
		{
			SimpleVector differenceImage = ImageHelper.ConvertSimpleMatrixToVector(SimpleOperators.subtract(emptyImages[i], meanEmptyImage));
			SimpleVector channelizedImage = SimpleOperators.multiply(channelMatrix.transposed(), differenceImage);
		
			MeanCovariance[1].add(SimpleOperators.multiplyOuterProd(channelizedImage, channelizedImage));
		}
		
		MeanCovariance[1].divideBy(emptyImages.length);
				
		//mean of covariance matrixes
		return ImageHelper.GetMeanImage(MeanCovariance);
	}

	/**
	 * Creates the template for the observer.  
	 * @param testModels The array of images with present object.
	 * @param emptyImages The array of empty images.
	 * @param channelMatrix The matrix containing the channel image vectors.
	 * @return  The result value.
	 */
	public static SimpleVector CreateTemplate(Grid2D[] testModels, Grid2D[] emptyImages, SimpleMatrix channelMatrix)
	{
		//convert Grid2D images to SimpleMatrix
		SimpleMatrix[] objectImagesMatrix = new SimpleMatrix[testModels.length];
		SimpleMatrix[] emptyImagesMatrix = new SimpleMatrix[emptyImages.length];
		
		for(int i = 0; i < testModels.length; ++i)
		{
			objectImagesMatrix[i] = ImageHelper.ConvertGrid2DToSimpleMatrix(testModels[i]);
		}
		for(int i = 0; i < emptyImages.length; ++i)
		{
			emptyImagesMatrix[i] = ImageHelper.ConvertGrid2DToSimpleMatrix(emptyImages[i]);
		}
		
		
		SimpleMatrix meanDifferenceImage = Observer.GetMeanDifferenceImage(objectImagesMatrix, emptyImagesMatrix);
					
		SimpleMatrix covarianceMatrix = Observer.GetCovarianceMatrix(objectImagesMatrix, emptyImagesMatrix, channelMatrix);
		
		//convert meanDifferenceImage to columns
		SimpleVector meanDifferenceVector = ImageHelper.ConvertSimpleMatrixToVector(meanDifferenceImage);
		
		//calculate channelized mean-difference image s_v
		SimpleVector s_v = SimpleOperators.multiply(channelMatrix.transposed(), meanDifferenceVector);
			
		//create template
		SimpleMatrix C_vInverted = covarianceMatrix.inverse(InversionType.INVERT_QR);
		
		SimpleVector templateVector = SimpleOperators.multiply(s_v, C_vInverted);
		
		return templateVector;
	}
	
	/**
	 * Calculates the result of the observer.  
	 * @param model The test model.  
	 * @param template The template of the observer.
	 * @param channelMatrix The matrix containing the channel image vectors.
	 * @return  The result value.
	 */
	public static double GetResultValue(Grid2D model, SimpleVector template, SimpleMatrix channelMatrix)
	{
		//convert Grid2D images to SimpleMatrix
		SimpleVector modelVector = ImageHelper.ConvertSimpleMatrixToVector(ImageHelper.ConvertGrid2DToSimpleMatrix(model));
		
		//the channelized model image
		SimpleVector vVector = SimpleOperators.multiply(channelMatrix.transposed(), modelVector);
		
		double resultValue = SimpleOperators.multiplyInnerProd(template, vVector);
				
		return resultValue;
	}
}

/*
 * Copyright (C) 2010-2014 - Iris Kellermann 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/