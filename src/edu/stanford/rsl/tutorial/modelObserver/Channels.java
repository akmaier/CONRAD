package edu.stanford.rsl.tutorial.modelObserver;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * @author Iris Kellermann
 */

public class Channels 
{	
	/**
	 * Calculates the result of the Laguerre-Gauß function for the given input values.  
	 * @param r The radial value.
	 * @param degree The degree of the LG function.
	 * @return  The result value.
	 */
	public static double GetLGFunctionValue(double r, int degree, double gaussValue)
	{
		double result = (Math.sqrt(2) / gaussValue) * Math.exp( (- Math.PI * r * r ) / (gaussValue * gaussValue) ) * Laguerre( ( (2 * Math.PI * r * r) / ( gaussValue * gaussValue) ), degree );
		
		return result;
	}
	
	/**
	 * Calculates the result of the Laguerre function for the given input values.  
	 * @param value The input value.
	 * @param degree The degree of the Laguerre function.
	 * @return  The result value.
	 */
	public static double Laguerre(double value, int degree)
	{
		double result = 0.0;
		
		for(int i = 0; i < degree + 1; ++i)
		{
			result += Math.pow(-1, i) * ( Factorial(degree) / (Factorial(i) * Factorial(degree - i) ) ) * (Math.pow(value, i) / Factorial(i) );
		}
		
		return result;
	}
	
	/**
	 * Calculates the factorial of the input value.  
	 * @param value The input value.
	 * @return  The result value.
	 */
	public static double Factorial(int value)
	{
		if(value <= 1)
		{
			return 1;
		}
		
		return value * Factorial(value - 1);
	}
	
	/**
	 * Creates a Laguerre-Gauß channel image.  
	 * @param width The width of the image.
	 * @param height The height of the image.
	 * @param degree The degree of the LG function.
	 * @return  The result image.
	 */
	public static Grid2D CreateLGChannelImage(int width, int height, int degree, double gaussValue)
	{
		Grid2D resultImage = new Grid2D(width, height);
		
		for(int x = 0; x < width; ++x)
		{
			int xValue = - width / 2 + x;
					
			for(int y = 0; y < height; ++y)
			{
				int yValue = - height / 2 + y;
				
				double r = Math.hypot(xValue, yValue);
				
				resultImage.putPixelValue(x, y, GetLGFunctionValue(r, degree, gaussValue));
			}
		}
		
		return resultImage;
	}

	/**
	 * Creates a matrix of concatenated channel image vectors.  
	 * @param channelCount The number of channels.
	 * @param imageSize The size of the images.
	 * @return  The SimpleMatrix containing the channel image vectors.
	 */
	public static SimpleMatrix CreateChannelMatrix(int channelCount, int imageSize, double gaussValue)
	{		
		SimpleVector[] channelColumns = new SimpleVector[channelCount];
		for(int i = 0; i < channelCount; ++i)
		{
			channelColumns[i] = ImageHelper.ConvertSimpleMatrixToVector(ImageHelper.ConvertGrid2DToSimpleMatrix(Channels.CreateLGChannelImage(imageSize, imageSize, i, gaussValue)));
		}
		
		//TODO - wrong operator in conrad/numerics/SimpleOperators?
		//SimpleMatrix channelMatrix = SimpleOperators.concatenateHorizontally(channelColumns);
		
		SimpleMatrix channelMatrix = ImageHelper.concatenateHorizontally(channelColumns);
		
		return channelMatrix;
		
	}
}

/*
 * Copyright (C) 2010-2014 - Iris Kellermann 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/