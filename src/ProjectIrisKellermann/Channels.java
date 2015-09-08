package ProjectIrisKellermann;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;

public class Channels 
{
	public static double gaussValue = 70.0;
	
	public static double GetFunctionValue(double r, int degree)
	{
		double result = (Math.sqrt(2) / gaussValue) * Math.exp( (- Math.PI * r * r ) / (gaussValue * gaussValue) ) * Laguerre( ( (2 * Math.PI * r * r) / ( gaussValue * gaussValue) ), degree );
		
		return result;
	}
	
	public static double Laguerre(double value, int degree)
	{
		double result = 0.0;
		
		for(int i = 0; i < degree + 1; ++i)
		{
			result += Math.pow(-1, i) * ( Factorial(degree) / (Factorial(i) * Factorial(degree - i) ) ) * (Math.pow(value, i) / Factorial(i) );
		}
		
		return result;
	}
	
	public static double Factorial(int value)
	{
		if(value <= 1)
		{
			return 1;
		}
		
		return value * Factorial(value - 1);
	}
	
	public static Grid2D CreateLGChannelImage(int width, int height, int degree)
	{
		Grid2D resultImage = new Grid2D(width, height);
		
		for(int x = 0; x < width; ++x)
		{
			int xValue = - width / 2 + x;
					
			for(int y = 0; y < height; ++y)
			{
				int yValue = - height / 2 + y;
				
				double r = Math.hypot(xValue, yValue);
				
				resultImage.putPixelValue(x, y, GetFunctionValue(r, degree));
			}
		}
		
		return resultImage;
	}
	
	public static double MultiplyImages(Grid2D image1, Grid2D image2) throws Exception
	{
		int width1 = image1.getWidth();
		int width2 = image2.getWidth();
		int height1 = image1.getHeight();
		int height2 = image2.getHeight();
		if(width1 != width2 || height1 != height2)
		{
			throw new Exception("Image sizes must be equal!");
		}
		
		double result = 0.0;
		
		for(int i = 0; i < height1; ++i)
		{
			for(int j = 0; j < width1; ++j)
			{
				result += image1.getPixelValue(j, i) * image2.getPixelValue(j, i);
			}
		}
		
		return result;
	}

}
