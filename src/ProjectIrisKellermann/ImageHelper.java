package ProjectIrisKellermann;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;

public class ImageHelper {

	public static Grid2D ConvertImageToArray(Grid2D image)
	{
		Grid2D array = new Grid2D(1, image.getWidth() * image.getHeight());
		
		for(int x = 0; x < image.getWidth(); ++x)
		{
			for(int y = 0; y < image.getHeight(); ++y)
			{
				array.putPixelValue(0, x * image.getHeight() + y, image.getPixelValue(x, y));
			}
		}
		
		return array;
	}

	public static Grid2D MatrixMultiplication(Grid2D matrix1, Grid2D matrix2) throws Exception
	{
		if(matrix1.getWidth() != matrix2.getHeight())
		{
			throw new Exception("Matrix dimensions must fit.");
		}
		
		Grid2D resultMatrix = new Grid2D(matrix1.getHeight(), matrix2.getWidth());
		
		for(int x = 0; x < resultMatrix.getWidth(); ++x)
		{
			for(int y = 0; y < resultMatrix.getHeight(); ++y)
			{
				double result = 0.0;
				
				for(int k = 0; k < matrix1.getWidth(); ++k)
				{
					result += matrix1.getPixelValue(k, y) * matrix2.getPixelValue(x, k);
				}
				
				resultMatrix.putPixelValue(x, y, result);
			}
		}
		
		return resultMatrix;
		
	}
	
	public static Grid2D Transpose(Grid2D image)
	{
		Grid2D transposedImage = new Grid2D(image.getHeight(), image.getWidth());
		
		for(int x = 0; x < transposedImage.getWidth(); ++x)
		{
			for(int y = 0; y < image.getHeight(); ++y)
			{
				transposedImage.putPixelValue(x, y, image.getPixelValue(y, x));
			}
		}
		
		return transposedImage;
	}

	public static Grid2D SubstractImages(Grid2D image1, Grid2D image2) throws Exception
	{
		if(image1.getWidth() != image2.getWidth() || image1.getHeight() != image2.getHeight())
		{
			throw new Exception("Image dimensions must be equal!");
		}
		
		Grid2D resultImage = new Grid2D(image1.getWidth(), image1.getHeight());
		
		for(int x = 0; x < image1.getWidth(); ++x)
		{
			for(int y = 0; y < image1.getHeight(); ++y)
			{
				resultImage.putPixelValue(x, y, image1.getPixelValue(x, y) - image2.getPixelValue(x, y));
			}
		}
		
		return resultImage;
	}

	public static Grid2D GetMeanImage(Grid2D[] images) throws Exception
	{
		int width = images[0].getWidth();
		int height = images[0].getHeight();
		
		for(int i = 1; i < images.length; ++i)
		{
			if(images[i].getWidth() != width || images[i].getHeight() != height)
			{
				throw new Exception("Image dimensions must be equal!");
			}
		}
		
		Grid2D resultImage = new Grid2D(width, height);

		for(int i = 0; i < images.length; ++i)
		{
			for(int x = 0; x < width; ++x)
			{
				for(int y = 0; y < height; ++y)
				{
					resultImage.putPixelValue(x, y, resultImage.getPixelValue(x, y) + images[i].getPixelValue(x, y) / images.length);
				}
			}
		}
		
		return resultImage;
	}
}
