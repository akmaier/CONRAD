package ProjectIrisKellermann;

import java.util.Random;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.filtering.PoissonNoiseFilteringTool;
import edu.stanford.rsl.conrad.filtering.LogPoissonNoiseFilteringTool;
import edu.stanford.rsl.tutorial.parallel.ParallelProjector2D;
import ij.ImageJ;

public class Model extends Grid2D
{
	
	public Model(int width, int height)
	{
		super(width, height);
		
		DrawCircle(this, width/2, height/2, width/3, 0.2);
		
	}
	
	private void DrawCircle(Grid2D grid, int x, int y, int radius, double intensity)
	{
		for(int i = 0; i < this.getWidth(); ++i)
		{
			for(int j = 0; j < this.getHeight(); j++)
			{ 
				if((i-x)*(i-x) + (j-y)*(j-y) < (radius * radius))
				{
					grid.putPixelValue(i,j,intensity);
				}
			}
		}
	}
	
	public Grid2D ModelVariation()
	{
		Grid2D newModel = new Grid2D(this.getWidth(), this.getHeight());
		
		Random rand = new Random();
		
		int random = rand.nextInt(10);
			
		DrawCircle(newModel, newModel.getWidth()/2 + random, newModel.getHeight()/2 + random, newModel.getWidth()/3, 1.0);
		
		return newModel;
	}

	public Grid2D EmptyImage()
	{
		Grid2D newImage = new Grid2D(this.getWidth(), this.getHeight());
		
		for(int i = 0; i < this.getHeight(); ++i)
		{
			for(int j = 0; j < this.getWidth(); ++j)
			{
				newImage.putPixelValue(j, i, 0.0);
			}
		}
		
		return newImage;
	}
	
	public Grid2D PoissonNoise(Grid2D sinogram)
	{		
		Grid2D filteredModel = null;
		
		LogPoissonNoiseFilteringTool poissonFilt = new LogPoissonNoiseFilteringTool();
		
		try {
			filteredModel = poissonFilt.applyToolToImage(sinogram);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return filteredModel;
	}
	
	public Grid2D CreateSinogram(Grid2D model)
	{
		Grid2D sinogram = new Grid2D(model.getWidth(), 180);

		for(int s = -model.getWidth()/2; s < model.getWidth()/2; ++s)
		{
			for(int theta = 0; theta < 180; ++theta)
			{
				double degree = theta * Math.PI/180;
				float sum = 0;

				if(theta < 45 || theta > 135)
				{
					for(int y = 0; y < model.getHeight(); ++y)
					{
						double x = (s - (y - model.getHeight()/2) * Math.sin(degree))/Math.cos(degree) + model.getWidth()/2;

						float interpolate = InterpolationOperators.interpolateLinear(model, x, y);

						sum += interpolate;
					}					
				}

				else
				{
					for(int x = 0; x < model.getWidth(); ++x)
					{
						double y = (s - (x - model.getWidth()/2) * Math.cos(degree))/ Math.sin(degree) + model.getHeight()/2;
						
						float interpolate = InterpolationOperators.interpolateLinear(model, x, y);
						
						sum += interpolate;
					}

				}
				sinogram.putPixelValue(s + model.getWidth()/2, theta, sum);
			}
		}

		return sinogram;

	}

	public Grid2D[] CreateTestModels(int number)
	{
		Grid2D[] resultArray = new Grid2D[number];
		
		int i = 0;
		
		for (; i < number/4; ++i)
		{
			resultArray[i] = this.ModelVariation();
		}
		
		for (; i < number/2; ++i)
		{
			Grid2D poissonSinogram = this.PoissonNoise(this.CreateSinogram(this.ModelVariation()));
			Grid2D filteredSinogram = Backproject.Filter(poissonSinogram);
			
			resultArray[i] = Backproject.Backprojection(filteredSinogram);
		}
		
		for (; i < number * 3/4; ++i)
		{
			Grid2D sinogram = this.CreateSinogram(this.ModelVariation());
			
			resultArray[i] = Backproject.Backprojection(sinogram);
		}
		
		for (; i < number; ++i)
		{
			Grid2D sinogram = this.CreateSinogram(this.ModelVariation());
			NumericPointwiseOperators.divideBy(sinogram, 40);
			Grid2D poissonSinogram = this.PoissonNoise(sinogram);
			NumericPointwiseOperators.multiplyBy(poissonSinogram, 40);
			Grid2D filteredSinogram = Backproject.Filter(poissonSinogram);
			
			resultArray[i] = Backproject.Backprojection(filteredSinogram);			
		}
		
		return resultArray;
	}

	public Grid2D[] CreateEmptyImages(int number)
	{
		Grid2D[] resultArray = new Grid2D[number];
		
		int i = 0;
		
		for(; i < number * 1/4; ++i)
		{
			resultArray[i] = this.EmptyImage();
		}
		
		for(; i < number/2; ++i)
		{
			Grid2D poissonSinogram = this.PoissonNoise(this.CreateSinogram(this.EmptyImage()));
			Grid2D filteredSinogram = Backproject.Filter(poissonSinogram);
			
			resultArray[i] = Backproject.Backprojection(filteredSinogram);
		}
		
		for (; i < number * 3/4; ++i)
		{
			Grid2D sinogram = this.CreateSinogram(this.EmptyImage());
			
			resultArray[i] = Backproject.Backprojection(sinogram);
		}
		
		for (; i < number; ++i)
		{
			Grid2D sinogram = this.CreateSinogram(this.EmptyImage());
			NumericPointwiseOperators.divideBy(sinogram, 40);
			Grid2D poissonSinogram = this.PoissonNoise(sinogram);
			NumericPointwiseOperators.multiplyBy(poissonSinogram, 40);
			Grid2D filteredSinogram = Backproject.Filter(poissonSinogram);
			
			resultArray[i] = Backproject.Backprojection(filteredSinogram);			
		}
		
		return resultArray;
	}
}
