package edu.stanford.rsl.tutorial.modelObserver;

import java.util.Random;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.filtering.LogPoissonNoiseFilteringTool;
import edu.stanford.rsl.tutorial.filters.RamLakKernel;
import edu.stanford.rsl.tutorial.parallel.ParallelBackprojector2D;
import edu.stanford.rsl.tutorial.parallel.ParallelProjector2D;

/**
 * @author Iris Kellermann
 */

public class Model extends Grid2D
{
	ParallelProjector2D projector;
	ParallelBackprojector2D backProjector;
	RamLakKernel ramLak;
	
	public Model(int width, int height)
	{
		super(width, height);
		
		this.setSpacing(1.0, 1.0);
		this.setOrigin(-width/2.d, -height/2.d);
		
		DrawCircle(this, width/2, height/2, width/3, 0.2);
		
		projector = new ParallelProjector2D(Math.PI, Math.PI/180.0, 400, 1);
		backProjector = new ParallelBackprojector2D(width, height, 1, 1);
		ramLak = new RamLakKernel(400, 1);
	}
	
	/**
	 * Set the type to be used for variation.
	 */
	public static enum VariationType
	{
		/** Shifts the object in the image/creates a random salt image */
		SimpleVariation,
		/** Projects and backprojects the image without using a filter */
		ProjectionWOFilter,
		/** Projects and backprojects the image and adds Poisson noise */
		ProjectionPoisson,
		/** Projects and backprojects the image, adds Poisson noise but without intensity shifting */
		ProjectionWrongPoisson,
		/** Shifts the object in the image and adds noise/creates a random salt image */
		SimpleVariationNoise
	}
		
	/**
	 * Draws a circle into the given Grid2D with the given radius and the given intensity at the position x,y.  
	 * @param grid The Grid2D to draw the circle in
	 * @param x The x value of the position.
	 * @param y The y value of the position.
	 * @param radius The radius of the circle.
	 * @param intensity The intensity for the circle.
	 */
	private void DrawCircle(Grid2D grid, int x, int y, double radius, double intensity)
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
	
	/**
	 * Creates a Grid2D of a variation of the model.
	 * @return  The image of the model variation.
	 */
	public Grid2D ModelVariation()
	{
		int width = this.getWidth();
		int height = this.getHeight();
		Grid2D newModel = new Grid2D(width, height);
		
		newModel.setSpacing(1.0, 1.0);
		newModel.setOrigin(-width/2.d, -height/2.d);
		
		Random rand = new Random();
		
		int random = rand.nextInt(10);
			
		DrawCircle(newModel, newModel.getWidth()/2 + random, newModel.getHeight()/2 + random, newModel.getWidth()/(rand.nextDouble() + 2.5), 1.0);
		
		return newModel;
	}

	/**
	 * Creates a new image with Salt noise of the same size as the model.
	 * @return  The salt image.
	 */
	public Grid2D CreateSaltImage()
	{
		int width = this.getWidth();
		int height = this.getHeight();
		
		Grid2D resultImage = new Grid2D(width, height);
		
		Random rand = new Random();
		
		for(int i = 0; i < width * height / 10; ++i)
		{
			resultImage.putPixelValue(rand.nextInt(width), rand.nextInt(height), rand.nextDouble() - 0.5);
		}
		
		resultImage.setSpacing(1.0, 1.0);
		resultImage.setOrigin(-width/2.d, -height/2.d);
		
		return resultImage;
	}
	
	/**
	 * Creates a new empty image of the same size as the model.  
	 * @return  The empty image.
	 */
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
	
	/**
	 * Applies poisson noise to the input sinogram.  
	 * @param sinogram The sinogram to apply the poisson noise.
	 * @return  The resulting sinogram.
	 */
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

	/**
	 * Creates a number of different test images with the object present.  
	 * @param number The number of images to create.
	 * @param type The type of the Variation of the model.
	 * @return  The array of test images.
	 */
	public Grid2D[] CreateTestModels(int number, VariationType type)
	{
		Grid2D[] resultArray = new Grid2D[number];
		
		switch(type)
		{
		case SimpleVariation:
			for (int i = 0; i < number; ++i)
			{
				resultArray[i] = this.ModelVariation();
			}
			break;
		case ProjectionWOFilter:
			for (int i = 0; i < number; ++i)
			{
				Grid2D modelVariation = this.ModelVariation();
				Grid2D sinogram = projector.projectPixelDriven(modelVariation);
				resultArray[i] = backProjector.backprojectPixelDriven(sinogram);
			}
			break;
		case ProjectionPoisson:
			for (int i = 0; i < number; ++i)
			{
				Grid2D sinogram = projector.projectPixelDriven(this.ModelVariation());
				NumericPointwiseOperators.divideBy(sinogram, 40);
				Grid2D poissonSinogram = this.PoissonNoise(sinogram);
				NumericPointwiseOperators.multiplyBy(poissonSinogram, 40);

				Grid2D filteredSinogram = new Grid2D(poissonSinogram);
				
				for (int j = 0; j < filteredSinogram.getSize()[1]; j++)
				{
					ramLak.applyToGrid(filteredSinogram.getSubGrid(j));
				}
				resultArray[i] =  backProjector.backprojectPixelDriven(filteredSinogram);			
			}
			break;
		case ProjectionWrongPoisson:
			for (int i = 0; i < number; ++i)
			{
				Grid2D poissonSinogram = this.PoissonNoise(projector.projectPixelDriven(this.ModelVariation()));
				
				Grid2D filteredSinogram = new Grid2D(poissonSinogram);
				
				for (int j = 0; j < filteredSinogram.getSize()[1]; j++)
				{
					ramLak.applyToGrid(filteredSinogram.getSubGrid(j));
				}
				
				resultArray[i] = backProjector.backprojectPixelDriven(filteredSinogram);
			}
			break;
		case SimpleVariationNoise:
			for (int i = 0; i < number; ++i)
			{
				Grid2D result = this.ModelVariation();
				NumericPointwiseOperators.addBy(result, this.CreateSaltImage());
				resultArray[i] = result;
			}
			break;
		default:
			break;
		}
		return resultArray;
	}

	/**
	 * Creates a number of different empty salt images.  
	 * @param number The number of images to create.
	 * @param type The type of the variation.
	 * @return  The array of empty images.
	 */
	public Grid2D[] CreateEmptyImages(int number, VariationType type)
	{
		Grid2D[] resultArray = new Grid2D[number];
		
		switch(type)
		{
		case SimpleVariation:
			for (int i = 0; i < number; ++i)
			{
				resultArray[i] = this.CreateSaltImage();
			}
			
			break;
		case ProjectionWOFilter:
			for (int i = 0; i < number; ++i)
			{
				Grid2D sinogram = projector.projectPixelDriven(this.CreateSaltImage());
				
				resultArray[i] = backProjector.backprojectPixelDriven(sinogram);
			}
			
			break;
		case ProjectionPoisson:
			for (int i = 0; i < number; ++i)
			{
				Grid2D sinogram = projector.projectPixelDriven(this.CreateSaltImage());
				NumericPointwiseOperators.divideBy(sinogram, 40);
				Grid2D poissonSinogram = this.PoissonNoise(sinogram);
				NumericPointwiseOperators.multiplyBy(poissonSinogram, 40);

				Grid2D filteredSinogram = new Grid2D(poissonSinogram);
				
				for (int j = 0; j < filteredSinogram.getSize()[1]; j++)
				{
					ramLak.applyToGrid(filteredSinogram.getSubGrid(j));
				}
				
				resultArray[i] = backProjector.backprojectPixelDriven(filteredSinogram);			
			}
			
			break;
		case ProjectionWrongPoisson:
			for (int i = 0; i < number; ++i)
			{
				Grid2D poissonSinogram = this.PoissonNoise(projector.projectPixelDriven(this.CreateSaltImage()));

				Grid2D filteredSinogram = new Grid2D(poissonSinogram);
				
				for (int j = 0; j < filteredSinogram.getSize()[1]; j++)
				{
					ramLak.applyToGrid(filteredSinogram.getSubGrid(j));
				}
				
				resultArray[i] = backProjector.backprojectPixelDriven(filteredSinogram);
			}
			
			break;
		case SimpleVariationNoise:
			return this.CreateEmptyImages(number, VariationType.SimpleVariation);
		}
		
		return resultArray;
	}
}

/*
 * Copyright (C) 2010-2014 - Iris Kellermann 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/