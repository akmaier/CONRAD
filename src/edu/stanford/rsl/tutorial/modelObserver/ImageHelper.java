package edu.stanford.rsl.tutorial.modelObserver;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * @author Iris Kellermann
 */

public class ImageHelper {
	/**
	 * Calculates the mean image of the given image array.  
	 * @param images The input images.
	 * @return  The result image.
	 */
	public static SimpleMatrix GetMeanImage(SimpleMatrix[] images)
	{
		int rows = images[0].getRows();
		int columns = images[0].getCols();
		
		SimpleMatrix resultMatrix = new SimpleMatrix(rows, columns);
				
		resultMatrix.add(images);
		resultMatrix.divideBy(images.length);
		
		return resultMatrix;
	}

	/**
	 * Converts a Grid2D to a SimpleMatrix.  
	 * @param grid The Gri2D to convert.
	 * @return  The result matrix.
	 */
	public static SimpleMatrix ConvertGrid2DToSimpleMatrix(Grid2D grid)
	{
		double[][] gridDouble = new double[grid.getHeight()][grid.getWidth()];
		
		for(int i = 0; i < grid.getWidth(); ++i)
		{
			for(int j = 0; j < grid.getHeight(); ++j)
			{
				gridDouble[j][i] = grid.getPixelValue(i, j);
			}
		}
		
		SimpleMatrix gridMatrix = new SimpleMatrix(gridDouble);
		
		return gridMatrix;
	}

	/**
	 * Converts a SimpleMatrix to a SimpleVector by concatenating the columns vertically.  
	 * @param matrix The matrix to convert.
	 * @return  The result vector.
	 */
	public static SimpleVector ConvertSimpleMatrixToVector(SimpleMatrix matrix)
	{
		SimpleVector[] columns = new SimpleVector[matrix.getCols()];
		
		for(int i = 0; i < matrix.getCols(); ++i)
		{
			columns[i] = matrix.getCol(i);
		}
		
		return SimpleOperators.concatenateVertically(columns);
	}

	/**
	 * Creates a new matrix which is composed of all input column vectors, stacked next to each other.  
	 * @param columns  The vectors to stack.
	 * @return  The horizontally concatenated matrix.
	 */
	public static SimpleMatrix concatenateHorizontally(SimpleVector... columns) {
		final int cols = columns.length;
		assert cols >= 1 : new IllegalArgumentException("Supply at least one vector to concatenate!");
		final int rows = columns[0].getLen();
		assert rows >= 1 : new IllegalArgumentException("Vectors have to contain at least one element each!");
		SimpleMatrix result = new SimpleMatrix(rows, cols);
		for (int c = 0; c < cols; ++c)
			result.setColValue(c, columns[c]);
		return result;
	}
}

/*
 * Copyright (C) 2010-2014 - Iris Kellermann 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/