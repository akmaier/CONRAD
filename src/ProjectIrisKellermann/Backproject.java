package ProjectIrisKellermann;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid1DComplex;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;

public class Backproject{

	public static Grid2D Filter(Grid2D sinogram)
	{
		Grid1DComplex[] gridrows = new Grid1DComplex[sinogram.getHeight()];

		Grid1D[] realGridrows = new Grid1D[sinogram.getHeight()];

		for(int i = 0; i < sinogram.getHeight(); ++i)
		{
			//create 1Dcomplex and compute the forward fourier transformation
			gridrows[i] = new Grid1DComplex(sinogram.getSubGrid(i));

			Grid1DComplex currentGrid = gridrows[i];

			currentGrid.transformForward();		

			float counter = 0;

			float stepsize = 1.0f/(2*(currentGrid.getSize()[0]));

			for(int j = 0; j < currentGrid.getSize()[0]; ++j)
			{
				currentGrid.setRealAtIndex(j, currentGrid.getRealAtIndex(j) * counter);
				currentGrid.setImagAtIndex(j, currentGrid.getImagAtIndex(j) * counter);

				if(j >= currentGrid.getSize()[0]/2)
				{
					counter -= stepsize;
				}
				else
				{
					counter += stepsize;
				}
			}

			currentGrid.transformInverse();
			realGridrows[i] = currentGrid.getRealSubGrid(0, currentGrid.getSize()[0]);

		}

		Grid2D resultGrid = new Grid2D(sinogram.getWidth(), sinogram.getHeight());

		for(int i = 0; i < sinogram.getWidth(); ++i)
		{
			for(int j = 0; j < sinogram.getHeight(); ++j)
			{
				resultGrid.setAtIndex(i, j, realGridrows[j].getAtIndex(i));
			}
		}



		return resultGrid;
	}


	public static Grid2D FilterRamLak(Grid2D sinogram)
	{
		Grid1DComplex[] gridrows = new Grid1DComplex[sinogram.getHeight()];

		Grid1D[] realGridrows = new Grid1D[sinogram.getHeight()];

		Grid1DComplex RamLak = new Grid1DComplex(sinogram.getWidth());

		int middle = RamLak.getSize()[0]/2;
		RamLak.setRealAtIndex(middle, 1.0f/4);

		for(int i = middle + 1; i < RamLak.getSize()[0]; i += 2)
		{
			RamLak.setRealAtIndex(i, (float)(-(1/((i-middle)*(i-middle)*(Math.PI)*(Math.PI) ) ) ) );
		}
		for(int i = middle -1; i >= 0; i -= 2)
		{
			RamLak.setRealAtIndex(i, (float)(-(1/((i-middle)*(i-middle)*(Math.PI)*(Math.PI) ) ) ) );
		}

		RamLak.transformForward();


		for(int i = 0; i < sinogram.getHeight(); ++i)
		{
			//create 1Dcomplex and compute the forward fourier transformation
			gridrows[i] = new Grid1DComplex(sinogram.getSubGrid(i));

			Grid1DComplex currentGrid = gridrows[i];

			currentGrid.transformForward();	

			//multiply with ramlak filter
			for(int j = 0; j < currentGrid.getSize()[0]; ++j)
			{
				currentGrid.multiplyAtIndex(j, RamLak.getAtIndex(j));
			}

			currentGrid.transformInverse();
			realGridrows[i] = currentGrid.getRealSubGrid(0, currentGrid.getSize()[0]);


		}

		Grid2D resultGrid = new Grid2D(sinogram.getWidth(), sinogram.getHeight());

		for(int i = 0; i < sinogram.getWidth(); ++i)
		{
			for(int j = 0; j < sinogram.getHeight(); ++j)
			{
				resultGrid.setAtIndex(i, j, realGridrows[j].getAtIndex(i));
			}
		}


		return resultGrid;

	}



	public static Grid2D Backprojection(Grid2D projection)
	{
		Grid2D resultImage = new Grid2D(projection.getWidth(), projection.getWidth());

		for(int x = 0; x < resultImage.getWidth(); ++x)
		{
			for(int y = 0; y < resultImage.getHeight(); ++y)
			{				
				for(int theta = 0; theta < 180; ++theta)
				{
					double rad = theta * Math.PI/180;

					double s = (x - resultImage.getWidth()/2) * Math.cos(rad) + (y-resultImage.getHeight()/2) * Math.sin(rad);

					float interpolated = InterpolationOperators.interpolateLinear(projection, s + resultImage.getHeight()/2, theta);

					resultImage.addAtIndex(x, y, interpolated);

				}

			}
		}

		return resultImage;
	}

}
