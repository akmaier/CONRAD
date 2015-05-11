package ProjectIrisKellermann;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.filtering.LogPoissonNoiseFilteringTool;
import ProjectIrisKellermann.Model;

public class Main {

	public static void main(String args[])
	{
		Model model = new Model(200,200);
		
		model.show();
		
		Grid2D[] testModels = model.CreateTestModels(10);
		
		for(int i = 0; i < testModels.length; ++i)
		{
			testModels[i].show();
		}
		
		/*Grid2D newModel = model.ModelVariation();
		
		newModel.show();
		
		Grid2D sinogram = model.CreateSinogram(newModel);
		
		NumericPointwiseOperators.divideBy(sinogram, 40);
		
		sinogram.show();
		
		Grid2D PoissonSinogram = model.PoissonNoise(sinogram);
		
		sinogram.show();
		
		NumericPointwiseOperators.multiplyBy(PoissonSinogram, 40);
		
		Grid2D filteredSinogram = Backproject.Filter(PoissonSinogram);
		
		Grid2D backprojection = Backproject.Backprojection(filteredSinogram);
		
		backprojection.show(); */
		
		//new ImageJ();
	}
}


