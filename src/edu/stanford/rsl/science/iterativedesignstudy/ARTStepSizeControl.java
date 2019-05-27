package edu.stanford.rsl.science.iterativedesignstudy;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;


public class ARTStepSizeControl extends ART{
	Grid2D diff2D;
	NumericGrid gradientSinogram;
	
	public ARTStepSizeControl(Projector projector, Backprojector backprojector, NumericGrid diff2D, NumericGrid gradientSinogram) {
		super(projector, backprojector);

		this.diff2D = (Grid2D)diff2D;
		this.gradientSinogram = gradientSinogram;
	}
	
	public void setStepSize(NumericGrid recon){ //TODO:Nicht vollstaendig!
		stepSize = (-1.0) * (NumericPointwiseOperators.sum(NumericPointwiseOperators.multipliedBy(diff2D, gradientSinogram)))
				/ (NumericPointwiseOperators.sum(NumericPointwiseOperators.multipliedBy(gradientSinogram, gradientSinogram)));		
	}
	
	public void imageUpdate(NumericGrid imageUpdate) {
		gradientSinogram = projector.project(imageUpdate, gradientSinogram);
	}

	public void setDiff(NumericGrid diff, int index) {
		for (int j = 0; j < diff.getSize()[0]; j++) {			
			diff2D.setAtIndex(j, index, ((Grid1D)diff).getAtIndex(j));
		}
	}
	
	
}
