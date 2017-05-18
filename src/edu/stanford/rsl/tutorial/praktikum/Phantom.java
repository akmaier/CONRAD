package edu.stanford.rsl.tutorial.praktikum;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import ij.ImageJ;


public class Phantom extends Grid2D{

	public Phantom(int width, int height, double[] spacing) {
		super(width, height);
		this.setSpacing(spacing);
		
		double[] origin = {-(width-1) * (spacing[0]/2), -(height-1) * (spacing[1]/2)};
		
		this.setOrigin(origin);
		
		
		//create square with intensity 0.2
		int edgeLength = width/8;
		for (int row =  edgeLength ; row < width/2 +  edgeLength ; row++) {
			for (int col =  edgeLength ; col < height/2 +  edgeLength ; col++) {
				this.setAtIndex(row, col, (0.2f));
			}
		}
		
		//create circle with intensity 0.5
		int xCenter = width/2 + width/8;
		int yCenter = height/2 + height/4;
		int radius = width/10;
		for (int row = xCenter - radius; row < xCenter + radius; row++){
			for (int col = yCenter - radius; col < yCenter + radius; col++){
				if (((row-xCenter)*(row-xCenter)) + ((col-yCenter)*(col-yCenter)) <= radius * radius){
					this.setAtIndex(row, col, (.5f));
				}
			}
		}
		
		//create triangle with intensity 0.7
		int count = 0;
		for (int row = 2 * height/8; row < height/2; row++){
			for (int col = 4 * width/8; col < width; col++){
				if(col >= (6 * width/8) - count && col <= (6 * width/8) + count){
					this.setAtIndex(col, row, (.7f));
				}
			}
			count++;
		}
	}
	
	public static void main(String[] args){
		ImageJ ui = new ImageJ();
		double[] spacingA = {0.5, 0.5};
		Phantom a = new Phantom(512, 512, spacingA);
		
		double[] spacingB = {0.8, 0.8};
		Phantom b = new Phantom(512, 512, spacingB);
		
		a.show("Phantom A");
		b.show("Phantom B");
		
		NumericGrid c = NumericPointwiseOperators.subtractedBy(a, b);
		c.show("Phantom A - B");
		
		double[] origin = a.getOrigin();
		
		System.out.println("Origin of phantom A lies at: " +origin[0]+ ", " +origin[1]);
		
		float aMin = NumericPointwiseOperators.min(a);
		float aMax = NumericPointwiseOperators.max(a);
		float aMean = NumericPointwiseOperators.mean(a);
		
		System.out.println("Minimum of phantom A: " +aMin);
		System.out.println("Maximum of phantom A: " +aMax);
		System.out.println("Mean of phantom A: " +aMean);
		
		float inter = InterpolationOperators.interpolateLinear(a, 256, 256);
		System.out.println("Linear interpolation: " +inter);
		
		
	}

}
