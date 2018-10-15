package edu.stanford.rsl.tutorial.basics.videoTutorials;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;

public class videotutorialsGrids {
	
	public static void main(String [] args) {
		
		//create grid and set spacing and origin
		Grid2D grid1 = new Grid2D(300,300);
		grid1.setSpacing(1,1);
		grid1.setOrigin(0,0);
		
		//create a circle
		circle(grid1,75);
		
		//show the grid
		grid1.show("grid1");
		
		//create grid and set spacing and origin
		Grid2D grid2 = new Grid2D(300,300);
		grid2.setSpacing(1,1);
		grid2.setOrigin(-150,-150);
		
		//create a circle
		circle(grid2, 75);
				
		//show the grid
		grid2.show("grid2");
		
		//get and print the world coordinates of the indices (150,150)
		double[] idx = grid2.indexToPhysical(150, 150);
		System.out.println("World coordinates of indices (150,150): (" + idx[0] + ", " + idx[1] + ")");
		
		//create grid and set spacing and origin
		Grid2D grid3 = new Grid2D(300,300);
		grid3.setSpacing(2,1);
		grid3.setOrigin(-300,-150);
		
		//create a circle
		circle(grid3, 75);
		
		//show the grid
		grid3.show("grid3");
		
		//copy of a grid
		Grid2D copyOfGrid1 = new Grid2D(grid1);
		
		//show the grid
		copyOfGrid1.show("copyOfGrid1");
		
		//get and print the values at indices (0,0) and (150,150) of grid3
		float val0 = grid3.getAtIndex(0, 0);
		System.out.println("Value at (0,0): " + val0);
		float val150 = grid3.getAtIndex(150, 150);
		System.out.println("Value at (150,150): " + val150);
		
	}
	
	/**
	 * Creates a white circle with the given radius around the center 
	 * of the world coordinate system.
	 * @param grid
	 * @param r the radius of the circle
	 */	
	public static void circle (Grid2D grid, int r) {
		
		int width = grid.getWidth();
		int height = grid.getHeight();

		for (int x = -r; x <= r; x++) {
			for (int y = -r; y <= r; y++) {
				
				//test if the coordinates are in the circle
				if (x * x + y * y <= (r * r)) {
					
					//get the corresponding indices
					double[] idx = grid.physicalToIndex(x, y);
					int i = (int) idx[0];
					int j = (int) idx[1];
					
					if (i>=0 && j>=0 && i<width && j<height) {
						//set value at the indices
						grid.setAtIndex(i, j, 1.f);
					}
					
				}
			}
		}

	}
	
}
