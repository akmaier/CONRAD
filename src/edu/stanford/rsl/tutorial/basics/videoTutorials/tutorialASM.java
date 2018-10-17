package edu.stanford.rsl.tutorial.basics.videoTutorials;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels.ActiveShapeModel;
import edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels.GPA;
import edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels.PCA;
import edu.stanford.rsl.conrad.geometry.shapes.mesh.DataMatrix;
import edu.stanford.rsl.conrad.geometry.shapes.mesh.Mesh;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import ij.ImageJ;

/**
 * Tutorial code to show how the ActiveShapeModel class is used.
 * A simplistic example helps to understand the concept. 
 * This code is used for our video tutorial series on the Conrad API.
 * 
 * @author Anna Gebhard
 * 
 * Further readings:
 * [1] 	T. Cootes (2000), “An Introduction to Active Shape Models”,
 * 		in Model-based Methods in Analysis of Biomedical Images (Oxford Univ Press) Chap. 7, pp. 223–248
 * [2]	T. Heimann, H. Meinzer (2009), “Statistical Shape Models for 3D Medical Image Segmentation: A Review”,
 *		Medical Image Analysis, vol.13, pp. 543–563
 * [3]	J. C. Gower et al. (1975), “Generalized Procrustes Analysis”,
 * 		Psychometrika, vol. 40, pp. 33–51
 * 
 */

public class tutorialASM {
	
	public static void main(String [] args) {
		
		new ImageJ();
		
		int K = 2; //number of training samples
        int N = 400; //number of sampling points for shape
       
        int dimX = 400; //x-dimension of images
        int dimY = 400; //y-dimension of images
       
        double dX = 1.0; //spacing in x
        double dY = 1.0; //spacing in y
       
        int offX = dimX/2; //shift in x to image center
        int offY = dimY/2; //shift in y to image center
       
        int[] shapePar = {50, 80};
		
		//create two grids for the two shapes
		Grid2D grid1 = new Grid2D(dimX,dimY);
		grid1.setSpacing(dX,dY);
		grid1.setOrigin(-offX,-offY);
		Grid2D grid2 = new Grid2D(dimX,dimY);
		grid2.setSpacing(dX,dY);
		grid2.setOrigin(-offX,-offY);
		
		//create two different trapezoids on the grids
		createTrap(grid1, shapePar[0]);
		createTrap(grid2, shapePar[1]);
		
		//show the grids
		grid1.show("shape1");
		grid2.show("shape2");
		
		//create two shape matrices
		SimpleMatrix shape1 = new SimpleMatrix(N,2);
		SimpleMatrix shape2 = new SimpleMatrix(N,2);
		
		//create the shapes from the grids
		createShapeFromGrid(shape1, grid1);
		createShapeFromGrid(shape2, grid2);
		
		//use generalized procrustes analysis to get the mean shape (consensus)
		GPA gpa = new GPA(K);
		gpa.addElement(0, shape1);
		gpa.addElement(1, shape2);
		gpa.runGPA();
		SimpleMatrix consensus = gpa.getScaledAndShiftedConsensus();
		
		//create a grid for the mean shape
		Grid2D con = new Grid2D(dimX,dimY);
		con.setSpacing(dX,dY);
		con.setOrigin(-offX,-offY);
		
		//get the grid from the mean shape matrix
		createGridFromShape(con, consensus);
		
		//show the mean shape
		con.show("mean shape");
		
		//run the principal component analysis
		DataMatrix datam = new DataMatrix(gpa);
		PCA pca = new PCA(datam);
		pca.run();
		
		//calculate the number of principal modes
		double[] ev = pca.eigenValues;
		int numpc = pca.getPrincipalModesOfVariation(ev);
		System.out.println("Number of principal components: " + numpc);
		System.out.println("eigenvalue 1: " + ev[0]);
		System.out.println("eigenvalue 2: " + ev[1]);
		
		//create the active shape model
		ActiveShapeModel asm = new ActiveShapeModel(pca);
		
		//use the weights 1 and 0 to create a new shape
		double[] weights = {1, 0};
		Mesh mesh1 = asm.getModel(weights);
		SimpleMatrix shape3 = mesh1.getPoints();
		
		Grid2D grid3 = new Grid2D(dimX,dimY);
		grid3.setSpacing(dX,dY);
		grid3.setOrigin(-offX,-offY);
		
		createGridFromShape(grid3, shape3);
		
		//show the shape with the weights 1 and 0
		grid3.show("Weights: (1,0)");
		
		//use the weights 0 and 20 to create a new shape
		weights[0] = 0;
		weights[1] = 200;
		mesh1 = asm.getModel(weights);
		shape3 = mesh1.getPoints();
		
		Grid2D grid4 = new Grid2D(dimX,dimY);
		grid4.setSpacing(dX,dY);
		grid4.setOrigin(-offX,-offY);
		
		createGridFromShape(grid4, shape3);
		
		//show the shape with the weights 0 and 20
		grid4.show("Weights: (0,200)");
		
		//use the weights 0 and 0 to create a new shape (should be the mean shape)
		weights[0] = 0;
		weights[1] = 0;
		mesh1 = asm.getModel(weights);
		shape3 = mesh1.getPoints();

		Grid2D grid5 = new Grid2D(dimX,dimY);
		grid5.setSpacing(dX,dY);
		grid5.setOrigin(-offX,-offY);
		createGridFromShape(grid5, shape3);
		
		//show the shape with the weights 1 and 0
		grid5.show("Weights: (0,0)");
	
	}
	
	/**
	 * creates a trapezoid on the grid between the points 
	 * (-100,-100), (100,-100), (-d,100), (d,100)
	 * (consists of 400 points)
	 * @param grid the grid the trapezoid is created on
	 * @param d half the length of one side of the trapezoid
	 */
	public static void createTrap (Grid2D grid, int d) {
		
		int a = 200/(100-d);
		int b = -100+100*a;
		
		for (int i=-100; i<=-d; i++) {
			int h = (int) grid.physicalToIndex(i, a*i+b)[0];
			int k = (int) grid.physicalToIndex(i, a*i+b)[1];
			int l = (int) grid.physicalToIndex(-i, a*i+b)[0];
			int m = (int) grid.physicalToIndex(-i, a*i+b)[1];
			grid.setAtIndex(h, k, 1);
			grid.setAtIndex(l, m, 1);
		}
		
		for (int i=-d; i<=d; i++) {
			int h = (int) grid.physicalToIndex(i, 100)[0];
			int k = (int) grid.physicalToIndex(i, 100)[1];
			grid.setAtIndex(h, k, 1);
		}
		
		for (int i=-100; i<=100; i++) {
			int h = (int) grid.physicalToIndex(i, -100)[0];
			int k = (int) grid.physicalToIndex(i, -100)[1];
			grid.setAtIndex(h, k, 1);
		}
	}
	
	/**
	 * creates the shape in matrix form (the points are the rows of the matrix
	 * from the shape in the grid
	 * @param shape the matrix which will be filled with points
	 * @param grid the grid with a shape formed by 400 points
	 */
	public static void createShapeFromGrid(SimpleMatrix shape, Grid2D grid) {
		
		int count = 0;
		
		for (int i=-200;i<200;i++) {
			for (int j=-200;j<200;j++) {
				int h = (int) grid.physicalToIndex(i,j)[0];
				int k = (int) grid.physicalToIndex(i,j)[1];
				if (grid.getAtIndex(h, k) == 1) {
					shape.setElementValue(count, 0, i);
					shape.setElementValue(count, 1, j);
					count++;
				}
			}
		}
	}
	
	/**
	 * creates a grid from the shape in matrix form
	 * @param grid the grid the shape will be written on
	 * @param shape the shape in matrix form
	 */
	public static void createGridFromShape(Grid2D grid, SimpleMatrix shape) {
		
		for (int i=0; i<shape.getRows(); i++) {
			double x = shape.getElement(i, 0);
			double y = shape.getElement(i, 1);
			int a = (int) grid.physicalToIndex(x, y)[0];
			int b = (int) grid.physicalToIndex(x, y)[1];
			grid.setAtIndex(a, b, 1);
		}
	}
	
}

/*
 * Copyright (C) 2010-2018 - Anna Gebhard
 * CONRAD is developed as an Open Source project under the GNU General Public License
 * (GPL).
 */