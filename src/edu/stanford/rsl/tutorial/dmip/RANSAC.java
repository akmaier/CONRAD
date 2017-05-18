package edu.stanford.rsl.tutorial.dmip;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.InversionType;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import ij.gui.Plot;
import java.util.ArrayList;

/**
 * Exercise 4 of Diagnostic Medical Image Processing (DMIP)
 * @author Bastian Bier
 *
 */

public class RANSAC {

	/**
	 * Function calculation the ransac model estimate
	 * 
	 * @param points  	The point cloud where the line should be fit through
	 * @param minParam  The number of parameters required for the model  	
	 * @param p_opt		The probability of picking the right points for the models during the iterations
	 * @param p_out		The probability of an outliner  	  						
	 * 					
	 * @return solution the parameters of the resulting RANSAC line in a SimpleVector: [m,c]
	 */ 
	
	public SimpleVector commonRansac(SimpleMatrix points, int mn, double p_opt, double p_out){
		
		// Calculating the amount of required iterations
		int it = (int) (Math.log(1 - p_opt) / Math.log(1 - Math.pow(1 - p_out, mn)) + 1 );
		
		// Error of the best fitted line
		double error = Double.POSITIVE_INFINITY;
		
		// Solution vector
		SimpleVector solution = new SimpleVector(mn);
		
		for(int i = 0; i < it; i++)
		{
			// Select mn random points
			// Calculate the indexes of the points
			ArrayList<Integer> indexes = new ArrayList<Integer>();
			int randIdx = (int) (Math.random() / (1.f / points.getRows()));
			indexes.add(randIdx);
			
			for(int n = 1; n < mn; n++){
				randIdx = (int) (Math.random() / (1.f / points.getRows()));
				
				while(indexes.contains(randIdx))
				{
					randIdx = (int) (Math.random() / (1.f / points.getRows()));
				}
				indexes.add(randIdx);
			}

			// Calculate the parameters
			SimpleMatrix a = new SimpleMatrix(mn,mn);
			
			for(int n = 0;  n < mn; n++){
				a.setRowValue(n, points.getRow(indexes.get(n)));
			}
			
			SimpleVector lineParams = fitline(a);
			
			
		
			
			// Calculate the error of the estimated line
			// update the error and the parameters, if the current line has a smaller error
			double cur_err = lineError(lineParams, points);
			
			
			if(cur_err < error)
			{
				error = cur_err;
				solution = lineParams;
			}

		}
		
		return solution;
	}
	
	/**
	 * Function calculating a line through a point cloud using the SVD
	 * 
	 * @param points  	The point cloud where the line should be fit through 
	 * 					2 points result in a exact line
	 * 					>2 points result in a regression line
	 * 
	 * @return x_result the parameters of the line in a SimpleVector: [m,c]
	 */
	public SimpleVector fitline(SimpleMatrix points){
		
		// Build up the measurement matrix
		SimpleMatrix m = new SimpleMatrix(points.getRows(),2);
		SimpleVector b = new SimpleVector(points.getRows());
		m.fill(1);
		
		for(int i = 0; i < points.getRows(); i++){
			m.setElementValue(i, 0, points.getElement(i, 0));
			b.setElementValue(i, points.getElement(i, 1));
		}
		
		// Solution vector containing the estimated parameters m and c 
		SimpleVector x_result = new SimpleVector(2);
		
		// Calculate the parameters using the Pseudo-Inverse
		x_result = SimpleOperators.multiply(m.inverse(InversionType.INVERT_SVD), b);
		
		return x_result;
	}
	
	/**
	 * Calculate the error of a line
	 * 
	 * @param line_params  	Parameters of the line
	 * @param points  		The point cloud where the line should be fit through. 
	 * 					
	 * @return error the calculated error
	 */
	public double lineError(SimpleVector line_params, SimpleMatrix points){
		
		// Threshold defining the allowed distance of a point to the line
		double thresh = 0.2;
		
		double m = line_params.getElement(0);
		double c = line_params.getElement(1);
		
		
		SimpleVector point = new SimpleVector(1, m*1 + c); 
		
		
		SimpleVector n = new SimpleVector(-m,1);
		n = n.normalizedL2();
		
		double d = SimpleOperators.multiplyInnerProd(point, n);
		double error = 0;
		
		for (int i = 0; i < points.getRows(); ++i)
		{
			double dp = Math.abs(SimpleOperators.multiplyInnerProd(points.getRow(i), n) - d);
			
			if (dp > thresh)
			{
				error++;
			}
		}
		
		error /= points.getRows();
		
		
		return error;
	}
	
	
	public static void main(String[] args) {
		
		//
		RANSAC ransac = new RANSAC();
		
		//
		// The point cloud is defined
		//
		
		SimpleMatrix pts = new SimpleMatrix(7,2);
		pts.setRowValue(0, new SimpleVector(0,0));
		pts.setRowValue(1, new SimpleVector(1,1));
		pts.setRowValue(2, new SimpleVector(2,2));
		pts.setRowValue(3, new SimpleVector(3,3));
		pts.setRowValue(4, new SimpleVector(3.2,1.9));
		pts.setRowValue(5, new SimpleVector(4,4));
		pts.setRowValue(6, new SimpleVector(10,1.8));
	

		//
		// Regression Line 
		//
		
		// Create a scatter plot of the point cloud and fit a regression line
		Plot scatterPlot = new Plot("Regression Line", "X", "Y", Plot.DEFAULT_FLAGS);
		scatterPlot.setLimits(0, 11, 0, 5);
		scatterPlot.addPoints(pts.getCol(0).copyAsDoubleArray(), pts.getCol(1).copyAsDoubleArray(), Plot.BOX);
		scatterPlot.show();
		
		// Calculate the regression line through the given point cloud
		SimpleVector regressionLine = ransac.fitline(pts);
		
		// Add the regression line
		double y11 = regressionLine.getElement(0) * 11 + regressionLine.getElement(1);
		double y0 = regressionLine.getElement(0) * 0 + regressionLine.getElement(1);
		scatterPlot.drawLine(0, y0, 11, y11);
		
		
		//
		// RANSAC 
		//
		
		// Parameters for RANSAC
		double p_opt = 0.9999; 		// probability how likely it is to pick the right mn points
		double p_out = 0.2;			// probability of an outlier
		int min_number = 2;			// minimum number of datapoints required to build the model

		// Create a scatter plot of the point cloud and fit a RANSAC line
		Plot ransac_plot = new Plot("Ransac Line", "X", "Y", Plot.DEFAULT_FLAGS);
		ransac_plot.setLimits(0, 11, 0, 5);
		ransac_plot.addPoints(pts.getCol(0).copyAsDoubleArray(), pts.getCol(1).copyAsDoubleArray(), Plot.BOX);
		
		
		// Compute a line using the RANSAC algorithm and plot it
		SimpleVector ransacLine = ransac.commonRansac(pts, min_number, p_opt, p_out);
		double y1 = ransacLine.getElement(0) * 0 + ransacLine.getElement(1);
		double y2 = ransacLine.getElement(0) * 11 + ransacLine.getElement(1);
		ransac_plot.drawLine(0, y1, 11, y2);
		ransac_plot.show();
			
	}
}
