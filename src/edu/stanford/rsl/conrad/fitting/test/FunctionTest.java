package edu.stanford.rsl.conrad.fitting.test;

import ij.ImageJ;

import java.util.Random;

import org.junit.Test;

import edu.stanford.rsl.conrad.fitting.ConstrainedRANSACFittedFunction;
import edu.stanford.rsl.conrad.fitting.GaussianFunction;
import edu.stanford.rsl.conrad.fitting.LinearFunction;
import edu.stanford.rsl.conrad.fitting.RANSACFittedFunction;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;

/**
 * Class to test the functions.
 * @author akmaier
 *
 */
public class FunctionTest {
	/** Here we test RANSAC.
	 *	We produce a data set that has a couple of outliers
	 *	Most of the data to reflect the correct model parameters
	 *	On average RANSAC will do better on this test.
	 *  Try and have fun!
	 */
	@Test
	public void testRANSAC(){
		new ImageJ();
		int cardinality = 1000;
		double noise = 500;
		double x[] = new double [cardinality]; 
		double y[] = new double [cardinality];
		LinearFunction input = new LinearFunction();
		input.setM(0.5);
		input.setT(1.5);
		Random random = new Random();
		GaussianFunction gf = new GaussianFunction();
		gf.setMu(0);
		gf.setSigma(1.0);

		for (int i =0;i<cardinality; i++){
			x[i] = i;
			y[i] = input.evaluate(x[i]) + (gf.evaluate((random.nextDouble()-.05)*100)/gf.evaluate(0) * noise)*(random.nextDouble()-.5);
		}
		VisualizationUtil.createScatterPlot("Linear Fit", x, y, new LinearFunction()).show();
		RANSACFittedFunction ransac = new RANSACFittedFunction( new LinearFunction());
		ransac.setEpsilon(1);
		ransac.setNumberOfTries(10000);
		VisualizationUtil.createScatterPlot("RANSAC Linear Fit", x, y, ransac).show();
		try {
			Thread.sleep(1000000000);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	
	/** Here we test constrained RANSAC.
	 *	We produce a data set that has many outliers
	 *	Most of the data to reflect the correct model parameters
	 *	On average RANSAC will do better on this test.
	 *  Try and have fun!
	 */
	@Test
	public void testConstrainedRANSAC(){
		new ImageJ();
		int cardinality = 1000;
		double noise = 500;
		double x[] = new double [cardinality]; 
		double y[] = new double [cardinality];
		LinearFunction input = new LinearFunction();
		input.setM(0.5);
		input.setT(1.5);
		Random random = new Random();
		GaussianFunction gf = new GaussianFunction();
		gf.setMu(0);
		gf.setSigma(1.0);
		for (int i =0;i<cardinality; i++){
			x[i] = i;
			y[i] = input.evaluate(x[i]) + (gf.evaluate((random.nextDouble()-.5)*2)/gf.evaluate(0) * noise)*(random.nextDouble()-.5);
		}
		VisualizationUtil.createScatterPlot("Linear Fit", x, y, new LinearFunction()).show();
		ConstrainedRANSACFittedFunction ransac = new ConstrainedRANSACFittedFunction( new LinearFunction());
		ransac.setEpsilon(10);
		ransac.setNumberOfTries(10000);
		ransac.setLowerbound(new double[] {0.25, 1});
		ransac.setUpperbound(new double[] {0.75, 2});
		VisualizationUtil.createScatterPlot("Constrained RANSAC Linear Fit", x, y, ransac).show();
		RANSACFittedFunction ransac2 = new RANSACFittedFunction( new LinearFunction());
		ransac2.setEpsilon(10);
		ransac2.setNumberOfTries(10000);
		VisualizationUtil.createScatterPlot("RANSAC Linear Fit", x, y, ransac2).show();
		
		try {
			Thread.sleep(1000000000);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
}
