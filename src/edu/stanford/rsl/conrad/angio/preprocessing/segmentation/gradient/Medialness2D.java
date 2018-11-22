/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.preprocessing.segmentation.gradient;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.process.FloatProcessor;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.RegKeys;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.Dijkstra2D.EightConnectedLattice;

/**
 * Implements the Medialness measure as proposed by Gülsün et al. for the 2D case.
 * Here, we 
 * @author Mathias Unberath
 *
 */
public class Medialness2D {
		
	/** Scales are to be given in mm */
	private double[] scales = new double[]{0,0.5,1};
	
	ArrayList<float[]> filterBankDerivative = null;
	
	private Grid3D medialness = null;
	private Grid3D radii = null;
	private Grid3D directions = null;
		
	Grid3D stack = null;
	int[] gSize = null;
	double[] gSpace =  null;
	
	private double minR;
	private double maxR;
	private double dRadius = 0;
	private double[] lineSamplesPos = null; // in px
	private double[] lineSamplesNeg = null; // in px
	private int nRadii = 10;
	
	private int lineLength = 0;
	
	
	public static void main(String[] args){
		String test = ".../test";
		String testFile = test+ ".tif";
			
		new ImageJ();
		ImagePlus imp = IJ.openImage(testFile);
		Grid3D stack = ImageUtil.wrapImagePlus(imp);
		imp.show();
		
		Medialness2D medialnessFilt = new Medialness2D(stack, 0.5, 5, 10);
		medialnessFilt.setScales(new double[]{0.5,1,1.5,2});
		medialnessFilt.evaluate();
		Grid3D med = medialnessFilt.getMedialnessImage();
		med.show();
		IJ.saveAsTiff(ImageUtil.wrapGrid3D(med, ""), test+"_medialness_mag.tif");
		
		Grid3D rad = medialnessFilt.getRadiusImage();
		rad.show();
		IJ.saveAsTiff(ImageUtil.wrapGrid3D(rad, ""), test+"_medialness_rad.tif");
		
		Grid3D ang = medialnessFilt.getDirectionsImage();
		ang.show();
		IJ.saveAsTiff(ImageUtil.wrapGrid3D(ang, ""), test+"_medialness_ang.tif");
		
	}

	public Medialness2D(Grid3D stack, double minRadius, double maxRadius, int nRadii){
		
		this.stack = stack;		
		this.gSize = stack.getSize();
		double[] gSpacing = stack.getSpacing();
		if(gSpacing[0] != gSpacing[1]){
			//TODO RESAMPLE IMAGE TO SAME SPACING
			// Else, line extraction becomes complicating and directions are not properly defined
			gSpace = new double[]{gSpacing[0], gSpacing[0], gSpacing[2]};
		}else{
			gSpace = gSpacing;
		}
		this.minR = minRadius;
		this.maxR = maxRadius;
		this.nRadii = nRadii;
	}
	
	public Medialness2D(Grid2D slice, double minRadius, double maxRadius, int nRadii){
		
		this.gSize = slice.getSize();
		double[] gSpacing = slice.getSpacing();
		if(gSpacing[0] != gSpacing[1]){
			//TODO RESAMPLE IMAGE TO SAME SPACING
			// Else, line extraction becomes complicating and directions are not properly defined
			gSpace = new double[]{gSpacing[0], gSpacing[0], 1};
		}else{
			gSpace = gSpacing;
		}
		this.minR = minRadius;
		this.maxR = maxRadius;
		this.nRadii = nRadii;
	}
	
	private void init(){
		setupFilterBank();
		lineLength = (int)Math.ceil(maxR / gSpace[0])
					+ filterBankDerivative.get(scales.length-1).length/2 + 1;
		
		this.dRadius = (maxR-minR) / nRadii;
		this.lineSamplesPos = new double[nRadii];
		this.lineSamplesNeg = new double[nRadii];
		
		for(int i = 0; i < nRadii; i++){
			lineSamplesPos[i] = lineLength + (minR + i*dRadius)/gSpace[0];
			lineSamplesNeg[i] = lineLength - (minR + i*dRadius)/gSpace[0];
		}
	}
	
	public void prepareForSerializedEvaluation(){
		init();
	}
	
	public double[] evaluateAtPoint(Grid2D slice, int u, int v, EightConnectedLattice neighbor){
		double[] response = new double[3];
		
		int alpha = neighbor.getShift()[2];
		double angle = Math.PI/4 * alpha;	
			
		Grid1D line = getLineAtPointAndOrientation(slice, u, v, angle);
		ArrayList<Grid1D> deriv = derive(line);	
			
		double[] responseAndRadius = calculateEdgeResponse(deriv);
		response[0] = responseAndRadius[0];
		response[1] = responseAndRadius[1]*dRadius+minR;
		response[2] = angle;
		
		return response;
	}
	
	public void evaluate(){
		init();
		
		this.medialness = new Grid3D(gSize[0],gSize[1],gSize[2]);
		medialness.setSpacing(gSpace[0], gSpace[1], gSpace[2]);
		this.radii = new Grid3D(gSize[0],gSize[1],gSize[2]);
		radii.setSpacing(gSpace[0], gSpace[1], gSpace[2]);
		this.directions = new Grid3D(gSize[0],gSize[1],gSize[2]);
		directions.setSpacing(gSpace[0], gSpace[1], gSpace[2]);
		
		
		if(Configuration.getGlobalConfiguration() == null){
			Configuration.loadConfiguration();
		}
		ExecutorService executorService = Executors.newFixedThreadPool(
				Integer.valueOf(Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.MAX_THREADS)));
		Collection<Future<?>> futures = new LinkedList<Future<?>>();
		
		for(int count = 0; count < gSize[2]; count++){
			final int k = count;
			futures.add(
				executorService.submit(new Runnable() {
					@Override
					public void run() {			
						System.out.println("Medialness on slice "+String.valueOf(k+1)+" of "+gSize[2]+".");
						Grid2D medSlice = new Grid2D(gSize[0], gSize[1]);
						medSlice.setSpacing(gSpace[0], gSpace[1]);
						Grid2D radiiSlice = new Grid2D(gSize[0], gSize[1]);
						radiiSlice.setSpacing(gSpace[0], gSpace[1]);
						Grid2D dirSlice = new Grid2D(gSize[0], gSize[1]);
						dirSlice.setSpacing(gSpace[0], gSpace[1]);
						for(int u = 0; u < gSize[0]; u++){				
							for(int v = 0; v < gSize[1]; v++){
								double maxEdgeResponse = -Double.MAX_VALUE;
								double maxRadius = 0;
								double dir = 0;
								for(int alpha = 0; alpha < 8; alpha++){						
									double angle = Math.PI/4 * alpha;	
									
									Grid1D line = getLineAtPointAndOrientation(stack.getSubGrid(k), u, v, angle);
									ArrayList<Grid1D> deriv = derive(line);	
									
									double[] responseAndRadius = calculateEdgeResponse(deriv);
									double response = responseAndRadius[0];
									double radius = responseAndRadius[1];
									if(response > maxEdgeResponse){
										maxEdgeResponse = response;
										maxRadius = radius;
										dir = angle;
									}
									maxEdgeResponse = Math.max(maxEdgeResponse, response);
								}
								medSlice.setAtIndex(u, v, (float)maxEdgeResponse);
								radiiSlice.setAtIndex(u, v, (float)(maxRadius*dRadius+minR));
								dirSlice.setAtIndex(u, v, (float)dir);
							}
						}
						medialness.setSubGrid(k, medSlice);
						radii.setSubGrid(k, radiiSlice);
						directions.setSubGrid(k, dirSlice);
					}
				}) // exe.submit()
			); // futures.add()	
		}
		for (Future<?> future : futures){
			   try{
			       future.get();
			   }catch (InterruptedException e){
			       throw new RuntimeException(e);
			   }catch (ExecutionException e){
			       throw new RuntimeException(e);
			   }
		}
	}
	
	private double[] calculateEdgeResponse(ArrayList<Grid1D> derivatives){
		double response = 0;
		double radius = 0;
		
		for(int s = 0; s < scales.length; s++){
			double[] maxPos = new double[nRadii];
			double normPos = 1;
			
			double[] maxNeg = new double[nRadii];
			double normNeg = 1;
			Grid1D grad = derivatives.get(s);
			double[] valsPos = new double[nRadii];
			double[] valsNeg = new double[nRadii];
			// precalculate so that we can use normalized edgeresponse according to Gülsün also in the 2D case
			for(int r = 0; r < nRadii; r++){
				// here we are on the positive side. We want to have negative gradients and punish positive gradients
				valsPos[r] = -InterpolationOperators.interpolateLinear(grad, lineSamplesPos[r]);
				maxPos[r] = (r==0)?0:Math.max(valsPos[r], maxPos[r-1]);
				normPos = Math.max(-valsPos[r], normPos);
				// here we are on the negative side. therefore we use the opposite sign of the gradient
				valsNeg[r] = +InterpolationOperators.interpolateLinear(grad, lineSamplesNeg[r]);
				maxNeg[r] = (r==0)?0:Math.max(valsNeg[r], maxNeg[r-1]);
				normNeg = Math.max(-valsNeg[r], normNeg);
			}
			for(int r = 0; r < nRadii; r++){
				double medialness = (Math.max((-valsPos[r] - maxPos[r]), 0) / normPos
									+ Math.max((-valsNeg[r] - maxNeg[r]), 0) / normNeg) / 2;
				if(medialness > response){
					response = medialness;
					radius = r;
				}
			}
		}		
		return new double[]{response, radius};
	}
	
	private void setupFilterBank() {
		this.filterBankDerivative = new ArrayList<float[]>();
		for(int i = 0; i < scales.length; i++){
			double scale = scales[i];
			float[] derivative;
			if(scale == 0){
				derivative = new float[]{-1,0,1};
			}else{
				int filtSize = 1+2*4*(int)Math.ceil(scale/gSpace[0]);
				derivative = new float[filtSize];
				double denom = scale*scale;
				double norm = (1 / Math.sqrt(2*Math.PI*denom));
				for(int j = 0; j < filtSize; j++){
					double x = (j-(filtSize-1)/2)*gSpace[0];
					derivative[j] = (float)( (- x / denom) * norm * Math.exp(- 0.5 * x*x / denom) );
				}
			}
			filterBankDerivative.add(derivative);
		}		
	}

	
	public void setScales(double[] scales) {
		this.scales = scales;
	}

	private Grid1D getLineAtPointAndOrientation(Grid2D slice, double u, double v, double alpha){
		Grid1D line = new Grid1D(2*lineLength+1);
		line.setSpacing(gSpace[0]);
		
		SimpleVector uAlpha = new SimpleVector(Math.cos(alpha), Math.sin(alpha));
		SimpleVector uAlphaOrth = new SimpleVector(uAlpha.getElement(1), -uAlpha.getElement(0));
		
		SimpleVector x0 = new SimpleVector(u,v);
		x0.multiplyBy(gSpace[0]);
		x0.add(uAlpha.multipliedBy(0.5*gSpace[0]));

		for(int i = -lineLength; i < lineLength+1; i++){
			SimpleVector x = x0.clone();
			x.add(uAlphaOrth.multipliedBy(i*gSpace[0]));
			double[] idx = new double[]{x.getElement(0)/gSpace[0], x.getElement(1)/gSpace[0]};
			double val = 0;
			if(idx[0] < 0 ||  idx[0] > gSize[0]-1 || idx[1] < 0 ||  idx[1] > gSize[1]-1){
			}else{
				val = InterpolationOperators.interpolateLinear(slice, idx[0], idx[1]);
			}
			line.setAtIndex(i+lineLength, (float)val);
		}		
		return line;
	}
	
	/**
	 * Derives a line using all scale-space derivative kernels and a returns an array list of derivatives
	 * @param line
	 * @return
	 */
	private ArrayList<Grid1D> derive(Grid1D line){
		ArrayList<Grid1D> derivatives = new ArrayList<Grid1D>();
		for(int i = 0; i < filterBankDerivative.size(); i++){
			float[][] array = new float[1][line.getSize()[0]];
			array[0] = line.getBuffer().clone();
			float[] filt = filterBankDerivative.get(i);
			FloatProcessor deriv = new FloatProcessor(array);
			deriv.convolve(filt, 1, filt.length);
			derivatives.add(new Grid1D(deriv.getFloatArray()[0]));
		}
		return derivatives;		
	}
	
	public Grid3D getMedialnessImage(){
		return medialness;
	}
	
	public Grid3D getRadiusImage(){
		return radii;
	}
	
	public Grid3D getDirectionsImage(){
		return directions;
	}
}
