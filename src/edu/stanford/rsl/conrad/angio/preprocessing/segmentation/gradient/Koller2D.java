/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.preprocessing.segmentation.gradient;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.RegKeys;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.Roi;
import ij.plugin.filter.Convolver;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

public class Koller2D {
	Grid3D stack = null;
	int[] gSize = null;
	double[] gSpace =  null;
	private double[] scales;
	private ArrayList<float[]> filterBankDerivative;
	private double[][] xValueD;
	private double[][] yValueD;
	private double[][] xValueDN;
	private double[][] yValueDN;
	private Grid3D vesselnessImage;
	private int[][] scaleAtPoint;
	
	private Roi roi = null;

	//constructors
	public Koller2D(Grid3D g){
		this.stack = g;		
	}

	public Koller2D(String filename) {
		this.stack = ImageUtil.wrapImagePlus(IJ.openImage(filename));
	}
	//setter
	public void setScales(double... sc){
		this.scales = sc;
	}
	//getter
	public Grid3D getResult(){
		return vesselnessImage;
	}
	//main reads the image constructs it and calls the algorithm
	public static void main(String[] args){
		new ImageJ();

		String filename = ".../test1.tif";
		Roi roi =  new Roi(60,210,710,460);
		Koller2D v = new Koller2D(filename);
		v.setRoi(roi);
		v.setScales(new double[]{0.6, 1.0, 1.4, 1.7, 2.5});
		v.evaluate();
		Grid3D filt = v.getResult();
		filt.show();
	}
	
	
	//calls the private functions for initialisation and the algorithm
	public void evaluate(){
		init();
		evaluateInternal();
	}
	//initializes the stacksize, the spacing, the filterbanks
	private void init(){
		gSize = this.stack.getSize();
		gSpace = this.stack.getSpacing();
		
		if(roi == null){
			this.roi = new Roi(0,0,gSize[0],gSize[1]);
		}
		
		if(scales == null){
			scales = new double[6-1];
			for (int i = 1; i < 6 ; i++){
				scales[i-1] = gSpace[0]*i;
			}
		}
		setupFilterBank();
	}
	
	//sets up the filterbank. We need a laplacian and a gaussian filter for each scale
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
	
	//calls the algorithm for each slice.
	public void evaluateInternal() {
		Grid3D filtered = new Grid3D(gSize[0], gSize[1], gSize[2]);
		Grid3D size = new Grid3D(gSize[0], gSize[1], gSize[2]);
		filtered.setSpacing(gSpace);
		for(int k = 0; k < gSize[2]; k++){
			System.out.println("Koller filtering on slice "+String.valueOf(k+1)+" of "+gSize[2]+".");
			ImagePlus[] hessians = getHessian(stack.getSubGrid(k));
			//hessians[0].show("stack at" + " " + scales[k]);
			Grid2D v = getFilterResponse(hessians);
			Grid2D s = scalesToGrid(this.scaleAtPoint);
			size.setSubGrid(k, s);
			filtered.setSubGrid(k, v);
		}
//		filtered.show("filtered");
//		size.show("Point scales");
//		Grid3D mask = toMask(filtered,size);
//		mask.show("Binary mask.");
		this.vesselnessImage = filtered;
	}

	public Grid3D toMask(Grid3D cent, Grid3D size){
		Grid3D lines = new Grid3D(gSize[0],gSize[1],gSize[2]);
		lines.setSpacing(gSpace);
		for(int k = 0; k < gSize[2]; k++){
			for(int i = 0; i < gSize[0]; i++){
				for(int j = 0; j < gSize[1]; j++){
					float val = cent.getAtIndex(i, j, k);
					if(val > 0.5){
						float siz = size.getAtIndex(i, j, k);
						int sizPx = (int)Math.ceil(siz/gSpace[0]);
						for(int x = -sizPx/2; x < sizPx/2; x++){
							int idx = i+x;
							if(idx > 0 && idx < gSize[0]){
								for(int y = -sizPx/2; y < sizPx/2; y++){
									int idy = j+y;
									if(idy > 0 && idy < gSize[1]){
										lines.setAtIndex(idx, idy, k, 1);
									}
								}
							}
						}
					}
				}
			}
		}
		return lines;
	}
	
	private Grid2D scalesToGrid(int[][] arr){
		Grid2D g = new Grid2D(arr.length, arr[0].length);
		for(int i = 0; i < arr.length; i++){
			for(int j = 0; j < arr[0].length; j++){
				g.setAtIndex(i, j, (float)scales[arr[i][j]]);
			}
		}
		return g;
	}
	
	//calculates the direction of tubular structures proposed in the paper
	public ImagePlus[] getHessian(Grid2D stack2) {
		
		ImagePlus[] hessians = new ImagePlus[scales.length];
		Configuration.loadConfiguration();
		ExecutorService executorService = Executors.newFixedThreadPool(
				Integer.valueOf(Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.MAX_THREADS)));
		Collection<Future<?>> futures = new LinkedList<Future<?>>();
		
		for(int i = 0; i < scales.length; i++){
			final int count = i;
			futures.add(
				executorService.submit(new Runnable() {

					@Override
					public void run() {	
						Convolver c = null;	
						//get the filter for each scale
						float[] derivative = filterBankDerivative.get(count);
				
						float[] unityDerivative = new float[]{-1f,0f,1f};
						ImageProcessor gridAsImp = ImageUtil.wrapGrid2D(stack2);
						
						//calculate the values of the Hessian
						ImageProcessor ip_xx = gridAsImp.duplicate();
						c = new Convolver();
						c.convolveFloat(ip_xx, derivative, derivative.length, 1);
						c = new Convolver();
						c.convolveFloat(ip_xx, unityDerivative, unityDerivative.length, 1);
												
						ImageProcessor ip_xy = gridAsImp.duplicate();
						c = new Convolver();
						c.convolveFloat(ip_xy, derivative, derivative.length, 1);
						c = new Convolver();
						c.convolveFloat(ip_xy, unityDerivative, 1, unityDerivative.length);
					
						ImageProcessor ip_yy = gridAsImp.duplicate();
						c = new Convolver();
						c.convolveFloat(ip_yy, derivative, 1, derivative.length);
						c = new Convolver();
						c.convolveFloat(ip_yy, unityDerivative,1, unityDerivative.length);
						
						ImageProcessor ipDirect = new FloatProcessor(gSize[0], gSize[1]);
						//calculation of the direction at each point
						for (int x = 0; x < gSize[0]; x++){
							for (int y = 0; y < gSize[1]; y++){
								float s_xx = ip_xx.getf(x,y);
								float s_xy = ip_xy.getf(x,y);
								float s_yy = ip_yy.getf(x,y);
								double argument = 2*s_xy/(s_xx-s_yy);
								float direction =  (float) ((Math.atan(argument))/2);
								
								// y u no tell me blondel... Auswertung für den Fall dass die y-Richtung dominant ist:
								if (s_xx<s_yy){
									direction += Math.PI/2;
								}
								//direction = (float) Math.asin(Math.sqrt(argument));
								
								ipDirect.setf(x, y, direction);
							} 
						}
						//save all for debug reasons
						ImageStack hessianStack = new ImageStack(gSize[0], gSize[1]);
						hessianStack.addSlice("Hessian"+ "_Direction_Octave_at_"+scales[count]+"mm", ipDirect);
						hessianStack.addSlice("Hessian"+ "_Original_Octave_at_" + scales[count] +"mm", gridAsImp);
						hessianStack.addSlice("Hessian"+ "_Original_Octave_at_" + scales[count] +"mm", ip_xx);
						hessianStack.addSlice("Hessian"+ "_Original_Octave_at_" + scales[count] +"mm", ip_xy);
						hessianStack.addSlice("Hessian"+ "_Original_Octave_at_" + scales[count] +"mm", ip_yy);
						ImagePlus plus = new ImagePlus("Stack at "+scales+"mm", hessianStack);
						hessians[count] = plus;
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
			   }
		}
		return hessians;
	}
	
	
	
	public Grid2D getFilterResponse(ImagePlus[] hessians){
		
		MultiChannelGrid2D vesselness = new MultiChannelGrid2D(gSize[0], gSize[1], scales.length);
		vesselness.setSpacing(gSpace[0], gSpace[1]);
		
		//initialize the x and y values of the unitydirection at each scale
		this.xValueD = new double[gSize[0]][gSize[1]];
		this.yValueD = new double[gSize[0]][gSize[1]];
		this.xValueDN = new double[gSize[0]][gSize[1]];
		this.yValueDN = new double[gSize[0]][gSize[1]];
		this.scaleAtPoint = new int[gSize[0]][gSize[1]];
				
		//deriving at each scale
		for (int sc = 0 ; sc < scales.length ; sc++){
			
			Convolver c = null;	
			ImageStack hessian = hessians[sc].getImageStack();
			Grid2D direc = ImageUtil.wrapFloatProcessor((FloatProcessor) hessian.getProcessor(1));
			Grid2D orig = ImageUtil.wrapFloatProcessor((FloatProcessor) hessian.getProcessor(2));
			float[] derivative = filterBankDerivative.get(sc);
			ImageProcessor gridAsImp = ImageUtil.wrapGrid2D(orig);
			//calculating the filter response
			//deriving in x direction
			ImageProcessor ip_x = gridAsImp.duplicate();
			c = new Convolver();
			c.convolveFloat(ip_x, derivative, derivative.length, 1);
			//deriving in y direction
			ImageProcessor ip_y = gridAsImp.duplicate();
			c = new Convolver();
			c.convolveFloat(ip_y, derivative, 1, derivative.length);
		
			for (int x = 0; x < gSize[0]; x++){
				for (int y = 0 ; y < gSize[1] ; y++){
					if(roi.contains(x, y)){
						//save the maximal response
						double maxResponse = -Double.MAX_VALUE;
						//initialize unityvectors
						double angle = (double) direc.getAtIndex(x, y);
						SimpleVector uD = new SimpleVector(Math.cos(angle+Math.PI), Math.sin(angle+Math.PI));
						
						//multiply by the current scale
						uD.multiplyBy(scales[sc]);
						SimpleVector uDN = uD.clone();
						uDN.multiplyBy(-1);
						xValueD[x][y] = (uD.getElement(0));
						yValueD[x][y] = (uD.getElement(1));
						xValueDN[x][y] = (uDN.getElement(0));
						yValueDN[x][y] = (uDN.getElement(1));
						int xPointD = x+(int) Math.round(xValueD[x][y]);
						int yPointD = y+(int) Math.round(yValueD[x][y]);
						int xPointDN = x+(int) Math.round(xValueDN[x][y]);
						int yPointDN = y+(int) Math.round(yValueDN[x][y]);
						//check for boundaries
						if ( xPointD > 0 && xPointD < gSize[0] && yPointD > 0 && yPointD < gSize[1]
								&&  xPointDN > 0 && xPointDN < gSize[0] &&  yPointDN > 0 && yPointDN < gSize[1]){
							//nabla at point in unityvectordirection
							double s_xD = ip_x.getf(xPointD, yPointD);
							double s_yD = ip_y.getf(xPointD, yPointD);
							//weight by the unityvectorvalue
							double nablaD = s_xD*xValueD[x][y]/scales[sc] + s_yD*yValueD[x][y]/scales[sc];
							double s_xDN = ip_x.getf(xPointDN ,yPointDN);
							double s_yDN = ip_y.getf(xPointDN,yPointDN);
							double nablaDN = s_xDN*xValueDN[x][y]/scales[sc] + s_yDN*yValueDN[x][y]/scales[sc];
							maxResponse = Math.min(nablaD , nablaDN);
							//!other possible filterresponses
							// -> the min version gives a better response for the centerline 
							//!!the additionversion looks better
							//maxResponse = Math.sqrt(nablaD * nablaD + nablaDN* nablaDN);
							//maxResponse = nablaD + nablaDN;
							vesselness.putPixelValue(x, y, sc, (maxResponse));
						}
					}
				}	
			}
		}	
		//vesselness.getChannel(0).show("Derivativeresponse at scale" + scales[0]);
		//set new 2DGrid
		Grid2D filtered = new Grid2D(gSize[0], gSize[1]);
		filtered.setSpacing(gSpace[0], gSpace[1]);
		
//		new ImageJ();
//		vesselness.show();
		
		double maxV = -Double.MAX_VALUE;
		for(int x = 0; x < gSize[0]; x++){
			for(int y = 0; y < gSize[1]; y++){
				if(roi.contains(x, y)){
					//get the highest value computed over all scales
					double v = -Double.MAX_VALUE;
					scaleAtPoint[x][y] = 0;
					//for each scale
					for(int sca = 0; sca < scales.length; sca++){
						double weightedPointValue = vesselness.getPixelValue(x, y, sca);
						//save the best scale
						if(v < weightedPointValue){
							scaleAtPoint[x][y] = sca;
						}
						//save the highest response at this point
						maxV = Math.max(maxV, v);
						v = Math.max(v, weightedPointValue);
					}
					//set the filtered image to the maxvalue
					filtered.setAtIndex(x, y, (float)v);
				}
			}
		}
		
		//filtered.show("Value at the scale with most response");
		//new Grid for the multiscaledirectionmap
		Grid2D multiscaleDirectionMap = new Grid2D(gSize[0], gSize[1]);
		multiscaleDirectionMap.setSpacing(gSpace[0], gSpace[1]);
		//set the value of each point to the direction at the best scale
		//computationheavy but useless only for visualizationreasons
		for (int a = 0 ; a < gSize[0]; a++){
			for (int b = 0 ; b < gSize[1]; b++){
				for (int atScale = 0; atScale < scales.length; atScale++){
					if (atScale == scaleAtPoint[a][b]){
						ImageStack hessians2 = hessians[atScale].getImageStack();
						Grid2D direc3 = ImageUtil.wrapFloatProcessor((FloatProcessor) hessians2.getProcessor(1));
						multiscaleDirectionMap.setAtIndex(a, b, direc3.getAtIndex(a, b));
					}
				}
			}
		}
		//multiscaleDirectionMap.show("Multiscale Direction Map");
		//at the moment just the maximum
		Grid2D subpixelMaximum = new Grid2D(gSize[0], gSize[1]);
		subpixelMaximum.setSpacing(gSpace[0], gSpace[1]);
		for (int u = 0 ; u < gSize[0]; u++){
			for (int v = 0 ; v < gSize[1]; v++){
				int scale = scaleAtPoint[u][v];
				double angle = (double) multiscaleDirectionMap.getAtIndex(u, v);
				//get the orthogonal vectors
				SimpleVector uD = new SimpleVector(Math.cos(angle+Math.PI), Math.sin(angle+Math.PI));
				uD.multiplyBy(scales[scale]);
				SimpleVector uDN = uD.clone();
				uDN.multiplyBy(-1);
				SimpleVector xy = new SimpleVector(u,v);
				xy.multiplyBy(gSpace[0]);
				xValueD[u][v] = (uD.getElement(0));
				yValueD[u][v] = (uD.getElement(1));
				xValueDN[u][v] = (uDN.getElement(0));
				yValueDN[u][v] = (uDN.getElement(1));
				//point precalculation
				int xPointD = u+(int) Math.round(xValueD[u][v]);
				int yPointD = v+(int) Math.round(yValueD[u][v]);	
				int xPointDN = u+(int) Math.round(xValueDN[u][v]);
				int yPointDN = v+(int) Math.round(yValueDN[u][v]);

				//check for boundaries
				if ( xPointD > 0 && xPointD < gSize[0] && yPointD > 0 && yPointD < gSize[1]
						&&  xPointDN > 0 && xPointDN < gSize[0] &&  yPointDN > 0 && yPointDN < gSize[1]){
					float responseAtD = filtered.getPixelValue(xPointD,yPointD);
					float responseAtDNeg = filtered.getPixelValue(xPointDN, yPointDN);
					float value = filtered.getPixelValue(u, v);
					SimpleVector position = new SimpleVector(u,v);
					position.multipliedBy(gSpace[0]);
					if(responseAtD < value && responseAtDNeg < value){
						int posX = (int)position.getElement(0);
						int posY = (int)position.getElement(1);
						if (posX > 0 && posX < gSize[0] && posY > 0 && posY < gSize[1]){
							subpixelMaximum.putPixelValue(posX,posY, filtered.getPixelValue(posX,posY));
							
						}
					}
				}
			}
		}
		return subpixelMaximum;
	}

	public Roi getRoi() {
		return roi;
	}

	public void setRoi(Roi roi) {
		this.roi = roi;
	}
}

