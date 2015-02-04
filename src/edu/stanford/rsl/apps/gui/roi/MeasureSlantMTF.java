package edu.stanford.rsl.apps.gui.roi;


import ij.IJ;

import java.awt.Rectangle;

import edu.stanford.rsl.conrad.utils.FFTUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;

/**
 * Estimate the pre-sampled (pre-pixelated) MTF of an X-ray detector if a slanted edge is used.
 * 
 * This code is ported from C.  To see the original code, login to roentgen.stanford.edu and look at:
 * /home/group/data_fahrig/fahrigDirectoryFromLucas/mtf/original/mtfslant2.c
 * Be warned, that file is poorly commented!  I have done my best to infer the intention of the code.
 * 
 * @author Derek Macklin
 *
 */
public class MeasureSlantMTF extends EvaluateROI {
	
	/* Perform linear regression on a set of points
	 * 
	 * @param x		Array of x-coordinates of points
	 * @param y		Array of y-coordinates of points
	 * @param n		Number of points
	 * @return		An array containing three values: the slope of the best-fit line, the y-intercept for the best-fit line, and the Pearson correlation
	 */
	private double [] regress(double x[] , double y [], int n)
	{
		double sumx, sumy, sumxy, sumx2, sumy2;
		double slope, intercept, corr;
		double ret[]=new double[3];

		sumx = sumy = sumxy = sumx2 = sumy2 = 0.0;
		for (int i=0; i<n; ++i){
			sumx = sumx + x[i];
			sumy = sumy + y[i];
			sumxy = sumxy + x[i] * y[i];
			sumx2 = sumx2 + x[i] * x[i];
			sumy2 = sumy2 + y[i] * y[i];
		}

		   /* Linear least-squares regression */
		   slope = (n * sumxy - sumx * sumy ) / (n * sumx2 -  (sumx*sumx) );
		   intercept = (sumy - slope * sumx) / n;

		   /* sample Pearson correlation */
		   corr = (n * sumxy - sumx * sumy) / Math.sqrt((n * sumx2 - ( sumx*sumx )) * (n * sumy2 - ( sumy*sumy )) );
		   
		   ret[0]=slope;
		   ret[1]=intercept;
		   ret[2]=corr;
		   return ret;

	}
	
	@Override
	public Object evaluate() {
		if (configured) {
				int thispix[], hit[];
				int xsize,ysize,center_x, lt_limit, rt_limit,st_line,end_line;
				int midpt,tmp,sl_size, os_factor,dim,offset;
				double midval,M,B;
				double inflect[], xpos[], sledge[],tempedge[],tdata[],rdata[];
				double tdata_plot[],rdata_plot[];
				double fcenter_x,pixsize;
				double regarr[], slope, intercept;
				
				Rectangle bounds=roi.getBounds();
				
				xsize=bounds.width;
				ysize=bounds.height;
				
				// 2D array of the pixels of the roi
				int pixels[][]=new int[ysize][xsize];
				
				os_factor=(int)IJ.getNumber("Oversample Factor: ", 2);
				pixsize=IJ.getNumber("Pixel Size [mm]: ",0.5618);
				dim=(int)IJ.getNumber("Length of FFT array: ", 512);
				
				// Get the pixels specified by the bounding box
				for (int i=0; i<ysize;i++)
				{
					for(int j=0; j<xsize;j++)
					{
						thispix=image.getPixel(j+bounds.x,i+bounds.y);
						pixels[i][j]=thispix[0];
					}
				}
				
				center_x=xsize/2;
				
				// Columns of the image to include in the calculation
				// (Don't start quite at the edges)
				lt_limit = center_x - xsize/4;
				rt_limit = center_x + xsize/4;

				// Rows of the image to include in the calculation
				st_line = 0;
				end_line = ysize;

				// Array that will contain the "column" of the point of inflection
				// for each row of the selected roi.  I say "column" in quotes because
				// it will, in general, not be an integer value.
				inflect=new double[ysize];
				
				// Array that will contain the row numbers of the selected roi
				// (st_line, st_line + 1, st_line + 2, ..., end_line).
				xpos=new double[ysize];
				
				// Iterate over each row of the roi
				for (int i=st_line; i<end_line; i++)
				{	   
					double lt_ave, rt_ave;
					int l;
					lt_ave = 0.0; rt_ave = 0.0;
					
					// Find the average intensity of the left and right sides of the row
					for (int j=lt_limit;j<(lt_limit+20);j++)
					{
						lt_ave += pixels[i][j];
					}
					
					for (int j=rt_limit-20;j<rt_limit;j++)
					{
						rt_ave += pixels[i][j];
					}
					lt_ave/=20.;
				    rt_ave/=20.;
				      
				      
				    if (lt_ave<rt_ave)
				    {
				    	// First, find an initial estimate of the midpoint (inflection point) location
				    	midval =  (rt_ave - lt_ave)/2 + lt_ave;
				    	l=lt_limit;
				    	while (pixels[i][l] < midval) l++;
				    	midpt = --l;

				    	// Now, fine-tune that estimate
				    	lt_ave =  rt_ave = 0.0;
				    	for (int k=midpt-50;k<midpt-30;k++)
				    	{
				    		lt_ave += pixels[i][k];
				    	}
				    	for (int k=midpt+30;k<midpt+50;k++)
				    	{
				    		rt_ave += pixels[i][k];
				    	}
				        lt_ave/=20.;
				        rt_ave/=20.;
				        midval =  (rt_ave - lt_ave) /2.+ lt_ave;
				        l=midpt-30;
				        while (pixels[i][l] < midval) l++;
				        midpt = --l;
				    }

				    else
				    {
				    	// First, find an initial estimate of the midpoint (inflection point) location
				    	midval =  (lt_ave - rt_ave)/2 + rt_ave;
				    	l=lt_limit;
				    	while (pixels[i][l] > midval) l++;
				    	midpt = --l;

				    	// Now, fine-tune that estimate
				    	lt_ave = rt_ave = 0.0;
				    	for (int k=midpt-50;k<midpt-30;k++)
				    	{
				    		lt_ave += pixels[i][k];
				    	}
				    	for (int k=midpt+30;k<midpt+50;k++)
				    	{
				    		rt_ave += pixels[i][k];
				    	}
				    	lt_ave/=20.;
				    	rt_ave/=20.;

				    	midval =  (lt_ave - rt_ave)/2. + rt_ave;
				    	l=midpt-30;
				    	while (pixels[i][l] > midval) l++;
				    	midpt = --l;
				      }
				    
				    // Now, the idea is to find the "column" of the calculated inflection point with sub-integer accuracy.
				    // (Again, I put "column" in quotes because of the sub-integer accuracy involved).
				    // The calculated inflection point has intensity 'midval' and is assumed to be on a line between
				    // column 'midpt' and 'midpt'+1.
				    M = pixels[i][midpt+1]-pixels[i][midpt];
				    B = pixels[i][midpt] - M*midpt;

				    // inflect[i] is the i'th row's "column" index for the midpoint (inflection point)
				    inflect[i] = (midval- B)/M;
				    xpos[i]=i;

				}
				
				// Perform linear regression on the midpoint location to find the
				// best-fit estimate of where the edge is within the roi
				regarr=regress(xpos,inflect,end_line-st_line);
				slope=regarr[0];
				intercept=regarr[1];
				
				// Find the "column" (with sub-integer accuracy) of 'midval' (inflection point) for the center row of the roi
				fcenter_x = slope*(st_line+(end_line+st_line)/2)+intercept;
	
				// Prepare to deal with some array index offset business later
				tmp = Math.max((int)(fcenter_x-xsize/2.),0);
				
				// Size of the over-sampled edge response
				sl_size = xsize*os_factor;
				
				// Scale variables appropriately according to how much we over-sample
				fcenter_x = fcenter_x*os_factor;
				pixsize/=os_factor;
	
				// 'sledge' will contain the over-sampled edge response function
				sledge=new double[dim];
				hit=new int[dim];
				  
				for (int j=0;j<dim;j++)
				{
					sledge[j] = 0.0;
					hit[j] = 0;
				}
	
				// The idea for the following 'for' loop is to upsample each row
				// and add a shifted version of it to 'sledge'.
				// The shift ('delta' below) is how far that specific row's
				// "column" (again, sub-pixel accuracy) index for the inflection point
				// is from that of the center row's 
				for (int i=st_line;i<end_line;i++)
				{
					int delta;
					tempedge=new double[sl_size];
					for (int j=0;j<sl_size;j++)
					{
						tempedge[j] = 0;
					}
					
					for (int j=0;j<sl_size;j+=os_factor)
					{
				         tempedge[j] = pixels[i][j/os_factor+tmp];
					}
				      
					delta = (int)(inflect[i] *os_factor - fcenter_x);
					if (delta > 0 )
						for (int j=0;j<sl_size-delta;j++)
				    	   {
								sledge[j] += tempedge[j+delta];
								if (tempedge[j+delta]!=0)  hit[j]++;
				    	   }
				       if (delta <= 0)
				    	   for (int j=sl_size-1;j>=-delta;j--)
				    	   {
				    		   sledge[j] += tempedge[j+delta];
				    		   if (tempedge[j+delta]!=0)  hit[j]++;
				            }
				   }
				   
				   for (int j=0;j<dim;j++)
			       {
					   if (hit[j] > 0) sledge[j]/=(double)hit[j];
			       }

				   
				   // Pad for the array for the FFT
				   
				   if (sl_size < dim )
				   {
				      int pad = (dim-sl_size)/2;
				      double lt_ave=0.0, rt_ave = 0.0;
				      for (int k=0;k<20;k++)
				         lt_ave += sledge[k];
				      for (int k=sl_size-20;k<sl_size;k++)
				         rt_ave += sledge[k];
				      lt_ave/=20.;
				      rt_ave/=20.;
				      for (int k=(dim-pad);k<dim;k++)
				         sledge[k] = rt_ave;
				      for (int k=(dim-pad-1);k>pad;k--)
				         sledge[k]= sledge[k-pad];
				      for (int k=0;k<=pad; k++)
				         sledge[k] = lt_ave;
				      sl_size = dim;
				   }
				   
				   offset = (sl_size - dim)/2;
				   tdata=new double[dim];
				   rdata=new double[dim+1];

				   // Differentiate the ERF to get the LSF
				   for (int j=0;j<dim-1;j++)
				   {
					   tdata[j+1] = sledge[j+1+offset] - sledge[j+offset];
				   }
				   tdata[0]=0;
				   
				   // Shift the LSF
				   // (I don't know why this is necessary)
				   for (int i=1;i<dim/2;i++)
					   rdata[i] = tdata[i+dim/2];
				   for (int i=dim/2+1;i<dim+1;i++)
					   rdata[i] = tdata[i-dim/2];

				   // 'tdata' and 'rdata' have some funky indexing, so we want to make
				   // versions of them that are plot-able (i.e., have good indexing)
				   tdata_plot=new double[dim-1];
				   rdata_plot=new double[dim];
				   
				   for(int i=0;i<dim-1;i++) tdata_plot[i]=tdata[i+1];
				   for(int i=0;i<dim;i++) rdata_plot[i]=rdata[i+1];

				   // Plot ERF and LSF
				   VisualizationUtil.createPlot("ERF", sledge).show();
				   VisualizationUtil.createPlot("LSF", tdata_plot).show();

				   double MTF1[]=FFTUtil.fft(rdata_plot);				   
				   // This is from VisualizationUtil.java
				   // I should really modify/add methods in that file
	
				   // Calculate magnitude of MTF
				   double [] absValues = new double [MTF1.length / 2];
				   for (int i = 0; i < absValues.length; i ++){
					   absValues[i] = FFTUtil.abs(i, MTF1);
				   }
				   
				   // Normalize MTF to have max value of 1
				   for(int i = absValues.length-1; i >= 0; i--){
					   absValues[i]=absValues[i]/absValues[0];
				   }
				   double yValues[]=absValues;
				   
				   double xValues[]=new double[yValues.length];

				   // Create the frequency axis, scaled appropriately
				   for (int i=0; i < xValues.length; i++) xValues[i]=i*(1.0/xValues.length)*(1.0/pixsize);

				   // Plot MTF
				   VisualizationUtil.createPlot(xValues, yValues, "MTF", "Frequency [mm^(-1)]", "").show();
		}
		return null;
	}

	public void configure() throws Exception {
		image = IJ.getImage();
		roi = image.getRoi();
		if (roi != null){
			configured = true;
		}
	}

	@Override
	public String toString() {
		return "Measure MTF using a slanted edge";
	}


}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
