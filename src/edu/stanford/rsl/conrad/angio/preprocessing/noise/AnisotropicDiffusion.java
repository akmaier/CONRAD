package edu.stanford.rsl.conrad.angio.preprocessing.noise;


/* importing standard Java API Files and ImageJ packages */
import ij.*;
import ij.process.*;

import java.awt.*;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.utils.ImageUtil;

public class AnisotropicDiffusion{
	// the following are the input parameters, with default values assigned to them
	int nb_iter       = 20;    // Number of iterations
	int nb_smoothings = 1;     // Number of smoothings per iteration
	double dt         = 20.0;  // Adapting time step
	float a1          = 0.5f;  // Diffusion limiter along minimal variations
	float a2          = 0.9f;  // Diffusion limiter along maximal variations
	float edgeheight  = 5f;     // edge threshold
		 
	
	public static void main(String[] args) {
		String dir = ".../";
		String file = "test.tif";
		Grid3D img = ImageUtil.wrapImagePlus(IJ.openImage(dir+file));
		
		AnisotropicDiffusion aniso = new AnisotropicDiffusion();
		aniso.nb_iter = 10;
		aniso.nb_smoothings = 1;
		aniso.dt = 20;
		aniso.a1 = 0.5f;
		aniso.a2 = 0.9f;
		aniso.edgeheight = 5f;
		
		Grid2D filtered = aniso.run(img.getSubGrid(0));
		
		new ImageJ();
		filtered.show();
		
	}
	
	
	//-----------------------------------------------------------------------------------
	public Grid2D run(Grid2D slice) 
	{
		ImageProcessor ip = ImageUtil.wrapGrid2D(slice);
		// variables and arrays
		int channels = ip instanceof ColorProcessor ? 3 : 1;
		Rectangle r = ip.getRoi();
		int width = r.width;
		int height = r.height;
		int totalwidth = ip.getWidth();
		int totalheight = ip.getHeight();
		ImageProcessor ip2 = null;
		float[][][][] grad = new float[2][width][height][channels];
        float[][][] G = new float[width][height][3]; // must be clean for each slice
		float[][][] T = new float[width][height][3];
		float[][][] veloc = new float[width][height][3];
		float val1, val2;
		float vec1, vec2;
		double xdt;
		float[][][] ipf  = new float[width][height][channels];
		float[] pixel = new float[channels];
		float fx, fy;
		double ipfnew;
		boolean breaked = false;
		int iter = 0;
		double average, stddev, drange, drange2, drange0;
		
		float ipMax = -Float.MAX_VALUE;
		for (int x=width; x-->0;){
			for (int y=height; y-->0;){
				ipMax = Math.max(ipMax, ip.getPixelValue(x, y));
			}
		}
		float intensityTransform = 256f/ipMax;
		
		// consts
		final float c1 = (float)(0.25*(2-Math.sqrt(2.0))), c2 = (float)(0.5f*(Math.sqrt(2.0)-1));

		// convert image into the float channels
		for (int x=width; x-->0;)
			for (int y=height; y-->0;) 
			{
				pixel[0] = ip.getPixelValue(x+r.x, y+r.y);
				ipf[x][y][0] = pixel[0]*intensityTransform;                        
			}

		// get initial stats for later normalizing
		average = 0;
		float initial_max=ipf[0][0][0], initial_min=ipf[0][0][0], pix;
		for (int x=width; x-->0;)
			for (int y=height; y-->0;)
				for (int k=channels; k-->0;) 
				{
					pix = ipf[x][y][k];
					if (pix>initial_max) initial_max=pix;
					if (pix<initial_min) initial_min=pix;
					average += pix;
				}
		average /= (width*height*channels);
		// standard deviation
		stddev = 0;
        for (int x=width; x-->0;)
			for (int y=height; y-->0;)
				for (int k=channels; k-->0;) 
				{
					pix = ipf[x][y][k];
					stddev += (pix-average)*(pix-average);
				}
		stddev = Math.sqrt(stddev/(width*height*channels));
        
		//version 0.3 normalization
		drange= (edgeheight*stddev)/(initial_max-initial_min);
		drange0= (6*stddev)/(initial_max-initial_min);
		drange2= drange * drange;

		// PDE main iteration loop
		for (iter=0; (iter < nb_iter) && (!breaked); iter++)
		{
			System.out.println("On iteration "+String.valueOf(iter+1)+" of "+String.valueOf(nb_iter));
			// compute gradients
			float Ipp,Icp,Inp=0,Ipc,Icc,Inc=0,Ipn,Icn,Inn=0;
			// the following seems several times faster
			for (int x=width; x-->0;) 
			{
				int px=x-1; if (px<0) px=0;
				int nx=x+1; if (nx==width) nx--;
				for (int y=height; y-->0;) 
				{
					int py=y-1; if (py<0) py=0;
					int ny=y+1; if (ny==height) ny--;
					for (int k=channels; k-->0;) 
					{
						Ipp=ipf[px][py][k];
						Ipc=ipf[px][y] [k];
						Ipn=ipf[px][ny][k];
						Icp=ipf[x] [py][k];
						Icn=ipf[x] [ny][k];
						Inp=ipf[nx][py][k];
						Inc=ipf[nx][y] [k];
						Inn=ipf[nx][ny][k];
						float IppInn = c1*(Inn-Ipp);
						float IpnInp = c1*(Ipn-Inp);
						grad[0][x][y][k] = (float)(IppInn-IpnInp-c2*Ipc+c2*Inc);
						grad[1][x][y][k] = (float)(IppInn+IpnInp-c2*Icp+c2*Icn);
					}
				}
			}

			// compute structure tensor field G
//					G = new float[width][height][3]; // must be clean for each slice
			for (int x=width; x-->0;)
				for (int y=height; y-->0;) {
					G[x][y][0]= 0.0f;
					G[x][y][1]= 0.0f;
					G[x][y][2]= 0.0f;
					for (int k=channels; k-->0;) 
					{
					//version 0.2 normalization
						fx = grad[0][x][y][k];
						fy = grad[1][x][y][k];
						G[x][y][0] += fx*fx;
						G[x][y][1] += fx*fy;
						G[x][y][2] += fy*fy;
					}
				}

			// compute the tensor field T, used to drive the diffusion
			for (int x=width; x-->0;){
				for (int y=height; y-->0;) 
				{
					// eigenvalues:
					double a = G[x][y][0], b = G[x][y][1], c = G[x][y][1], d = G[x][y][2], e = a+d;
					double f = Math.sqrt(e*e-4*(a*d-b*c));
					double l1 = 0.5*(e-f), l2 = 0.5*(e+f);
					// more precise computing of quadratic equation
						if (e>0) { if (l1!=0) l2 = (a*d - b*c)/l1; }
						else     { if (l2!=0) l1 = (a*d - b*c)/l2; }

   
						val1=(float)(l2 / drange2);
						val2=(float)(l1 / drange2);
						// slight cheat speedup for default a1 value
					float f1 = (a1==.5) ? (float)(1/Math.sqrt(1.0f+val1+val2)) : (float)(Math.pow(1.0f+val1+val2,-a1));
					float f2 = (float)(Math.pow(1.0f+val1+val2,-a2));

					// eigenvectors:
					double u, v, n;
					if (Math.abs(b)>Math.abs(a-l1)) { u = 1; v = (l1-a)/b; }
					else { if (a-l1!=0) { u = -b/(a-l1); v = 1; } 
						   else { u = 1; v = 0; } 
					}
					n = Math.sqrt(u*u+v*v); u/=n; v/=n; 
					vec1 = (float)u; vec2 = (float)v;
					float vec11 = vec1*vec1, vec12 = vec1*vec2, vec22 = vec2*vec2;
					T[x][y][0] = f1*vec11 + f2*vec22;
					T[x][y][1] = (f1-f2)*vec12;
					T[x][y][2] = f1*vec22 + f2*vec11;
				}
			}
			
			// multiple smoothings per iteration
			for(int sit=0; sit < nb_smoothings && !breaked; sit++)
			{
                // compute the PDE velocity and update the iterated image
				Inp=Inc=Inn=0;
				// the following seems several times faster
				for (int x=width; x-->0;) 
				{
					int px=x-1; if (px<0) px=0;
					int nx=x+1; if (nx==width) nx--;
					for (int y=height; y-->0;) 
					{
						int py=y-1; if (py<0) py=0;
						int ny=y+1; if (ny==height) ny--;
						for (int k=channels; k-->0;) 
						{
							Ipp=ipf[px][py][k];
							Ipc=ipf[px][y] [k];
							Ipn=ipf[px][ny][k];
							Icp=ipf[x] [py][k];
							Icc=ipf[x] [y] [k];
							Icn=ipf[x] [ny][k];
							Inp=ipf[nx][py][k];
							Inc=ipf[nx][y] [k];
							Inn=ipf[nx][ny][k];
							float ixx = Inc+Ipc-2*Icc,
								iyy = Icn+Icp-2*Icc,
								ixy = 0.5f*(Ipp+Inn-Ipn-Inp);
							veloc[x][y][k] = T[x][y][0]*ixx + T[x][y][1]*ixy + T[x][y][2]*iyy; 
						}
					}
				}
				// find xdt coefficient
				if (dt>0) 
				{
					float max=veloc[0][0][0], min=veloc[0][0][0];
					for (int x=width; x-->0;)
						for (int y=height; y-->0;)
							for (int k=channels; k-->0;) 
							{
								if (veloc[x][y][k]>max) max=veloc[x][y][k];
								if (veloc[x][y][k]<min) min=veloc[x][y][k];
							}
					//version 0.2 normalization
					xdt = dt/Math.max(Math.abs(max), Math.abs(min))*drange0;
				} 
				else xdt = -dt;
				
				// update image
				for (int x=width; x-->0;)
					for (int y=height; y-->0;)
						for (int k=channels; k-->0;) 
						{
							ipfnew = ipf[x][y][k] + veloc[x][y][k]*xdt;
							ipf[x][y][k] = (float)ipfnew;
							// normalize image to the original range
							if (ipf[x][y][k] < initial_min) ipf[x][y][k] = initial_min;
							if (ipf[x][y][k] > initial_max) ipf[x][y][k] = initial_max;
						}

			} // smoothings per iteration
		}
		ip2 = ip.createProcessor(totalwidth, totalheight);
		for (int x=totalwidth; x-->0;){
			for (int y=totalheight; y-->0;){
				for (int k=channels; k-->0;) 
				{
					if ((x<r.x) || (x>=r.x+width) || (y<r.y) || (y>=r.y+height))
						ip2.putPixelValue(x,y,ip.getPixel(x,y)/intensityTransform);
					else 
					{
						pixel[k] = (int)ipf[x-r.x][y-r.y][k];
						ip2.putPixelValue(x,y,pixel[k]/intensityTransform);
					}
				}
			}
		}
		Grid2D filtered = ImageUtil.wrapImageProcessor(ip2);
		filtered.setSpacing(slice.getSpacing());
		return filtered;
	} // end of 'runTD' method
	
}
