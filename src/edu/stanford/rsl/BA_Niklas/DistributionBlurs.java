package edu.stanford.rsl.BA_Niklas;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;


public class DistributionBlurs {
	public static Grid2D img;
	
	public static void main(String args[])
	{
//		readImage();
		blur(10, 49);
//		writeImage();
	}
	
//	public static void readImage()
//	{
//		DistributionBlurs.img = null;
//		try {
//			img = ImageIO.read(new File("Poro.png"));
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
//	}
	
	public static void blur(double sigma, int kernelsize)
	{
		double[] kernel = createKernel(sigma, kernelsize);
		for(int i = 0; i < img.getWidth(); i++)
		{
			for(int j = 0; j < img.getHeight(); j++)
			{
				double overflow = 0;
				int counter = 0;
				int kernelhalf = (kernelsize - 1)/2;
				double value = 0;
				for(int k = i - kernelhalf; k < i + kernelhalf; k++)
				{
					for(int l = j - kernelhalf; l < j + kernelhalf; l++)
					{
						if(k < 0 || k >= img.getWidth() || l < 0 || l >= img.getHeight())
						{
							counter++;
							overflow += kernel[counter];
							continue;
						}
						
						double val = img.getAtIndex(k, l);
						value += val * kernel[counter];
						
						counter++;
					}
					counter++;
				}
				
				if(overflow > 0)
				{
					value = 0;
					
					counter = 0;
					for(int k = i - kernelhalf; k < i + kernelhalf; k++)
					{
						for(int l = j - kernelhalf; l < j + kernelhalf; l++)
						{
							if(k < 0 || k >= img.getWidth() || l < 0 || l >= img.getHeight())
							{
								counter++;
								continue;
							}
							
							double val = img.getAtIndex(k, l);
							value += val * kernel[counter]*(1/(1-overflow));
							
							counter++;
						}
						counter++;
					}
				}
				
//				DistributionBlurs.img.setRGB(i, j, new Color((int)red, (int)green, (int)blue).getRGB());
				DistributionBlurs.img.setAtIndex(i, j, (float) value);
			}
		}
	}
	
	public static double[] createKernel(double sigma, int kernelsize)
	{
		double[] kernel = new double[kernelsize*kernelsize];
		for(int i = 0;  i < kernelsize; i++)
		{
			double x = i - (kernelsize -1) / 2;
			for(int j = 0; j < kernelsize; j++)
			{
				double y = j - (kernelsize -1)/2;
				kernel[j + i*kernelsize] = 1 / (2 * Math.PI * sigma * sigma) * Math.exp(-(x*x + y*y) / (2 * sigma *sigma));
			}
		}
		float sum = 0;
		for(int i = 0; i < kernelsize; i++)
		{
			for(int j = 0; j < kernelsize; j++)
			{
				sum += kernel[j + i*kernelsize];
			}
		}
		for(int i = 0; i < kernelsize; i++)
		{
			for(int j = 0; j < kernelsize; j++)
			{
				kernel[j + i*kernelsize] /= sum;
			}
		}
		return kernel;
	}
}
	
//	public static void lineareAbbildung(double a, double b)
//	{
//		for(int x = 0; x < img.getWidth(); x++)
//		{
//			for(int y = 0; y < img.getHeight(); y++)
//			{
//				int rgb = (int) (a*DistributionBlurs.img.getRGB(x, y) + b);
//				DistributionBlurs.img.setRGB(x, y, rgb);
//			}
//		}
//	}
//	
//	public static void blackAndWhite()
//	{
//		for(int x = 0; x < img.getWidth(); x++)
//		{
//			for(int y = 0; y < img.getHeight(); y++)
//			{
//				int rgb = DistributionBlurs.img.getRGB(x, y);
//				Color c = new Color(rgb);
//				int grey = (int) (0.299 * c.getRed() + 0.587 * c.getGreen() + 0.114*c.getBlue());
//				Color c2 = new Color(grey, grey, grey);
//				DistributionBlurs.img.setRGB(x, y, c2.getRGB());
//			}
//		}
//	}
//	
//	public static void improveGrey()
//	{
//		int min = Integer.MAX_VALUE;
//		int max = Integer.MIN_VALUE;
//		for(int x = 0; x < img.getWidth(); x++)
//		{
//			for(int y = 0; y < img.getHeight(); y++)
//			{
//				int rgb = DistributionBlurs.img.getRGB(x, y);
//				if(rgb > max)
//				{
//					max = rgb;
//				}
//				if(rgb < min)
//				{
//					min = rgb; 
//				}
//			}
//		}
//		lineareAbbildung(255 / (max - min), - 255* min / (max -min));
//	}
//	
//	public static void writeImage()
//	{
//		File output = new File("PoroChanged.png");
//		try {
//			ImageIO.write(img, "png", output);
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
//	}
//}