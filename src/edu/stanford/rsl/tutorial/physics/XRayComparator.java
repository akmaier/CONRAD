/*
 * Copyright (C) 2014 - Andreas Maier, Tobias Miksch
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

package edu.stanford.rsl.tutorial.physics;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.UserUtil;
import ij.ImageJ;
import ij.ImagePlus;
import ij.process.ImageProcessor;

/**
 * @author Miksch Tobias
 *
 * Class that uses the functions of XRayAnalytics to analyze the results of my work.
 */

public class XRayComparator{

	private static String gt_path = "BoneCylinder_V0_pathtracing_4200M.tif";
	
	public static void compareSameGridValues(String title) {
		
		Grid2D grid = XRayAnalytics.openFile(title);
		XRayAnalytics.showGrid2D(grid, title, 0.0, 0.0);
		
		ImageProcessor image = ImageUtil.wrapGrid2D(grid);
		double min = image.getMin();
		double max = image.getMax();
		
		Grid2D result = XRayAnalytics.halfAddValue(grid,10 * 140000.f);
		XRayAnalytics.showGrid2D(result, "Added 10*eV to the left side!", min, max);
		
		Grid2D result2 = XRayAnalytics.divideValue(grid, 2);
		XRayAnalytics.showGrid2D(result2, "Only half the energy per pixel!", min, max);
		
		Grid2D result3 = XRayAnalytics.addValue(grid, 10 * 140000.f);
		XRayAnalytics.showGrid2D(result3, "Added 10*energyEV constant offset to the picture!", min, max);
		
		System.out.println("SSIM basic: " + XRayAnalytics.structuralSIM(grid, grid, max-min, 8, 2));
		
		System.out.println("\nGT_Image with only one side have additional light");
		System.out.println("Normal  RMSE: " + XRayAnalytics.computeRMSE(grid, result));
		System.out.println("Average RMSE: " + XRayAnalytics.averageRMSE(grid, result));
		System.out.println("SSIM: " + XRayAnalytics.structuralSIM(grid, result, max-min, 8, 2));
		
		System.out.println("\nGT_Image divided by 2");
		System.out.println("Normal  RMSE: " + XRayAnalytics.computeRMSE(grid, result2));
		System.out.println("Average RMSE: " + XRayAnalytics.averageRMSE(grid, result2));
		System.out.println("SSIM: " + XRayAnalytics.structuralSIM(grid, result2, max-min, 8, 2));
		
		System.out.println("\nGT_Image with constant offset 10*energyEV");
		System.out.println("Normal  RMSE: " + XRayAnalytics.computeRMSE(grid, result3));
		System.out.println("Average RMSE: " + XRayAnalytics.averageRMSE(grid, result3));
		System.out.println("SSIM: " + XRayAnalytics.structuralSIM(grid, result3, max-min, 8, 2));
	}
	
	public static void normalizeValuesOfTracing(String title, int rayNumber) {
		Grid2D grid = XRayAnalytics.openFile(title);
		
		Grid2D result2 = XRayAnalytics.divideValue(grid, rayNumber);
		XRayAnalytics.showGrid2D(result2, "All pixel values are divided by # of rays in millions!", 0.0, 0.0);
	}
	
	public static void testEnergyLevel(String title, int rayCount) {
		Grid2D grid = XRayAnalytics.openFile(title);
		
		double energySum = XRayAnalytics.totalEnergyCount(grid);
		System.out.println("EnergySum is: " + energySum / (rayCount * XRayTracer.mega));
	}
	
	
		
	public static void compareSSIM() {
		new ImageJ();
		
		String exampleFilePath = "GT_image/BoneCylinder_pt_rc_8624747115_dh_16419468_spt_9520286221_t_86400.0s.tif";
		Grid2D groundTruth = XRayAnalytics.openFile(exampleFilePath);
		System.out.println("Energy Sum of GT: " + XRayAnalytics.totalEnergyCount(groundTruth));
		groundTruth = XRayAnalytics.divideValue(groundTruth, 862.474711f); // 86.24747115 10^8 
		System.out.println("Energy Sum of GT after averageing by rays: " + XRayAnalytics.totalEnergyCount(groundTruth));
		System.out.println("GT image rmse: " + XRayAnalytics.computeRMSE(groundTruth, groundTruth));
		
		ImageProcessor gt = ImageUtil.wrapGrid2D(groundTruth);
		double min = gt.getMin();
		double max = gt.getMax();
		System.out.println("The min and max values of GT are: " + min + " | " + max);
		
		Grid2D images[] = new Grid2D[4];		
		images[0] = XRayAnalytics.openFile("Small_Test/BoneCylinder_pt_rc_60000000_dh_114187_spt_66229728_t_835.0s.tif");
		images[1] = XRayAnalytics.openFile("Small_Test/BoneCylinder_bdpt_rc_38461536_dh_3808624_spt_46266319_t_659.0s.tif");
		images[2] = XRayAnalytics.openFile("Small_Test/BoneCylinder_vpl_rc_18575848_dh_18575848_spt_55727544_t_1001.0s_80000.tif");
		images[3] = XRayAnalytics.openFile("Small_Test/BoneCylinder_vrl_rc_4854368_dh_4854368_spt_56026631_t_941.0s_80000.tif");
		
		for (int i = 0; i < images.length; i++) {
			System.out.println("Total energy of image " + i + " is: " + XRayAnalytics.totalEnergyCount(images[i]));
			
			ImageProcessor image = ImageUtil.wrapGrid2D(images[i]);
			double minImage = image.getMin();
			double maxImage = image.getMax();
//			System.out.println("The min and max values of " + i +  " are: " + minImage + " | " + maxImage);
		}
		
		images[0] = XRayAnalytics.divideValue(images[0], 6.0f); // 10^7
		images[1] = XRayAnalytics.divideValue(images[1], 3.846f);  // 10^7
		images[2] = XRayAnalytics.divideValue(images[2], 1.8576f);  // 10^7
		images[3] = XRayAnalytics.divideValue(images[3], 0.48543f); // 10^7
		
		System.out.println("After averaging the values:");
		for (int i = 0; i < images.length; i++) {
//			System.out.println("Total energy of image " + i + " is: " + XRayAnalytics.totalEnergyCount(images[i]));
			
			ImageProcessor image = ImageUtil.wrapGrid2D(images[i]);
			double minImage = image.getMin();
			double maxImage = image.getMax();
//			System.out.println("The min and max values of " + i +  " are: " + minImage + " | " + maxImage);
			
			if(min > minImage) min = minImage;
			if(max < maxImage) max = maxImage;
		}
		
		//Open grids with same settings.
		XRayAnalytics.showGrid2D(groundTruth, "BoneCylinder_GT_image", min, max);
		// Should we normalise the results by dividing trough the number of rays sent?!
		for (int i = 0; i < images.length; i++) {
			XRayAnalytics.showGrid2D(images[i], "BoneCylinder_version_" + i, min, max);
			
			double energySum = XRayAnalytics.totalEnergyCount(images[i]);
			
			double rmse = XRayAnalytics.computeRMSE(groundTruth, images[i]);
			double ssim2 = XRayAnalytics.structuralSIM(groundTruth, images[i], max-min, 8, 2);
			
			double ssim4 = XRayAnalytics.structuralSIM(groundTruth, images[i], max-min, 9, 5);
			
			System.out.println("TotalEnergy: " + energySum + " | rmse: " + rmse + " | ssim4: " + ssim4 + " | ssim2: " + ssim2);
		}
	}
	
	public static void main(String[] args) {
		compareSSIM();
		
/*		new ImageJ();
		Grid2D groundTruth = XRayAnalytics.openFile("Equal_Rays/BoneCylinder_vpl_rc_185758512_dh_185758512_spt_557275536_t_6139.0s_160000.tif");
		XRayAnalytics.showGrid2D(groundTruth, "indirect lighting", 0, 0);
		Grid2D directLighting = XRayAnalytics.openFile("GT_image/BoneCylinder_vpl_rc_9846400_dh_10000000_spt_0_t_113.0s_80.tif");
		XRayAnalytics.showGrid2D(directLighting, "directLighting", 0, 0);
		
		Grid2D fusion = XRayAnalytics.addGrid(groundTruth, directLighting);
		XRayAnalytics.showGrid2D(fusion, "fusion_absolut", 0, 0);
		
		groundTruth = XRayAnalytics.divideValue(groundTruth, 1857.58512f);
		//directLighting = XRayAnalytics.divideValue(directLighting, 98.46400f);
		
		Grid2D fusion2 = XRayAnalytics.addGrid(groundTruth, directLighting);
		XRayAnalytics.showGrid2D(fusion2, "fusion_average", 0, 0);
*/
	}
}
