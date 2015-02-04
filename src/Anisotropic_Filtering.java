

import javax.swing.SwingUtilities;

import edu.stanford.rsl.conrad.filtering.multiprojection.anisotropic.AnisotropicFilterFunction;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.UserUtil;
import edu.stanford.rsl.conrad.volume3d.JTransformsFFTVolumeHandle;
import edu.stanford.rsl.conrad.volume3d.ParallelVolumeOperator;
import edu.stanford.rsl.conrad.volume3d.Volume3D;
import ij.IJ;
import ij.ImagePlus;
import ij.plugin.PlugIn;


/**
 * Filters the projection data with a 2-D filter which is selected via the ImageJ GUI.
 * 
 * @author Andreas Maier
 *
 */
public class Anisotropic_Filtering implements PlugIn {

	public Anisotropic_Filtering(){
		
	}
	
	@Override
	public void run(String arg) {
		// We don't want ImageJ to run this Thread.
		SwingUtilities.invokeLater(new Runnable() {
			public void run(){
				try{
					Configuration.loadConfiguration();
					ImagePlus current = IJ.getImage();	
					boolean uneven = current.getStackSize() % 2 == 1;
					int context = UserUtil.queryInt("Context Size: ", 16);
					float smoothness = (float) UserUtil.queryDouble("Smoothness: ", 2.0);
					float lowerTensorLevel = (float) UserUtil.queryDouble("Lower limit in Tensor: ", 0.0);
					float upperTensorLevel = (float) UserUtil.queryDouble("Upper limit in Tensor: ", 1.0);
					float highPassLowerLevel = (float) UserUtil.queryDouble("Lower limit in high pass sigmoid: ", 0.5);
					float highPassUpperLevel = (float) UserUtil.queryDouble("Upper limit in high pass sigmoid: ", 1.5);
					float lpUpper = (float) UserUtil.queryDouble("Strengh of low pass filter: ", 1.5f);
					int A = (int) UserUtil.queryDouble("Exponent: ", 1);
					float B = (float) UserUtil.queryDouble("B: ", 2f);
					float ri = (float) UserUtil.queryDouble("ri: ", 1.5f);
					float a = (float) UserUtil.queryDouble("a: ", 1.0f);
					boolean showAbsoluteTensor = UserUtil.queryBoolean("Display Magnitude Image of Tensor?");
					int margin = context;
					if (margin%2==1) margin ++;
					Volume3D vol = new Volume3D(current, margin, 3,uneven);
					AnisotropicFilterFunction filter = new AnisotropicFilterFunction(new JTransformsFFTVolumeHandle(new ParallelVolumeOperator()), new ParallelVolumeOperator());
					System.out.println("Start: " + System.currentTimeMillis());
					Volume3D [] filtered = filter.computeAnisotropicFilteredVolume(vol, lowerTensorLevel, upperTensorLevel, highPassLowerLevel, highPassUpperLevel, smoothness, A, B, ri, a, lpUpper);
					filtered[0].getImagePlus("Anisotropic Filtered " + current.getTitle(), margin, 3,uneven).show();
					if (showAbsoluteTensor) {
						filtered[1].getImagePlus("Tensor Norm " + current.getTitle(), margin, 3,uneven).show();
					}
					filter = null;
					vol = null;
					CONRAD.gc();
				} catch (Exception e){
					e.printStackTrace();
				}
				System.out.println("End: " + System.currentTimeMillis());
			}
		});

	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
