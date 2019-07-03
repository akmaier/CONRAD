package edu.stanford.rsl.BA_Niklas;

import edu.stanford.rsl.BA_Niklas.Regression;
import edu.stanford.rsl.BA_Niklas.PolynomialRegression;
import edu.stanford.rsl.BA_Niklas.ListInFile;
import com.zetcode.linechartex.ScatterPlot;
import edu.stanford.rsl.conrad.numerics.DecompositionSVD;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.BA_Niklas.PhaseContrastImages;
import edu.stanford.rsl.BA_Niklas.ProjectorAndBackprojector;
import edu.stanford.rsl.apps.Conrad;
import edu.stanford.rsl.conrad.data.numeric.*;
import ij.ImageJ;
import java.util.Random;

import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang.builder.ReflectionToStringBuilder;

import java.util.*; 

/**
 * Bachelor thesis of Niklas Bubeck
 * 
 * This class simulates data, forward-projects it and does a filtered-backprojection.
 * 
 * @author Lina Felsner
 *
 */
public class Bubeck_Niklas_BA {
	
	static int size = 128;
	static int nr_ellipses = 20;
    static SimpleMatrix[] R; 	// rotation matrices. 
    static double[][] t; 	// displacement vectors.   
    static double[][] ab; 	// the length of the principal axes.    
    static double[] rho; 	// signal intensity.
    static double[] df; 	// dark-field signal intensity.
	static int detector_width = 200;
	static int nr_of_projections = 360;
    
    /**
     * creates random ellipses
     */
    public static void create_ellipse() {
        R = new SimpleMatrix[nr_ellipses];    
        t = new double[nr_ellipses][2];         
        ab = new double[nr_ellipses][2]; 
        rho = new double[nr_ellipses]; 
        df = new double[nr_ellipses]; 
        
        for(int i=0; i<nr_ellipses; i++){
            
        	// translation between -0.5 and 0.5
            t[i][0] = Math.random() - 0.5; // delta_x
            t[i][1] = Math.random() - 0.5; // delta_y
            
            // size between 0 and 0.75
            ab[i][0] = Math.random() * 0.75; // a
            ab[i][1] = Math.random() * 0.75; // b
            
            // rotation
            double alpha = Math.random()*90;
            R[i] = new SimpleMatrix(new double[][]{{Math.cos(alpha), -Math.sin(alpha)}, {Math.sin(alpha), Math.cos(alpha)}});		
             
            // values
            rho[i] = Math.random();
            df[i] = Math.random();
            
        }
    }        

    /**
     * computes the absorption and dark-field signal at position {x,y}
     * @param x - position x-direction
     * @param y - position y-direction
     * @return array od absorption and dark-field value
     */
    public static double[] getSignal(double x, double y){
    
        double[] r = {x,y};
        double[] p = new double[2];        
        double sum = 0.0;
       
        double signal_absorption = 0.0;  
        double signal_dark_field = 0.0;       

        
        for(int i=0; i<nr_ellipses; i++){ // loop through each of the ellipsoids
        	SimpleVector pos = new SimpleVector(new double[] {r[0]-t[i][0],r[1]-t[i][1]});
            p = SimpleOperators.multiply(R[i], pos).copyAsDoubleArray();    
            sum = Math.pow(p[0]/ab[i][0],2) + Math.pow(p[1]/ab[i][1],2);        
            signal_absorption += (sum<=1.0)?rho[i]:0;   
            signal_dark_field += (sum<=1.0)?df[i]:0;      
        }
  
        return new double[] {signal_absorption, signal_dark_field};
    }
	
    /**
     * simulates ellipses
     * @return Phase contrast images containing absorption (phase) and dark-field phantoms
     */
	public static PhaseContrastImages simulate_data() {
	
		Grid2D amplitude = new Grid2D(size, size);
		Grid2D phase = new Grid2D(size,size);
		Grid2D dark_field = new Grid2D(size, size);
		
		create_ellipse();
			
		for (int i=0; i< size; i++) {
			for (int j = 0; j< size; j++){
				double[] value = getSignal(((float)((size/2) - i))/(size/2),	((float)((size/2) - j))/(size/2));
				amplitude.setAtIndex(i, j, (float) value[0]);
				dark_field.setAtIndex(i, j, (float) value[1]);
			}
		}
			
		PhaseContrastImages pci = new PhaseContrastImages(amplitude, phase, dark_field);
		return pci;
	}
	
    /**
     * simulates truncation
     * @param pci_sino - sinograms 
     * @param min - minimum on coordinate-axes 
     * @param max - maximum on coordinate-axes
     * @param axes - determines the used axe 
     * @param value - determines the value that will be set in range of min - max on Axes - axe
     * @param random - generates a random value that will be set. @param value will be overwritten
     * @param filter - filters the image with given filter (gauss, sobel, ....)
     * @return  Phase contrast images containing absorption (phase) and dark-field phantoms with artifacts
     */
//	TODO: filters
	public static PhaseContrastImages fake_truncation(PhaseContrastImages pci_sino, int min, int max, String axes, int value, boolean random, String filter){
		int width = pci_sino.getWidth();
		int height = pci_sino.getHeight();
		Grid2D fake_amp = (Grid2D) pci_sino.getAmp();
		Grid2D fake_dark = (Grid2D) pci_sino.getDark();
		Grid2D fake_phase = (Grid2D) pci_sino.getPhase();
		for (int i= 0; i< width; i++){
			for(int j =0; j < height; j++){
				if (axes == "x"){
					if (i > min && i < max){
						if (random == true){
							value = (int) (Math.random() * 255);
						}
						
						fake_amp.setAtIndex(i, j, value);
						fake_dark.setAtIndex(i, j, value);
						fake_phase.setAtIndex(i, j, value);
					}
				}else if (axes == "y"){
					if (j > min && j < max){
						if (random == true){
							value = (int) (Math.random() * 255);
						}
						fake_amp.setAtIndex(i, j, value);
						fake_dark.setAtIndex(i, j, value);
						fake_phase.setAtIndex(i, j, value);
					}
				}
				
			}
		}
		
		NumericGridOperator test = new NumericGridOperator();
		
		PhaseContrastImages pci_sino_fake = new PhaseContrastImages(fake_amp,  fake_phase, fake_dark);
		return pci_sino_fake;
	}
	

	public static NumericGrid initial_estimation(NumericGrid absorption, NumericGrid dark){
		NumericGrid initial = NumericPointwiseOperators.subtractedBy(absorption, dark);
//		NumericPointwiseOperators.abs(initial);
		
		return initial;
		
	}
	
	public static NumericGrid thresholding_map(NumericGrid fabso, NumericGrid dark, float thresh){
		Grid2D thresh_map = new Grid2D(size, size);
		for (int i = 0; i < size; i++){
			for(int j = 0; j < size; j++){
				int idx[] = {i, j};
				if(fabso.getValue(idx) < (thresh * NumericPointwiseOperators.max(dark))){
					thresh_map.setAtIndex(i, j, 1);
				}else{
					thresh_map.setAtIndex(i, j, 0);	
				}
				
			}
		}
		
		return thresh_map;
	}
	
	public static NumericGrid refinement(NumericGrid fabso, NumericGrid thresh_map){
		NumericPointwiseOperators.multipliedBy(fabso, thresh_map);
		return fabso ;
	}
	
	public static PhaseContrastImages region_of_interest(PhaseContrastImages sino_original, PhaseContrastImages sino_roi, int iter_num){
		ProjectorAndBackprojector p = new ProjectorAndBackprojector(360, 2*Math.PI);
		Grid2D roi_amp = new Grid2D(size, size);
		Grid2D roi_phase = new Grid2D(size, size);
		Grid2D roi_dark = new Grid2D(size, size);
		PhaseContrastImages pci_recon = new PhaseContrastImages(roi_amp,  roi_phase, roi_dark);
		
		NumericPointwiseOperators.subtractBy(sino_original.getAmp(), sino_roi.getAmp());
		NumericPointwiseOperators.subtractBy(sino_original.getPhase(), sino_roi.getPhase());
		NumericPointwiseOperators.subtractBy(sino_original.getDark(), sino_roi.getDark());
//		sino_original.show("sino_original");
			
		PhaseContrastImages pci_reko = p.filtered_backprojection(sino_original, size);
		return pci_reko;
	}
	

	
	/**
	 * 
	 * @param chars
	 * @return the max value in the array of chars
	 */
//	private static double maxValue(double[] chars) {
//	    double max = chars[0];
//	    for (int ktr = 0; ktr < chars.length; ktr++) {
//	        if (chars[ktr] > max) {
//	            max = chars[ktr];
//	        }
//	    }
//	    return max;
//	}

	


	public static double[][] get_comp_points(NumericGrid abso, NumericGrid dark){
	    List<Float> abslist = new ArrayList<Float>();
	    List<Float> darklist = new ArrayList<Float>();

		Grid2D abs = (Grid2D) abso;
		Grid2D dar = (Grid2D) dark;
		
		for(int i = 0; i < size; i++){
			for(int j = 0; j < size; j++){
				
				abslist.add(abs.getAtIndex(i, j));
				darklist.add(dar.getAtIndex(i, j));
			}
		}
		ListInFile.export(darklist, "C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Files/dark.csv");
		ListInFile.export(abslist, "C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Files/abso.csv");

//		while(abslist.remove("0.0")) {}
//		while(darklist.remove("0.0")) {}
		
		double[][] points = new double[abslist.size()][2];
	    for (int i = 0; i < points.length; i++) {
	        points[i][0] = darklist.get(i);
	        points[i][1] = abslist.get(i);
	    }

		return points;
	}
	
	
	public static PhaseContrastImages iterative_reconstruction(PhaseContrastImages pci_sino, int iter_num, int error){		
		
		// Build picture with ones		
		Grid2D ones_amp = new Grid2D(size, size);
		Grid2D ones_phase = new Grid2D(size, size);
		Grid2D ones_dark = new Grid2D(size, size);
		
		for(int i = 0; i < size; i++){
			for(int j = 0; j < size; j++){
				ones_amp.setAtIndex(i, j, 1);
				ones_phase.addAtIndex(i, j, 1);
				ones_dark.setAtIndex(i, j, 1);
			}
		}
		
		// forward + back projection 
		PhaseContrastImages ones = new PhaseContrastImages(ones_amp, ones_phase, ones_dark);
//		ones.show("ones");
		ProjectorAndBackprojector o = new ProjectorAndBackprojector(360, 2*Math.PI);
		PhaseContrastImages sino_ones = o.project(ones, new Grid2D(200, 360));
//		sino_ones.show("sino_ones");
		PhaseContrastImages ones_reko = o.backprojection_pixel(sino_ones, size);
//		ones_reko.show("ones_reko");
		
		// Build empty picture 
		Grid2D recon_amp = new Grid2D(size, size);
		Grid2D recon_phase = new Grid2D(size,size);
		Grid2D recon_dark = new Grid2D(size, size);
		PhaseContrastImages pci_recon = new PhaseContrastImages(recon_amp,  recon_phase, recon_dark);
//		pci_recon.show("old pci_recon");
		
		//iteration
		int i = 1;
		while( i <= iter_num || error == 5){
			
			System.out.println("Iteration Nr: "+ i);
			
			//Project picture 
			ProjectorAndBackprojector p = new ProjectorAndBackprojector(360, 2*Math.PI); 
			PhaseContrastImages sino_recon = p.project(pci_recon, new Grid2D(200, 360));
//			sino_recon.show("sino_recon");

			//Build difference of recon_sino and given sino
			NumericPointwiseOperators.subtractBy(sino_recon.getAmp(), pci_sino.getAmp());
			NumericPointwiseOperators.subtractBy(sino_recon.getPhase(), pci_sino.getPhase());
			NumericPointwiseOperators.subtractBy(sino_recon.getDark(), pci_sino.getDark());
//			sino_recon.show("sino_difference");
			
			//calc absolute
//			NumericPointwiseOperators.abs(sino_recon.getAmp());
//			NumericPointwiseOperators.abs(sino_recon.getPhase());
//			NumericPointwiseOperators.abs(sino_recon.getDark());
//			sino_recon.show("abs");

			//Todo: refinement and the other stuff ...
			
				
			// Backproject to reconstruct 
			PhaseContrastImages pci_reko = p.backprojection_pixel(sino_recon, size);
//			pci_reko.show("backprojected");
			
			// scale/normalise
			NumericPointwiseOperators.divideBy(pci_reko.getAmp(), ones_reko.getAmp());
			NumericPointwiseOperators.divideBy(pci_reko.getPhase(), ones_reko.getPhase());
			NumericPointwiseOperators.divideBy(pci_reko.getDark(), ones_reko.getDark());
//			pci_recon.show("scaled");
			
			// scale with alpha
			double alpha = 0.25;
			NumericPointwiseOperators.multiplyBy(pci_reko.getAmp(), (float) alpha);
			NumericPointwiseOperators.multiplyBy(pci_reko.getPhase(), (float) alpha);
			NumericPointwiseOperators.multiplyBy(pci_reko.getDark(), (float) alpha);
			
			
			//subtrackt on picture
			NumericPointwiseOperators.subtractBy(pci_recon.getAmp(), pci_reko.getAmp());
			NumericPointwiseOperators.subtractBy(pci_recon.getPhase(), pci_reko.getPhase());
			NumericPointwiseOperators.subtractBy(pci_recon.getDark(), pci_reko.getDark());
//			pci_recon.show("recon");
			
			
			// Adding on picture
//			NumericPointwiseOperators.addBy(pci_recon.getAmp(), pci_reko.getAmp());
//			NumericPointwiseOperators.addBy(pci_recon.getPhase(), pci_reko.getPhase());
//			NumericPointwiseOperators.addBy(pci_recon.getDark(), pci_reko.getDark());
//			pci_recon.show("recon");
			if(i == 1 || i == iter_num){
				pci_recon.show("recon");
			}
			i++;
		}
		

		System.out.println("iterative reconstruction done");
		return pci_recon;
	}

	
	
//	public static PhaseContrastImages iterative_landweber(PhaseContrastImages pci_sino, int iter_num, int error){		
//		
//		// Build empty picture 
//		Grid2D recon_amp = new Grid2D(size, size);
//		Grid2D recon_phase = new Grid2D(size,size);
//		Grid2D recon_dark = new Grid2D(size, size);
//		PhaseContrastImages pci_recon = new PhaseContrastImages(recon_amp,  recon_phase, recon_dark);
//		
////		pci_recon.show("old pci_recon");
	
//	//transpose sino matrix
//	Grid2D transpose_amp = new Grid2D(pci_sino.getWidth(), pci_sino.getHeight());
//	Grid2D transpose_phase = new Grid2D(pci_sino.getWidth(), pci_sino.getHeight());
//	Grid2D transpose_dark = new Grid2D(pci_sino.getWidth(), pci_sino.getHeight());
//	
//	Grid2D original_amp = (Grid2D) pci_sino.getAmp();
//	Grid2D original_dark = (Grid2D) pci_sino.getDark();
//	Grid2D original_phase = (Grid2D) pci_sino.getPhase();
//	
//	for(int i = 0; i<pci_sino.getWidth(); i++){
//		for(int j = 0; j < pci_sino.getHeight(); j++){
//			transpose_amp.setAtIndex(i, j, original_amp.getAtIndex(j, i));
//			transpose_phase.setAtIndex(i, j, original_phase.getAtIndex(j, i));
//			transpose_dark.setAtIndex(i, j, original_dark.getAtIndex(j, i));
//		}
//	}
//	PhaseContrastImages pci_transpose = new PhaseContrastImages(transpose_amp,  transpose_phase, transpose_dark);
//	
//	// AtA
//	
//	NumericPointwiseOperators.multiplyBy(pci_transpose.getAmp(), pci_sino.getAmp());
//	NumericPointwiseOperators.multiplyBy(pci_transpose.getPhase(), pci_sino.getPhase());
//	NumericPointwiseOperators.multiplyBy(pci_transpose.getDark(), pci_sino.getDark());
//	float max_amp = NumericPointwiseOperators.max(pci_transpose.getAmp());
//	pci_transpose.show("pci_AtA");
//	
//	Grid2D ata_amp_grid = (Grid2D) pci_transpose.getAmp();
//	Grid2D ata_dark_grid = (Grid2D) pci_transpose.getDark();
//	Grid2D ata_phase_grid = (Grid2D) pci_transpose.getPhase();
//	
//	SimpleMatrix ata_amp = new SimpleMatrix(pci_transpose.getWidth(),pci_transpose.getHeight());
//	SimpleMatrix ata_phase = new SimpleMatrix(pci_transpose.getWidth(),pci_transpose.getHeight());
//	SimpleMatrix ata_dark = new SimpleMatrix(pci_transpose.getWidth(),pci_transpose.getHeight());
//	for(int i = 0; i<pci_sino.getWidth(); i++){
//		for(int j = 0; j < pci_sino.getHeight(); j++){
//			ata_amp.setElementValue(i, j, ata_amp_grid.getAtIndex(i, j));
//			ata_phase.setElementValue(i, j, ata_phase_grid.getAtIndex(i, j));
//			ata_dark.setElementValue(i, j, ata_dark_grid.getAtIndex(i, j));
//		}
//	}
//	DecompositionSVD decompose = new DecompositionSVD(ata_amp, true, true, true);
//	double [] sv = decompose.getSingularValues();
//	double max = maxValue(sv);
//	System.out.println("Max Eigenvalue: "+ max);
//	
//		
//		int i = 1;
//		while( i <= iter_num || error == 5){
//			//ax = pci_sino * pci_recon
//					
//			// p = 
//
//		return pci_recon;
//	}
	
	
	
	/**
	 * MAIN 
	 * @param args
	 */
	public static void main(String[] args) {
		new ImageJ();
		
		// create phantoms
		PhaseContrastImages pci = simulate_data();
		pci.show("pci");
		
		// define geometry
		int detector_width = 200;
		int nr_of_projections = 360;	
		ProjectorAndBackprojector p = new ProjectorAndBackprojector(nr_of_projections, 2*Math.PI); 
		
		PhaseContrastImages pci_sino = p.project(pci, new Grid2D(detector_width, nr_of_projections));
//		pci_sino.show("pci_sino");
		
		
//		PhaseContrastImages pci_fake = fake_truncation(pci, 25, 50, "x", 0, false, "pending");
//		pci_fake.show("pci-fake");
		// project
		
		
//		PhaseContrastImages pci_sino_fake = p.project(pci_fake, new Grid2D(detector_width, nr_of_projections));
//		pci_sino_fake.show("pci_sino_fake");
//		PhaseContrastImages pci_roi_reko = iterative_region_of_interest(pci_sino, pci_sino_fake, 5);
//		pci_roi_reko.show("pci_roi_reko");
		
		// fake truncation	
//		PhaseContrastImages pci_sino_fake = fake_truncation(pci_sino, 25, 50, "x", 0, false, "pending");
//		PhaseContrastImages pci_sino_fake2 = fake_truncation(pci_sino_fake, 150, 175, "x", 0, false, "pending");
//		pci_sino_fake2.show("fake_sinograms");
		
		// backproject (reconstruct)
//		PhaseContrastImages pci_reko = p.backprojection_pixel(pci_sino, size);
//		pci_reko.show("reconstruction");
		
//		PhaseContrastImages pci_diff = calculate_PCI(pci, pci_reko, false);
//		pci_diff.show("difference");
			
	
	
	
//		NumericPointwiseOperators.subtractBy(pci_reko.getAmp(), pci_reko.getAmp());
		
//		PhaseContrastImages end = iterative_reconstruction(pci_sino, 300, 0);
//		end.show("end");
		NumericGrid dark = pci.getDark();
		NumericGrid amp = pci.getAmp();
		
//		NumericGrid fabso = initial_estimation(amp, dark);
//		for(int i = 1; i <= 5; i++){
//			System.out.println("iteration: "+ i);
//			NumericGrid thresh_map = thresholding_map(fabso, dark, (float) 0.0001);
////			thresh_map.show("thresh_map");
//			fabso = refinement(fabso, thresh_map);
////			fabso.show("fabso");
//		}
//		NumericPointwiseOperators.subtractBy(dark, fabso);
//		dark.show("dabso");
//		System.out.println("done");
//		double[][] points = {{1.0, 20.0}, {2.0, 30.0}, {3.0, 40}, {5, 60}};
		double [][] points = get_comp_points(amp, dark);
		System.out.println(ReflectionToStringBuilder.toString(points));
		ScatterPlot.plot(points);
		Regression.deg1(points);
		PolynomialRegression.calc_regression(points, 2);
		PolynomialRegression.calc_regression(points, 3);
		

	}

}
