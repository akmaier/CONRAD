package edu.stanford.rsl.BA_Niklas;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.BA_Niklas.PhaseContrastImages;
import edu.stanford.rsl.BA_Niklas.ProjectorAndBackprojector;
import ij.ImageJ;


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
	
	static int nr_ellipses = 5;
    static SimpleMatrix[] R; 	// rotation matrices. 
    static double[][] t; 	// displacement vectors.   
    static double[][] ab; 	// the length of the principal axes.    
    static double[] rho; 	// signal intensity.
    static double[] df; 	// dark-field signal intensity.
    
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
	
	public static PhaseContrastImages fake_truncation(PhaseContrastImages pci_sino, int min, int max){
		int width = pci_sino.getWidth();
		int height = pci_sino.getHeight();
		Grid2D fake_amp = (Grid2D) pci_sino.getAmp();
		Grid2D fake_dark = (Grid2D) pci_sino.getDark();
		Grid2D fake_phase = (Grid2D) pci_sino.getPhase();
		for (int i= 0; i< width; i++){
			for(int j =0; j < height; j++){
				if (i > min && i < max){
					fake_amp.setAtIndex(i, j, 0);
					fake_dark.setAtIndex(i, j, 0);
					fake_phase.setAtIndex(i, j, 0);
				}
			}
		}
		PhaseContrastImages pci_sino_fake = new PhaseContrastImages(fake_amp,  fake_phase, fake_dark);
		return pci_sino_fake;
	}

	/**
	 * MAIN 
	 * @param args
	 */
	public static void main(String[] args) {
		new ImageJ();
		
		// create phantoms
		PhaseContrastImages pci = simulate_data();
		pci.show("simulated data");
		
		// define geometry
		int detector_width = 200;
		int nr_of_projections = 360;	
		ProjectorAndBackprojector p = new ProjectorAndBackprojector(nr_of_projections, 2*Math.PI); 
		
		// project
		PhaseContrastImages pci_sino = p.project(pci, new Grid2D(detector_width, nr_of_projections));
		pci_sino.show("sinograms");
		
		// fake truncation	
		PhaseContrastImages pci_sino_fake = fake_truncation(pci_sino, 25, 50);
		pci_sino_fake.show("fake_sinograms");
		
		// backproject (reconstruct)
		PhaseContrastImages pci_reko = p.filtered_backprojection(pci_sino_fake, size);
		pci_reko.show("reconstruction");
		
		System.out.println("done");
	}

}
