package edu.stanford.rsl.BA_Niklas;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
//import edu.stanford.rsl.science.felsner.Utils;
//import edu.stanford.rsl.science.felsner.pci.DerivativeAndIntegration;
import edu.stanford.rsl.BA_Niklas.PhaseContrastImages;
import edu.stanford.rsl.science.iterativedesignstudy.ParallelBackprojectorPixelDriven;
import edu.stanford.rsl.science.iterativedesignstudy.ParallelBackprojectorPixelDrivenCL;
import edu.stanford.rsl.science.iterativedesignstudy.ParallelBackprojectorRayDriven;
import edu.stanford.rsl.science.iterativedesignstudy.ParallelProjectorRayDriven;
//TODO ParallelBackprojectorRayDrivenCL ?
import edu.stanford.rsl.science.iterativedesignstudy.ParallelProjectorRayDrivenCL;
import edu.stanford.rsl.tutorial.filters.DerivativeKernel;
import edu.stanford.rsl.tutorial.filters.HilbertKernel;
import edu.stanford.rsl.tutorial.filters.RamLakKernel;


/**
 * parallel projectors and backprojectors for PCI
 * 
 * @author Lina Felsner
 *
 */
public class ProjectorAndBackprojector {
	private static boolean debug = false;
	
	int nr_of_projections;
	double angular_range;
	
	/**
	 * define parameters for 2D circular trajectory
	 * 
	 * @param nrProj - nr of projections
	 * @param angRange - angular range in rad
	 */
	public ProjectorAndBackprojector(int nrProj, double angRange) {
		
		this.nr_of_projections = nrProj;
		this.angular_range = angRange;
		
	}

	/*
	 * **************************** sinograms *******************************************************
	 */
	
	/**
	 * computes the sinogram from the projection data
	 * 
	 * @param data - projection images
	 * @return the sinograms
	 */
	public static Grid3D create_sinogram(Grid3D data) {
		Grid3D sino = new Grid3D(data.getSize()[0], data.getSize()[2], data.getSize()[1]);
		Grid3D tmp = new Grid3D(data);
	
		for (int k = 0; k < sino.getSize()[2]; k++) 
			for (int j = 0; j < sino.getSize()[1]; j++) 
				for (int i = 0; i < sino.getSize()[0]; i++) 
					sino.setAtIndex(i, j, k, tmp.getAtIndex(i, k, j));
		sino.setSpacing(new double[] {1.0d, 1.0d});
		return sino;
	}
	
	/**
	 * computes the sinograms from the projection data
	 * 
	 * @param data - projection images
	 * @return the sinograms of all three signals
	 */
	public static PhaseContrastImages create_sinograms(PhaseContrastImages data) {
		Grid3D sino_amp = new Grid3D(data.getWidth(), data.getDepth(), data.getHeight());
		Grid3D sino_phase = new Grid3D(data.getWidth(), data.getDepth(), data.getHeight());
		Grid3D sino_dark = new Grid3D(data.getWidth(), data.getDepth(), data.getHeight());
		
		Grid3D tmp_amp = new Grid3D((Grid3D) data.getAmp());
		Grid3D tmp_phase = new Grid3D((Grid3D) data.getPhase());
		Grid3D tmp_dark = new Grid3D((Grid3D) data.getDark());
		
		for (int k = 0; k < sino_amp.getSize()[2]; k++) {
			for (int j = 0; j < sino_amp.getSize()[1]; j++) {
				for (int i = 0; i < sino_amp.getSize()[0]; i++) {
					sino_amp.setAtIndex(i, j, k, tmp_amp.getAtIndex(i, k, j));
					sino_phase.setAtIndex(i, j, k, tmp_phase.getAtIndex(i, k, j));
					sino_dark.setAtIndex(i, j, k, tmp_dark.getAtIndex(i, k, j));
				}
			}
		}

		// create PCI and return
		PhaseContrastImages pci_sino = new PhaseContrastImages(sino_amp, sino_phase, sino_dark);	
		if(debug) pci_sino.show("sinogram");
		
		return pci_sino;
	}
	
	
	/*
	 * ****************************** (ray driven) forward projection **********************************************
	 */
	
	/**
	 * forward projection of all three PCI signals
	 * 
	 * @param object - PCI reconstruction/object data
	 * @param sinogram - sinogram used for the dimensions
	 * @return sinograms for all three signals
	 */
	public PhaseContrastImages project(PhaseContrastImages object, Grid2D sinogram) {
		
		Grid2D sino_amp = project((Grid2D)object.getAmp(), sinogram);
		Grid2D sino_phase = project((Grid2D)object.getPhase(), sinogram);
//		sino_phase = DerivativeAndIntegration.apply_derivative(sino_phase);
		Grid2D sino_dark = project((Grid2D)object.getDark(), sinogram);
		
		PhaseContrastImages pci = new PhaseContrastImages(sino_amp, sino_phase, sino_dark);
		return pci;
	}
	
	/**
	 * forward projection of a object/reconstruction
	 * 
	 * @param object - reconstruction/object data
	 * @param sinogram - sinogram used for the dimensions
	 * @return sinogram
	 */
	public Grid2D project(Grid2D object, Grid2D sinogram) {
		int x = sinogram.getWidth() - 1;
		
		if (sinogram.getHeight() != nr_of_projections) {
			System.err.println("sinogram height doen not match nuber of projections.");
			return null;
		}
		
		// Parallel geometry params
		double maxTheta = angular_range;  //2*Math.PI; 	//	angular range in radian
		double deltaTheta = angular_range / (double)nr_of_projections; //angular step size in radian
		double maxS = x; //detector size in mm
		double deltaS = 1; //detector element size in mm
//		ParallelProjectorRayDrivenCL projector = new ParallelProjectorRayDrivenCL(maxTheta, deltaTheta, maxS, deltaS);
		ParallelProjectorRayDriven projector = new ParallelProjectorRayDriven(maxTheta, deltaTheta, maxS, deltaS);

		
		Grid2D sino = new Grid2D(sinogram.getWidth(), sinogram.getHeight()); 
		Grid2D obj = new Grid2D(object);
		
		sino.setSpacing(new double[] {deltaS, deltaTheta});
		obj.setSpacing(new double[] {1d, 1d});
		sino = (Grid2D) projector.project(obj, sino);
		
		return sino;
	}
	
	
	/*
	 * ******************************* ray-driven backprojection *****************************************
	 */
	
	/**
	 * ray driven backprojection
	 * 
	 * @param sino - sinograms of all three PCI signals
	 * @return PCI reconstructions
	 */
	public PhaseContrastImages backprojection_ray(PhaseContrastImages sino) {
		System.out.println("ray driven reco PCI");
		
		// sinograms
		Grid2D amp = new Grid2D((Grid2D)sino.getAmp());			
		Grid2D phase = new Grid2D((Grid2D)sino.getPhase());
		Grid2D dark = new Grid2D((Grid2D)sino.getDark());

		Grid2D reko_amp = backprojection_ray(amp);
		System.out.println("reco amp done");
		
		Grid2D reko_phase = backprojection_ray(phase);
		System.out.println("reco phase done");
		
		Grid2D reko_dark = backprojection_ray(dark);
		System.out.println("reco dark done");
		
		PhaseContrastImages pci = new PhaseContrastImages(reko_amp, reko_phase, reko_dark);
		
		return pci;
	}
	
	/**
	 * ray driven backprojection
	 * 
	 * @param input_sino - sinogram 
	 * @return reconstruction
	 */
	public Grid2D backprojection_ray(Grid2D input_sino){

		Grid2D sino = new Grid2D(input_sino);
		sino.setSpacing(new double[] {1.d, angular_range/nr_of_projections}); 
		
	
		int reko_size = sino.getWidth() -1;
		Grid2D reko = new Grid2D(reko_size, reko_size);
		reko.setSpacing(new double[] {1.0d, 1.0d});
		
		// Backproject
		ParallelBackprojectorRayDriven backproj = new ParallelBackprojectorRayDriven();
		reko = backproj.backprojectRayDriven(sino, reko);
		if(debug) reko.show("Ray Driven BP");
		
		// TODO set edges to zero?
//		Utils.set_edges_to_zero(reko);
		
		return reko;
	}
	
	
	/*
	 * ******************************* pixel-driven backprojection *****************************************
	 */
	
	/**
	 * pixel driven backprojection
	 * 
	 * @param sino - sinograms of all three PCI signals
	 * @return PCI reconstructions
	 */
	public PhaseContrastImages backprojection_pixel(PhaseContrastImages sino, int reko_size) {
		System.out.println("ray driven reco PCI");
		
		// sinograms
		Grid2D amp = new Grid2D((Grid2D)sino.getAmp());			
		Grid2D phase = new Grid2D((Grid2D)sino.getPhase());
		Grid2D dark = new Grid2D((Grid2D)sino.getDark());

		Grid2D reko_amp = backprojection_pixel(amp, reko_size);
		System.out.println("reco amp done");
		
		Grid2D reko_phase = backprojection_pixel(phase, reko_size);
		System.out.println("reco phase done");
		
		Grid2D reko_dark = backprojection_pixel(dark, reko_size);
		System.out.println("reco dark done");
		
		PhaseContrastImages pci = new PhaseContrastImages(reko_amp, reko_phase, reko_dark);
		
		return pci;
	}
	
	/**
	 * pixel driven backprojection
	 * 
	 * @param input_sino - sinogram 
	 * @return reconstruction
	 */
	public Grid2D backprojection_pixel(Grid2D input_sino, int reko_size){

		Grid2D sino =  new Grid2D(input_sino);
		sino.setSpacing(new double[] {1.d, angular_range/nr_of_projections}); 
		
		Grid2D reko_pixel = new Grid2D(reko_size, reko_size);
		reko_pixel.setSpacing(new double[] {1.0d, 1.0d});
		
		ParallelBackprojectorPixelDriven backproj_pixel = new ParallelBackprojectorPixelDriven(sino);
		reko_pixel =  backproj_pixel.backprojectPixelDriven(sino, reko_pixel);
		if(debug) reko_pixel.show("Pixel Driven BP");
		
		return reko_pixel;
	}
	

	
	/*
	 * ************************ filtered backprojection ***********************************************************
	 */
	
	/**
	 * Filter PCI data for fbp
	 * 
	 * @param sino
	 * @return filtered sinograms
	 */
	private PhaseContrastImages filter(PhaseContrastImages sino) {
		
		PhaseContrastImages grid_pci = new PhaseContrastImages(sino.getAmp(), sino.getPhase(), sino.getDark());
		
		// Filter with RamLak
		int ramLak_size = sino.getWidth();
		RamLakKernel ramLak = new RamLakKernel(ramLak_size, 1);
		
		Grid2D filtered_amp = (Grid2D) grid_pci.getAmp();
		for (int theta = 0; theta < filtered_amp.getSize()[1]; ++theta) {
			//System.out.print(theta);
			ramLak.applyToGrid(filtered_amp.getSubGrid(theta));
		}
		
		Grid2D filtered_dark = (Grid2D) grid_pci.getDark();
		for (int theta = 0; theta < filtered_dark.getSize()[1]; ++theta) {
			//System.out.print(theta);
			ramLak.applyToGrid(filtered_dark.getSubGrid(theta));
		}
		
		// Filter with Hilbert
		HilbertKernel hilbert = new HilbertKernel(1);
		
		Grid2D filtered_phase = (Grid2D) grid_pci.getPhase();
		for (int theta = 0; theta < filtered_phase.getSize()[1]; ++theta) {
			//System.out.print(theta);
			hilbert.applyToGrid(filtered_phase.getSubGrid(theta));
		}
		
		// set spacing
		double spacing = angular_range/nr_of_projections;
		filtered_amp.setSpacing(new double[] {1.d, spacing});
		filtered_phase.setSpacing(new double[] {1.d, spacing});
		filtered_dark.setSpacing(new double[] {1.d, spacing});
		
		PhaseContrastImages pci = new PhaseContrastImages(filtered_amp, filtered_phase , filtered_dark);
		
		return pci;
	}
	
	/**
	 * parallel filtered backprojection
	 * 
	 * @param sino: of PCI
	 * @return PCI reconstruction
	 */
	public PhaseContrastImages filtered_backprojection(PhaseContrastImages sinogram, int reko_size){
		
		System.out.println("fbp reco PCI");
		
		PhaseContrastImages sino = new PhaseContrastImages(sinogram.getAmp(), sinogram.getPhase(), sinogram.getDark());
		PhaseContrastImages filtered_sino = filter(sino);
		
		PhaseContrastImages reco = backprojection_pixel(filtered_sino, reko_size);
		
		return reco;
	}
	
	/**
	 * parallel filtered backprojection
	 * 
	 * @param sino: of PCI
	 * @return pixel based reconstruction
	 */
	public Grid2D filtered_backprojection_phase(Grid2D sino, int reko_size){
		System.out.println("fbp reco");
		
		// Filter with Hilbert-filter
		HilbertKernel hilbert = new HilbertKernel(1);
		
		Grid2D filtered_phase = new Grid2D(sino);
		for (int theta = 0; theta < filtered_phase.getSize()[1]; ++theta) {
			//System.out.print(theta);
			hilbert.applyToGrid(filtered_phase.getSubGrid(theta));
		}
		
		Grid2D fbp = backprojection_pixel(filtered_phase, reko_size);

		return fbp;

	}	
	
	/**
	 * parallel filtered backprojection with ram-Lak
	 * 
	 * @param sino: absorption sinogram
	 * @return Reco of absorption signal
	 */
	public Grid2D filtered_backprojection(Grid2D sino, int reko_size){
		System.out.println("fbp reco");
		
		// Filter with RamLak
		Grid2D filteredSino = new Grid2D(sino);
		int ramLak_size = sino.getWidth();
		RamLakKernel ramLak = new RamLakKernel(ramLak_size, 1);
		for (int theta = 0; theta < filteredSino.getSize()[1]; ++theta) {
			//System.out.print(theta);
			ramLak.applyToGrid(filteredSino.getSubGrid(theta));
		}
		// filteredSino.show("The Filtered Sinogram");
		
		Grid2D fbp = backprojection_pixel(filteredSino, reko_size);

		return fbp;
	}	

}
