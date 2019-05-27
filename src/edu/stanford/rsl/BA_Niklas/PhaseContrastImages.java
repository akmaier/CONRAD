package edu.stanford.rsl.BA_Niklas;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;

/**
 * Container class for Phase-Contrast images
 * 
 * @author Lina Felsner
 *
 */
public class PhaseContrastImages {

	private NumericGrid amp;
	private NumericGrid phase;
	private NumericGrid dark;
	
	private int[] size;
	
	// constructor
	public PhaseContrastImages(NumericGrid amplitude, NumericGrid phase_contrast, NumericGrid dark_field){
		if(amplitude == null || phase_contrast == null || dark_field == null) 
			throw new IllegalArgumentException("Numeric Grid is null");
		
		this.amp = amplitude.clone();
		this.amp.setSpacing(amplitude.getSpacing());
		
		this.phase = phase_contrast.clone();
		this.phase.setSpacing(phase_contrast.getSpacing());
		
		this.dark = dark_field.clone();
		this.dark.setSpacing(dark_field.getSpacing());

		checkGridSizes();
		this.size = this.amp.getSize();
		
	}
	
	public PhaseContrastImages(PhaseContrastImages pci){

		this.amp = pci.getAmp().clone();
		this.amp.setSpacing(pci.getAmp().getSpacing());
		
		this.phase = pci.getPhase().clone();
		this.phase.setSpacing(pci.getPhase().getSpacing());
		
		this.dark = pci.getDark().clone();
		this.dark.setSpacing(pci.getDark().getSpacing());

		this.size = this.amp.getSize();
		
	}

	public int[] getSize() {
		return this.size;
	}

	public int getWidth() {
		return this.size[0];
	}
	
	public int getHeight() {
		return this.size[1];
	}

	public int getDepth() {
		return this.size[2];
	}
	
	public NumericGrid getAmp() {
		return this.amp;
	}
	
	public NumericGrid getPhase() {
		return this.phase;
	}
	
	public NumericGrid getDark() {
		return this.dark;
	}
		
	public void set_spacing(double... spacing) {
		if(spacing.length != size.length)
			throw new IllegalArgumentException("The spacing cannot be set. Wrong dimensionality");
		
		this.amp.setSpacing(spacing);
		this.phase.setSpacing(spacing);
		this.dark.setSpacing(spacing);
	}
	
	
	public void show() {
		this.amp.show("amp");
		this.phase.show("phase");
		this.dark.show("dark");
	}
	
	public void show(String name) {
		this.amp.show(name + " - amp");
		this.phase.show(name + " - phase");
		this.dark.show(name + " - dark");
	}
	
	private void checkGridSizes(){
		
		if (this.amp.getSize().length != this.phase.getSize().length || this.amp.getSize().length != this.dark.getSize().length){
			throw new IllegalArgumentException("Reference and evaluation grid cannot have different dimensions!");
		}else{
			for (int i = 0; i < this.amp.getSize().length; i++) {
				if (this.amp.getSize()[i] != this.phase.getSize()[i] || this.amp.getSize()[i] != this.dark.getSize()[i]) {
					throw new IllegalArgumentException("Reference and evaluation grid cannot have different sizes!");
				}
			}
		}
	
	}

	public PhaseContrastImages multipliedBy(Grid2D mask) {
		
		// TODO check if PhaseContrastImages is Grid2D or apply it slice wise
		
		Grid2D amplitude = 	(Grid2D) NumericPointwiseOperators.multipliedBy(this.amp, mask);
		Grid2D phase = 	(Grid2D) NumericPointwiseOperators.multipliedBy(this.phase, mask);
		Grid2D dark = 	(Grid2D) NumericPointwiseOperators.multipliedBy(this.dark, mask);

		PhaseContrastImages pci = new PhaseContrastImages(amplitude, phase, dark);
		return pci;
	}
	
	public void multiplyBy(Grid2D mask) {
		
		// TODO check if PhaseContrastImages is Grid2D  or apply it slice wise
		
		NumericPointwiseOperators.multiplyBy(this.amp, mask);
		NumericPointwiseOperators.multiplyBy(this.phase, mask);
		NumericPointwiseOperators.multiplyBy(this.dark, mask);
		
	}

	public PhaseContrastImages get_slice_PCI(int slice) {
		
		if(slice < size[2]) {
			PhaseContrastImages pci_slice = new PhaseContrastImages(this.amp.getSubGrid(slice), this.phase.getSubGrid(slice), this.dark.getSubGrid(slice));
			return pci_slice;
		}
		
		return null;
	}


}
