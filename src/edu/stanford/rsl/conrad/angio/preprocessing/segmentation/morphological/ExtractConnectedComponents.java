package edu.stanford.rsl.conrad.angio.preprocessing.segmentation.morphological;

import java.util.ArrayList;

import util.FindConnectedRegions;
import util.FindConnectedRegions.Results;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ImageProcessor;
import ij.process.StackConverter;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.angio.preprocessing.segmentation.morphological.tools.StructuringElement;

public class ExtractConnectedComponents {

	private int dilationSize = 4;
	
	boolean diagonal = true;
	boolean imagePerRegion = false;
	boolean imageAllRegions = true;
	private boolean showResults = false;
	boolean mustHaveSameValue = false;
	boolean startFromPointROI = false;
	boolean autoSubtract = false;
	double valuesOverDouble = 100;
	double minimumPointsInRegionDouble = 1;
	int stopAfterNumberOfRegions = (int) -1;
	
	public static void main(String[] args){
		String filename = ".../seg.tif";
		Grid3D img = ImageUtil.wrapImagePlus(IJ.openImage(filename));
		
		ExtractConnectedComponents concomp = new ExtractConnectedComponents();
		Grid2D comps = concomp.runSlice(img.getSubGrid(0));
		
		new ImageJ();
		comps.show();
		
	}
	
	public Grid2D runSlice(Grid2D img){
		Grid2D erodedMask = erodeSlice((Grid2D) img.clone());
		ImagePlus imp = ImageUtil.wrapGrid(erodedMask, "");
		FindConnectedRegions fcr = new FindConnectedRegions();
		Results res = null;
		try {
			res = fcr.run( imp,
				 diagonal,
				 imagePerRegion,
				 imageAllRegions,
				 isShowResults(),
				 mustHaveSameValue,
				 startFromPointROI,
				 autoSubtract,
				 valuesOverDouble,
				 minimumPointsInRegionDouble,
				 stopAfterNumberOfRegions,
				 true /* noUI */ );
		} catch( IllegalArgumentException iae ) {
			IJ.error(""+iae);
		}
		Grid3D impAsGrid = ImageUtil.wrapImagePlus(res.allRegions);
		impAsGrid.setSpacing(img.getSpacing());
		impAsGrid.setOrigin(img.getOrigin());
		return impAsGrid.getSubGrid(0);
	}
	
	public Grid3D runStack(Grid3D img){
		int[] gSize = img.getSize();
		Grid3D erodedMask = erodeStack((Grid3D) img.clone());
		Grid3D conComp = new Grid3D(erodedMask);
		conComp.setSpacing(img.getSpacing());
		conComp.setOrigin(img.getOrigin());
		for(int k = 0; k < gSize[2]; k++){
			ImagePlus imp = ImageUtil.wrapGrid(erodedMask.getSubGrid(k), "");
			FindConnectedRegions fcr = new FindConnectedRegions();
			Results res = null;
			try {
				res = fcr.run( imp,
					 diagonal,
					 imagePerRegion,
					 imageAllRegions,
					 isShowResults(),
					 mustHaveSameValue,
					 startFromPointROI,
					 autoSubtract,
					 valuesOverDouble,
					 minimumPointsInRegionDouble,
					 stopAfterNumberOfRegions,
					 true /* noUI */ );
			} catch( IllegalArgumentException iae ) {
				IJ.error(""+iae);
			}
			Grid3D impAsGrid = ImageUtil.wrapImagePlus(res.allRegions);
			conComp.setSubGrid(k, impAsGrid.getSubGrid(0));
		}
		return conComp;
	}
	
	public Grid3D run3D(Grid3D img){
		Grid3D conComp = new Grid3D(img);
		conComp.setSpacing(img.getSpacing());
		conComp.setOrigin(img.getOrigin());
		ImagePlus imp = ImageUtil.wrapGrid3D(img, "");
		FindConnectedRegions fcr = new FindConnectedRegions();
		Results res = null;
		try {
			res = fcr.run( imp,
				 diagonal,
				 imagePerRegion,
				 imageAllRegions,
				 isShowResults(),
				 mustHaveSameValue,
				 startFromPointROI,
				 autoSubtract,
				 valuesOverDouble,
				 minimumPointsInRegionDouble,
				 stopAfterNumberOfRegions,
				 true /* noUI */ );
		} catch( IllegalArgumentException iae ) {
			IJ.error(""+iae);
		}
		Grid3D impAsGrid = ImageUtil.wrapImagePlus(res.allRegions);
		
		return impAsGrid;
	}
	
	public Grid2D removeSmallConnectedComponentsSlice(Grid2D img, int ccSize){
		Grid2D conComps = runSlice(img);
		int[] gSize = conComps.getSize();
		
		ArrayList<Integer> ccs = new ArrayList<Integer>();
		ArrayList<Integer> ccSizes = new ArrayList<Integer>();
		for(int i = 0; i < gSize[0]; i++){
			for(int j = 0; j < gSize[1]; j++){
				int v = (int)conComps.getAtIndex(i, j);
				if(v == 0)
					continue;
				int idx = idxOf(ccs,v);
				if(idx < 0){
					ccs.add(v);
					ccSizes.add(1);
				}else{						
					ccSizes.set(idx, ccSizes.get(idx)+1);
				}
			}
		}
		ArrayList<Integer> toBeRem = new ArrayList<Integer>();
		for(int l = 0; l < ccs.size(); l++){
			if(ccSizes.get(l) < ccSize){
				toBeRem.add(ccs.get(l));
			}
		}			
		for(int i = 0; i < gSize[0]; i++){
			for(int j = 0; j < gSize[1]; j++){
				int v = (int)conComps.getAtIndex(i, j);
				if(v == 0)
					continue;
			
				int idx = idxOf(toBeRem,v);
				if(idx < 0){
					conComps.setAtIndex(i, j, img.getAtIndex(i, j));
				}else{
					conComps.setAtIndex(i, j, 0);
				}
			}
		}	
		return conComps;
	}
	
	public Grid3D removeSmallConnectedComponentsStack(Grid3D img, int ccSize){
		Grid3D conComps = runStack(img);
		int[] gSize = conComps.getSize();
		
		for(int k = 0; k < gSize[2]; k++){
			ArrayList<Integer> ccs = new ArrayList<Integer>();
			ArrayList<Integer> ccSizes = new ArrayList<Integer>();
			for(int i = 0; i < gSize[0]; i++){
				for(int j = 0; j < gSize[1]; j++){
					int v = (int)conComps.getAtIndex(i, j, k);
					if(v == 0)
						continue;
					int idx = idxOf(ccs,v);
					if(idx < 0){
						ccs.add(v);
						ccSizes.add(1);
					}else{						
						ccSizes.set(idx, ccSizes.get(idx)+1);
					}
				}
			}
			ArrayList<Integer> toBeRem = new ArrayList<Integer>();
			for(int l = 0; l < ccs.size(); l++){
				if(ccSizes.get(l) < ccSize){
					toBeRem.add(ccs.get(l));
				}
			}			
			for(int i = 0; i < gSize[0]; i++){
				for(int j = 0; j < gSize[1]; j++){
					int v = (int)conComps.getAtIndex(i, j, k);
					if(v == 0)
						continue;
				
					int idx = idxOf(toBeRem,v);
					if(idx < 0){
						conComps.setAtIndex(i, j, k, img.getAtIndex(i, j, k));
					}else{
						conComps.setAtIndex(i, j, k, 0);
					}
				}
			}
		}
		return conComps;
	}
	
	private Grid2D erodeSlice(Grid2D m){
		ImageProcessor imp = ImageUtil.wrapGrid2D((Grid2D)m.clone());
		
		StructuringElement structuringElementDilation = new StructuringElement("circle", getDilationSize(), false);
		
		ImageProcessor tempIp = imp.convertToByteProcessor();
		ImagePlus ip = new ImagePlus("temp", tempIp);
		ip = Morphology.dilate(ip, structuringElementDilation);
		
		return ImageUtil.wrapImageProcessor(ip.getProcessor());
	}
	
	private Grid3D erodeStack(Grid3D m){
		int[] s = m.getSize();
		
		ImagePlus imp = ImageUtil.wrapGrid((Grid3D)m.clone(), "");
		StackConverter sC = new StackConverter(imp);
		sC.convertToGray8();
		
		StructuringElement structuringElementDilation = new StructuringElement("circle", getDilationSize(), false);
		
		ImageStack stack = imp.getImageStack();
		for(int k = 0; k < m.getSize()[2]; k++){
			System.out.println("Enlarging slice "+ String.valueOf(k+1)+" of "+ String.valueOf(s[2])+".");
			ImageProcessor tempIp = stack.getProcessor(k+1);
			ImagePlus ip = new ImagePlus("temp", tempIp);
			ip = Morphology.dilate(ip, structuringElementDilation);
			//ip.show();
			stack.setProcessor(ip.getChannelProcessor(), k+1);
		}
		imp.setStack(stack);
		return ImageUtil.wrapImagePlus(imp);
	}

	private int idxOf(ArrayList<Integer> list, int test){
		for(int i = 0; i < list.size(); i++){
			if(list.get(i) == test){
				return i;
			}
		}
		return (-1);
	}
	
	public int getDilationSize() {
		return dilationSize;
	}

	public void setDilationSize(int dilationSize) {
		this.dilationSize = dilationSize;
	}

	public boolean isShowResults() {
		return showResults;
	}

	public void setShowResults(boolean showResults) {
		this.showResults = showResults;
	}

	public double getMinimumPointsInRegionDouble() {
		return minimumPointsInRegionDouble;
	}

	public void setMinimumPointsInRegionDouble(double minimumPointsInRegionDouble) {
		this.minimumPointsInRegionDouble = minimumPointsInRegionDouble;
	}
}
