/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.preprocessing;

import ij.IJ;
import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.angio.points.DistanceTransformUtil;
import edu.stanford.rsl.conrad.angio.preprocessing.segmentation.hessian.Frangi2D;
import edu.stanford.rsl.conrad.angio.preprocessing.segmentation.morphological.ExtractConnectedComponents;
import edu.stanford.rsl.conrad.angio.preprocessing.segmentation.morphological.HysteresisThresholding;
import edu.stanford.rsl.conrad.angio.util.data.collection.DataSet;
import edu.stanford.rsl.conrad.angio.util.data.collection.DataSets;
import edu.stanford.rsl.conrad.angio.util.data.organization.Angiogram;
import edu.stanford.rsl.conrad.angio.util.image.HistogramPercentile;
import edu.stanford.rsl.conrad.angio.util.image.ImageOps;

public class VolumetricSegmentationPipeline extends PreprocessingPipeline{

	private Grid3D volumetricResponse;
	double maxDistance = 5.0; // mm // 5.0; // mm //
	
	public static void main(String[] args) {
		DataSets datasets = DataSets.getInstance();
		
		int caseID = 51;
		
		String[] projFiles = new String[2];
		String[] outFiles = new String[2];
		
		String dir   = ".../";
		projFiles[0] = dir + "beats_7/contrast.tif";
		outFiles[0]  = dir + "beats_7/seg.tif";		
		
		projFiles[1] = dir + "static/contrast.tif";
		outFiles[1]  = dir + "static/seg.tif";
		
//		projFiles[1] = dir + "Noise_Level_2/c6_b020/contrast.tif";
//		outFiles[1]  = dir + "Noise_Level_2/c6_b020/seg.tif";
//		
//		projFiles[2] = dir + "Noise_Level_1/c5_b075/contrast.tif";
//		outFiles[2]  = dir + "Noise_Level_1/c5_b075/seg.tif";
//		
//		projFiles[3] = dir + "Noise_Level_2/c5_b075/contrast.tif";
//		outFiles[3]  = dir + "Noise_Level_2/c5_b075/seg.tif";
		
		DataSet ds = datasets.getCase(caseID);
		
		for(int i = 0; i < projFiles.length; i++){
			System.out.println("On "+ projFiles[i]);
			
			ds.setProjectionFile(projFiles[i]);
			
			VolumetricSegmentationPipeline prepPipe = new VolumetricSegmentationPipeline(ds);
			prepPipe.setRefHeartPhase(-1);
			prepPipe.evaluate();			
			
			IJ.saveAsTiff(ImageUtil.wrapGrid3D(prepPipe.getVolumetricResponse(),""), outFiles[i]);
			new ImageJ();
			prepPipe.getPreprocessedAngiogram().getProjections().show();
			prepPipe.getVolumetricResponse().show();
			System.out.println("Finished task "+String.valueOf(i+1)+" of "+String.valueOf(projFiles.length));
		}
		
	}
	
	public VolumetricSegmentationPipeline(DataSet ds) {
		super(ds);
	}

	public void evaluate(){
		if(!initialized){
			init();
		}
		System.out.println("Starting the preprocessing.");
		
		Angiogram prep = new Angiogram(gatedAngiogram);
		// noise filtering
		prep.setProjections(noiseSuppression(prep.getProjections()));
		// vessel segmentation, centerline extraction, distance transform
		prep.setSkeletons(centerlineExtractionPipeline(prep.getProjections()));
		prep.setReconCostMap(DistanceTransformUtil.slicewiseDistanceTransform(prep.getProjections(), prep.getSkeletons()));
		
		
		this.preprocessedAngiogram = prep;
		Grid3D vol = mergeSegs(this.volumetricResponse, prep.getReconCostMap());
		for(int k = 0; k < vol.getSize()[2]; k++){
			Grid2D slice = new Grid2D(vol.getSubGrid(k));
			// Remove small connected components
			ExtractConnectedComponents conComp = new ExtractConnectedComponents();
			conComp.setShowResults(false);
			conComp.setDilationSize(0);
			Grid2D hystCC = conComp.removeSmallConnectedComponentsSlice(slice, 25);
			hystCC = ImageOps.thresholdImage(hystCC, 0.0);
			vol.setSubGrid(k, hystCC);
		}
		this.volumetricResponse = vol;		
		System.out.println("All done.");
	}
		
	private Grid3D mergeSegs(Grid3D vessel, Grid3D distanceMap) {
		int[] gSize = vessel.getSize();
		Grid3D refined = new Grid3D(vessel);
		for(int k = 0; k < gSize[2]; k++){
			for(int i = 0; i < gSize[0]; i++){
				for(int j = 0; j < gSize[1]; j++){
					if(vessel.getAtIndex(i, j, k)>0){
						float dist = distanceMap.getAtIndex(i, j, k);
						if(dist > maxDistance){
							refined.setAtIndex(i, j, k, 0);
						}
					}
				}
			}
		}
		return refined;
	}
	
	@Override
	public Grid3D vesselEnhancement(Grid3D img){		
		Frangi2D vness = new Frangi2D(img);
		vness.setScales(dataset.getPreproSet().getHessianScales());
		vness.setRoi(dataset.getPreproSet().getRoi());
		vness.setStructurenessPercentile(dataset.getPreproSet().getStructurenessPercentile());
		vness.evaluate(dataset.getPreproSet().getGammaThreshold());
		Grid3D vnessImg = vness.getResult();
		for(int k = 0; k < img.getSize()[2]; k++){
			Grid2D slice = new Grid2D(vnessImg.getSubGrid(k));
			// Hysteresis thresholding of Sato response
			HistogramPercentile histPerc = new HistogramPercentile(slice);
			double[] th = new double[2];
			th[0] = histPerc.getPercentile(dataset.getPreproSet().getVesselPercentile()[0]);
			th[1] = histPerc.getPercentile(dataset.getPreproSet().getVesselPercentile()[1]);
			HysteresisThresholding hyst = new HysteresisThresholding(th[1], th[0]);
			Grid2D hysted = hyst.run(slice);
			// Remove small connected components
			ExtractConnectedComponents conComp = new ExtractConnectedComponents();
			conComp.setShowResults(false);
			conComp.setDilationSize(dataset.getPreproSet().getConCompDilation());
			Grid2D hystCC = conComp.removeSmallConnectedComponentsSlice(hysted, dataset.getPreproSet().getConCompSize());
			Grid2D normalized = ImageOps.normalizeOutsideMask(vnessImg.getSubGrid(k), hystCC, th[0], 0.0f);
			vnessImg.setSubGrid(k, normalized);
		}
		this.volumetricResponse = vnessImg;
//		new ImageJ();
//		volumetricResponse.show();
		return vnessImg;
	}
	
	
	public Grid3D getVolumetricResponse(){
		return this.volumetricResponse; 
	}
	
}
