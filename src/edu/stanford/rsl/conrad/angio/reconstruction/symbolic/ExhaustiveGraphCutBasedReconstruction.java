/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.reconstruction.symbolic;

import ij.ImageJ;
import ij.ImagePlus;
import ij.Prefs;
import ij.gui.PointRoi;
import ij.plugin.frame.RoiManager;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.RegKeys;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.Skeleton;
import edu.stanford.rsl.conrad.angio.util.data.collection.DataSet;
import edu.stanford.rsl.conrad.angio.util.data.organization.Angiogram;
import edu.stanford.rsl.conrad.angio.util.data.organization.AngiographicView;
import edu.stanford.rsl.conrad.angio.util.io.PointAndRadiusIO;

public class ExhaustiveGraphCutBasedReconstruction {

	private DataSet dataset;
	
	private Grid3D projections = null;
	private Grid3D costMap = null;
	private ArrayList<Skeleton> skels = null;
	private Projection[] pMats = null;
	private double[] primAng = null;
	
	private ArrayList<AngiographicView> views = null;
	private ArrayList<ArrayList<PointND>> reconstructedPoints = null;
	private ArrayList<PointND> finalRecon = null;
	
	private double REPROJECTION_ERROR = 2.5d; // in mm
	private double SUPPRESSION_RADIUS = 1.5d;//0.46d; // in mm
			
	public ExhaustiveGraphCutBasedReconstruction(DataSet ds, Angiogram a){		
		init(ds, a.getProjections(), a.getReconCostMap(), a.getSkeletons(), a.getPMatrices(), a.getPrimAngles());
	}
	
	
	
	public ExhaustiveGraphCutBasedReconstruction(DataSet ds, Grid3D projections, Grid3D costMap, ArrayList<Skeleton> skeletons, 
			Projection[] pMatrices, double[] primAngles){		
		init(ds, projections,costMap,skeletons,pMatrices,primAngles);
	}

	private void init(DataSet ds, Grid3D proj, Grid3D cMap, ArrayList<Skeleton> skeletons,
			Projection[] pMatrices, double[] primAngles) {
		this.dataset = ds;
		this.projections = proj;
		this.costMap = cMap;
		this.skels = skeletons;
		this.pMats = pMatrices;
		this.primAng = primAngles;
		// initialize reconstructed point list, so that we can set points afterwards.
		reconstructedPoints = new ArrayList<ArrayList<PointND>>();
		for(int i = 0; i < pMats.length; i++){
			reconstructedPoints.add(new ArrayList<PointND>());
		}
		// initialize angiographic views
		views = new ArrayList<AngiographicView>();
		for(int i = 0; i < pMats.length; i++){
			AngiographicView view = new AngiographicView(pMats[i], costMap.getSubGrid(i), skels.get(i), primAng[i]);
			views.add(view);
		}			
	}
	
	public ArrayList<PointND> run(){
		reconstructWithReferenceViews();
		this.finalRecon = exhaustiveRefinementStep();
		return finalRecon;
	}
	
	/**
	 * Performs the iterative refinement using the error-tolerances specified in the class members.
	 * The underlying idea is, that we should use all segmented (i.e. skeleton pixels) not only the ones 
	 * in the selected / extraordinary view. In a perfect scenario, the reconstructed points would explain all 
	 * of the skeleton slab voxels. In the real case, however, segmentation imperfections (too few or too many pixels)
	 * and the neighborhood smoothness constraint lead to erroneous reconstructions. 
	 * In order to improve this condition we propose two things:
	 * a) reconstructed points should agree with the majority of observations (segmentations in our case). This requirement 
	 * can be checked by computing the second largest reprojection error.
	 * b) the majority of skeleton points should be explained by the reconstruction, i.e. at least one reconstructed is 
	 * closest to this point ( and also closer than the accepted threshold). If a point has not been the closest match 
	 * but there is a reconstructed point close to the suggested 3D point of the not-considered candidate, the two 
	 * 3D points are averaged. If the there is no close 3D point present yet, the suggested 3D point is added to the 
	 * reconstruction.
	 * @return
	 */
	private ArrayList<PointND> exhaustiveRefinementStep(){
		// exhaustive combination
		int numUsedBeforeMatching = 0;
		int totalPoints = 0;
		
		for(int v = 0; v < views.size(); v++){
			System.out.println("Refining view: "+String.valueOf(v+1));
			ArrayList<PointND> recons = reconstructedPoints.get(v);
			ArrayList<PointND> refinedReco = new ArrayList<PointND>();
			for(int i = 0; i < recons.size(); i++){
				// if the current point has not been used in the reconstruction yet
				// check if it is in agreement with the rest of the views (median error)
				PointND recoProposal = recons.get(i);
				double err = calculatePointError(recoProposal, v);
				// if the current point is in agreement with the rest of the views
				// check if there is a point close by in the reconstruction already
				if(err < REPROJECTION_ERROR){
					refinedReco.add(recoProposal);
				}
			}
			numUsedBeforeMatching += refinedReco.size();
			totalPoints += recons.size();
			reconstructedPoints.set(v, refinedReco);
		}// end refinement one single view
		System.out.println("After suppresion using "+
				String.valueOf(numUsedBeforeMatching)+" out of "+ Integer.valueOf(totalPoints)+".");
		
		// matching and suppression
		ArrayList<PointND> finalRecon = new ArrayList<PointND>();
		for(int v = 0; v < views.size(); v++){
			ArrayList<PointND> recons = reconstructedPoints.get(v);
			for(int i = 0; i < recons.size(); i++){
				PointND recoProposal = recons.get(i);
				ArrayList<PointND> same = new ArrayList<PointND>();
				same.add(recoProposal);
				for(int j = v+1; j < views.size(); j++){
					ArrayList<PointND> reduced = new ArrayList<PointND>();
					for(int k = 0; k < reconstructedPoints.get(j).size(); k++){
						PointND testPoint = reconstructedPoints.get(j).get(k);
						double eucDist = testPoint.euclideanDistance(recoProposal);
						if(eucDist < SUPPRESSION_RADIUS){
							same.add(testPoint);
						}else{
							reduced.add(testPoint);
						}
					}
					reconstructedPoints.set(j, reduced);
				}
				SimpleVector average = new SimpleVector(3);
				for(int j = 0; j < same.size(); j++){
					average.add(same.get(j).getAbstractVector());
				}
				average.divideBy(same.size());
				finalRecon.add(new PointND(average.copyAsDoubleArray()));					
			}

		}
		System.out.println("After exhaustive merging using "+ 
				String.valueOf(finalRecon.size())+" out of "+ Integer.valueOf(totalPoints)+".");
				
		return finalRecon;
	}
		
	/**
	 * Calculates the error of a 3D point given the segmentations / observations in all views.
	 * The error is defined as the second largest reprojection error in all segmentations.
	 * The definition allows for erroneous segmentations, 
	 * meaning vessel structures not segmented in several views or erroneously segmented structures, such as parts of 
	 * catheters.
	 * @param p3D
	 * @param fromView
	 * @return
	 */
	private double calculatePointError(PointND p3D, int fromView){
		float[] err = new float[views.size()-1];
		int idx = 0;
		for(int v = 0; v < views.size(); v++){
			if(v != fromView){
				PointND projectedInView = views.get(v).project(p3D);
				err[idx] = InterpolationOperators.interpolateLinear(views.get(v).getProjection(),
						projectedInView.get(0), projectedInView.get(1));
				idx++;
			}
		}
		Arrays.sort(err);
		
		return err[err.length-2];	
	}
		
	private void reconstructWithReferenceViews() {		
		
		if(Configuration.getGlobalConfiguration() == null){
			Configuration.loadConfiguration();
		}
		
		ExecutorService executorService = Executors.newFixedThreadPool(
				Integer.valueOf(Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.MAX_THREADS)));
		Collection<Future<?>> futures = new LinkedList<Future<?>>();
		
		for(int i = 0; i < pMats.length; i++){
			final int count = i;
			futures.add(
				executorService.submit(new Runnable() {
					@Override
					public void run() {						
						System.out.println( "Working on reconstruction " + Integer.valueOf(count+1) + 
											" of " + pMats.length+".");
						GraphCutCostMapRecon recon = 
								new GraphCutCostMapRecon(dataset,costMap,count,skels.get(count),pMats,primAng);
						recon.setDebug(false);
						ArrayList<PointND> pts = recon.reconstruct();
						
						reconstructedPoints.set(count, pts);
					}
				}) // exe.submit()
			); // futures.add()							
		}
		for (Future<?> future : futures){
		   try{
		       future.get();
		   }catch (InterruptedException e){
		       throw new RuntimeException(e);
		   }catch (ExecutionException e){
		       throw new RuntimeException(e);
		   }
		}		
	}
		
	public void toFile(String filename){
		
		ArrayList<Double> radii = new ArrayList<Double>();
		for(int i = 0; i < finalRecon.size(); i++){
			radii.add(1.0);
		}
		
		PointAndRadiusIO pointsIO = new PointAndRadiusIO();
		pointsIO.write(filename, finalRecon, radii);
	}
	
	public void displayProjections(ArrayList<PointND> pts){
		Grid3D proj = new Grid3D(this.projections);
		ImagePlus imp = ImageUtil.wrapGrid3D(proj,"");
		RoiManager manager = new RoiManager();
		int count = 0; 
		for(int v = 0; v < views.size(); v++){
			for(int i = 0; i < pts.size(); i++){
				PointND p = views.get(v).project(pts.get(i));
				PointRoi pRoi = new PointRoi(p.get(0),p.get(1));
				pRoi.setPosition(v+1);
				manager.add(imp, pRoi, count);
				count++;
			}
		}
		imp.getProcessor().setMinAndMax(0, 10);
		new ImageJ();
		imp.show();
		Prefs.showAllSliceOnly = true;
		manager.runCommand("Show All");
	}
	
	public void setExhaustiveReconParameters(double reprojErr, double suppressionRad){
		this.REPROJECTION_ERROR = reprojErr;
		this.SUPPRESSION_RADIUS = suppressionRad;
	}
	
}
