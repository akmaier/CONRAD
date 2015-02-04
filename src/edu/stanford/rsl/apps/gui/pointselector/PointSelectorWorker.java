package edu.stanford.rsl.apps.gui.pointselector;

import ij.IJ;
import ij.ImageListener;
import ij.ImagePlus;
import ij.gui.PointRoi;
import ij.gui.Roi;
import ij.plugin.frame.RoiManager;
import ij.process.FloatPolygon;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.utils.Configuration;


public class PointSelectorWorker implements ImageListener {


	ArrayList<ArrayList<double[]>> pointCorrs;
	RoiManager roiman;
	private ImagePlus image;

	private boolean configured = false;

	public PointSelectorWorker() {
		pointCorrs = new ArrayList<ArrayList<double[]>>();
		roiman = null;
		configured = false;
	}


	public void configure() throws Exception {
		roiman = (RoiManager.getInstance() != null) ? RoiManager.getInstance() : new RoiManager();
		image = IJ.getImage();
		if (image != null){
			if(roiman != null){
				configured = true;
				for (Roi x : roiman.getRoisAsArray()) {
					if (!(x instanceof PointRoi)) {
						configured = false;
						break;
					}
				}
			}
			else
				throw new UnsupportedOperationException("Selected Points not Valid");
		}
		else
			throw new UnsupportedOperationException("No valid image");
	}


	public boolean evaluate() {
		boolean success = true;
		if (!configured)
			try {
				configure();
			} catch (Exception e) {
				success = false;
				e.printStackTrace();
			}
		if (configured) {
			ArrayList<double[]> pcorr = new ArrayList<double[]>(); 
			int i =0;
			System.out.println("Adding Point Correspondences");
			System.out.println("----------------------------");
			for (Roi r : roiman.getRoisAsArray()) {
				roiman.select(image, roiman.getRoiIndex(r));
				pcorr.add(new double[] {r.getFloatPolygon().xpoints[0], r.getFloatPolygon().ypoints[0], (float)r.getPosition()-1});
				System.out.println("Adding Point Correspondences");
				System.out.println("Point " + i + ": " + pcorr.get(i));
				i++;
				//roiman.runCommand("Measure");
			}
			if (pcorr.size() > 0)
				pointCorrs.add(pcorr);
			else
				success = false;
		}
		return success;
	}


	public void removePointSet(int label){
		pointCorrs.remove(label);
	}

	public void removeAllPointSets(){
		pointCorrs.clear();
	}

	public ArrayList<double[]> getPointSet(int label){
		return pointCorrs.get(label);
	}

	public int getNumberOfPointSets(){
		return pointCorrs.size();
	}

	public ArrayList<ArrayList<double[]>> getAllPointSets(){
		return pointCorrs;
	}

	public void setAllPointSets(Object psets){
		if (pointCorrs.getClass().isAssignableFrom(psets.getClass())){
			ArrayList<double[]> tmp = new ArrayList<double[]>(0);
			if (tmp.getClass().isAssignableFrom(((ArrayList<?>)psets).get(0).getClass())){
				pointCorrs = (ArrayList<ArrayList<double[]>>)psets;
			}
			else{
				pointCorrs = new ArrayList<ArrayList<double[]>>(1);
				tmp = (ArrayList<double[]>)psets;
				
				Trajectory traj = Configuration.getGlobalConfiguration().getGeometry();
				double[] voxelSizes = traj.getReconVoxelSizes();
				double[] originInPx = new double[]{traj.getOriginInPixelsX(), traj.getOriginInPixelsY(), traj.getOriginInPixelsZ()};
				for (int i = 0; i < tmp.size(); i++) {
					for (int j = 0; j < tmp.get(i).length; j++) {
						tmp.get(i)[j]=tmp.get(i)[j]/voxelSizes[j]+originInPx[j];
					}
				}
				pointCorrs.add(tmp);
			}
		}
	}


	public void setRoiManagerPointSet(int pointSet){
		if(roiman != null){
			roiman.close();
		}

		roiman = new RoiManager();
		for (int i = 0; i < pointCorrs.get(pointSet).size(); i++) {
			double[] point = pointCorrs.get(pointSet).get(i);
			PointRoi pr = new PointRoi(point[0],point[1]);
			pr.setPosition((int)point[2]+1);
			roiman.addRoi(pr);
		}
		roiman.runCommand("Show All");

	}

	@Override
	public String toString() {
		return "Select point correspondences over projections";
	}

	public boolean isConfigured() {
		return configured;
	}


	@Override
	public void imageClosed(ImagePlus arg0) {
		if (arg0.equals(image))
		{
			configured = false;
			image = null;
		}

	}


	@Override
	public void imageOpened(ImagePlus arg0) {
		// TODO Auto-generated method stub
	}


	@Override
	public void imageUpdated(ImagePlus arg0) {
		// TODO Auto-generated method stub
	}



}

/*
 * Copyright (C) 2010-2014 - Martin Berger 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
