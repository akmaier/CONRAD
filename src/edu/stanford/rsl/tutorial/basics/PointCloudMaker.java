package edu.stanford.rsl.tutorial.basics;

import java.util.ArrayList;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.apps.gui.opengl.PointCloudViewer;

public class PointCloudMaker{
	
	
	Grid3D image;
	
	public PointCloudMaker(Grid3D image) {
		this.image = image;
	}
	
	public ArrayList<PointND> getPoints(int id){
		ArrayList<PointND> points = new ArrayList<PointND>();
		int [] size = image.getSize();
		for (int k=0; k <size[2]; k++){
			for (int j=0; j <size[1]; j++){
				for (int i=0; i <size[0]-1; i++){
					float one = image.getAtIndex(i, j, k);
					float two = image.getAtIndex(i+1, j, k);
					if (one != two) {
						PointND point = null;
						if (one == id){
							point = new PointND(General.voxelToWorld(new int [] {i,j,k}, image.getSpacing(), image.getOrigin()));
						}
						if (two == id){
							point = new PointND(General.voxelToWorld(new int [] {i+1,j,k}, image.getSpacing(), image.getOrigin()));
						}
						if (point != null) points.add(point);
					}
				}	
			}	
		}
		return points;
	}

	public static void main(String[] args){
		new ImageJ();
		MHDImageLoader loader = new MHDImageLoader();
		
		Grid3D image = loader.loadImage(args[0]);
		image.show("Loaded Data");
		PointCloudMaker ptsMaker = new PointCloudMaker(image);
		ArrayList<PointND> id1 = ptsMaker.getPoints(1);
		ArrayList<PointND> id2 = ptsMaker.getPoints(2);
		id2.addAll(id1);
		PointCloudViewer pcv = new PointCloudViewer("ID 1", id2);
		pcv.setVisible(true);
	}

}
