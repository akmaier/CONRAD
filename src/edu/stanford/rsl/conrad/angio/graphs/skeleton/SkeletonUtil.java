/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.graphs.skeleton;

import java.util.ArrayList;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import ij.process.ImageConverter;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.VesselBranch;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.VesselTree;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.MinimumSpanningTree;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.BranchPoint;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.Point;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.SkeletonBranch;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.SkeletonInfo;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.Skeletonize;

public class SkeletonUtil {

	
	public static ArrayList<Edge> vesselTreeToEdgeList(VesselTree vessels){
		ArrayList<Edge> list = new ArrayList<Edge>();
		for(int i = 0; i < vessels.size(); i++){
			VesselBranch vb = vessels.get(i);
			for(int j = 0; j < vb.size()-1; j++){
				PointND bp1 = new PointND(vb.get(j).getPhysCoordinates());
				PointND bp2 = new PointND(vb.get(j+1).getPhysCoordinates());
				list.add(new Edge(bp1,bp2));
			}
		}
		return list;
	}
	
	public static Skeleton vesselTreeToSkeleton(VesselTree vt){
		Skeleton skel = new Skeleton();
		for(int i = 0; i < vt.size(); i++){
			VesselBranch vb = vt.get(i);
			SkeletonBranch sb = new SkeletonBranch();
			for(int j = 0; j < vb.size(); j++){
				sb.add(vb.get(j));
			}
			skel.add(sb);
		}
		return skel;
	}
	
	public static ArrayList<Skeleton> vesselTreesToSkeletons(ArrayList<VesselTree> vts){
		ArrayList<Skeleton> skels = new ArrayList<Skeleton>();
		for(int k = 0; k < vts.size(); k++){
			skels.add(vesselTreeToSkeleton(vts.get(k)));
		}
		return skels;
	}
	
	public static Skeleton binaryImgToSkel(Grid2D img, double prune, boolean show){
		ImagePlus toBeSkeletonized = new ImagePlus();
		toBeSkeletonized.setProcessor(ImageUtil.wrapGrid2D(img));
		ImageConverter sc = new ImageConverter(toBeSkeletonized);
		sc.convertToGray8();
		
		Skeletonize skeletonizer = new Skeletonize(false);
		ImagePlus sFull = skeletonizer.run(toBeSkeletonized);
		
		SkeletonInfo skInfo = new SkeletonInfo();
		skInfo.run(sFull,false,true,false);
		
		Skeleton skel1 = new Skeleton(prune);
		skel1.run(skInfo);
		
		Skeleton skel = new Skeleton(prune);
		for(int i = 0; i < skel1.size(); i++){
			SkeletonBranch sb = skel1.get(i);
			if(!sb.isEmpty()){
				skel.add(sb);
			}
		}
			
		ImageStack stack = toBeSkeletonized.getImageStack();		
		ImageStack skeletonStack = new ImageStack(stack.getWidth(), stack.getHeight());
		ImagePlus toBeVisualized = new ImagePlus();
		toBeVisualized.setProcessor(stack.getProcessor(1));
		ImageStack proc = new Skeleton().visualize(toBeVisualized, skel, false);
			
		if(show){
			skeletonStack.addSlice(proc.getProcessor(1));
			ImagePlus fire = new ImagePlus();
			fire.setTitle("Skeletonized segmentations.");
			fire.setStack(skeletonStack);
			fire.show();
			IJ.run(fire, "Fire", null);
		}
		
		return skel;
	}
	
	public static Grid2D binaryImgToSkelimage(Grid2D img, double prune){
		ImagePlus toBeSkeletonized = new ImagePlus();
		toBeSkeletonized.setProcessor(ImageUtil.wrapGrid2D(img));
		ImageConverter sc = new ImageConverter(toBeSkeletonized);
		sc.convertToGray8();
		
		Skeletonize skeletonizer = new Skeletonize(false);
		ImagePlus sFull = skeletonizer.run(toBeSkeletonized);
		
		SkeletonInfo skInfo = new SkeletonInfo();
		skInfo.run(sFull,false,true,false);
		
		Skeleton skel1 = new Skeleton(prune);
		skel1.run(skInfo);
		
		Skeleton skel = new Skeleton(prune);
		for(int i = 0; i < skel1.size(); i++){
			SkeletonBranch sb = skel1.get(i);
			if(!sb.isEmpty()){
				skel.add(sb);
			}
		}
			
		ImageStack stack = toBeSkeletonized.getImageStack();		
		ImageStack skeletonStack = new ImageStack(stack.getWidth(), stack.getHeight());
		ImagePlus toBeVisualized = new ImagePlus();
		toBeVisualized.setProcessor(stack.getProcessor(1));
		ImageStack proc = new Skeleton().visualize(toBeVisualized, skel, false);
			
		skeletonStack.addSlice(proc.getProcessor(1));
		ImagePlus fire = new ImagePlus();
		fire.setTitle("Skeletonized segmentations.");
		fire.setStack(skeletonStack);		
		return ImageUtil.wrapImagePlus(fire).getSubGrid(0);
	}
	
	public static Skeleton binaryImgToSkel(Grid3D img, double prune, boolean show){
		ImagePlus toBeSkeletonized = ImageUtil.wrapGrid3D(img,"");
		ImageConverter sc = new ImageConverter(toBeSkeletonized);
		sc.convertToGray8();
		
		Skeletonize skeletonizer = new Skeletonize(false);
		ImagePlus sFull = skeletonizer.run(toBeSkeletonized);
		
		SkeletonInfo skInfo = new SkeletonInfo(false);
		skInfo.run(sFull,false,true,false);
		
		Skeleton skel = new Skeleton(prune);
		skel.run(skInfo);		
			
		ImageStack stack = sFull.getImageStack();		
		ImageStack skeletonStack = new ImageStack(stack.getWidth(), stack.getHeight());
		ImagePlus toBeVisualized = new ImagePlus();
		toBeVisualized.setStack(stack);
		ImageStack proc = new Skeleton().visualize(toBeVisualized, skel, false);
			
		if(show){
			for(int i = 0; i < img.getSize()[2]; i++){
				skeletonStack.addSlice(proc.getProcessor(i+1));
			}
			ImagePlus fire = new ImagePlus();
			fire.setTitle("Skeletonized segmentations.");
			fire.setStack(skeletonStack);
			fire.show();
			IJ.run(fire, "Fire", null);
		}
		
		return skel;
	}
	
	public static Grid3D costMapToVesselTreeImage(Grid3D costMap){
		int[] gSize = costMap.getSize();
		Grid3D vtImage = new Grid3D(costMap);
		for(int k = 0; k < gSize[2]; k++){
			for(int i = 0; i < gSize[0]; i++){
				for(int j = 0; j < gSize[1]; j++){
					if(costMap.getAtIndex(i, j, k) == 0){
						vtImage.setAtIndex(i, j, k, 1);
					}else{
						vtImage.setAtIndex(i, j, k, 0);
					}
				}
			}
		}
		return vtImage;
	}
	
	public static Grid2D skelToBinaryImg(Grid2D img, Skeleton skel){
		Grid2D g = new Grid2D(img.getSize()[0], img.getSize()[1]);
		g.setSpacing(img.getSpacing());
		g.setOrigin(img.getOrigin());
		for(int i = 0; i < skel.size(); i++){
			SkeletonBranch sb = skel.get(i);
			for(int j = 0; j < sb.size(); j++){
				g.setAtIndex(sb.get(j).x, sb.get(j).y, 1.0f);
			}
		}
		return g;
	}
	
	public static Grid3D vesselTreeToBinary(Grid3D img, ArrayList<ArrayList<VesselTree>> vts){
		int[] gSize = img.getSize();
		Grid3D vI = new Grid3D(gSize[0],gSize[1],gSize[2]);
		vI.setOrigin(img.getOrigin());
		vI.setSpacing(img.getSpacing());
		for(int k = 0; k < gSize[2]; k++){
			for(int i = 0; i < vts.size(); i++){
				VesselTree vt = vts.get(i).get(k);
				for(int j = 0; j < vt.size(); j++){
					VesselBranch vb = vt.get(j);
					for(int l = 0; l < vb.size(); l++){
						vI.setAtIndex(vb.get(l).x, vb.get(l).y, k, 2.0f);
					}
				}
				
			}
		}
		return vI;
	}
	
	public static ArrayList<Point> endPointsFromSkel(Skeleton skel){
		ArrayList<Point> endPts = new ArrayList<Point>();
		for(int i = 0; i < skel.size(); i++){
			SkeletonBranch sb = skel.get(i);
			for(int j = 0; j < sb.size(); j++){
				BranchPoint p = sb.get(j);
				if(p.isEND() && !p.isJUNCTION()){
					endPts.add(new Point(p.x,p.y,0));
				}
			}
		}
		return endPts;
	}
	
	public static ArrayList<PointND> binaryImageToPointList(Grid3D img){
		ArrayList<PointND> pts = new ArrayList<PointND>();
		int[] gSize = img.getSize();
		for(int k = 0; k < gSize[2]; k++){
			for(int i = 0; i < gSize[0]; i++){
				for(int j = 0; j < gSize[1]; j++){
					if(img.getAtIndex(i, j, k) > 0){
						pts.add(new PointND(i,j,k));
					}
				}
			}
		}
		return pts;
	}
	
	public static Point determineStartPoint(ArrayList<PointND> list){
		MinimumSpanningTree mst = new MinimumSpanningTree(list, 2);
		mst.run();
		ArrayList<ArrayList<PointND>> conComp = mst.getConnectedComponents();
		int idx = 0;
		int maxSize = 0;
		for(int i = 0; i < conComp.size(); i++){
			if(conComp.get(i).size() > maxSize){
				maxSize = conComp.get(i).size();
				idx = i;
			}
		}
		PointND p;
		Point sp = null;
		for(int i = 0; i < conComp.get(idx).size(); i++){
			p = conComp.get(idx).get(i);
			sp = new Point((int)(p.get(0)),(int)(p.get(1)),0);
			if(sp.x > 1){
				if(sp.y > 1){
					return sp;
				}				
			}
		}
		
		return sp;
	}
	
}
