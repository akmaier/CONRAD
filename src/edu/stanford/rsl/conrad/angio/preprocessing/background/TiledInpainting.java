/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.preprocessing.background;

import java.util.Collection;
import java.util.LinkedList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.plugin.filter.RankFilters;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.StackConverter;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.RegKeys;
import edu.stanford.rsl.conrad.angio.preprocessing.segmentation.morphological.Morphology;
import edu.stanford.rsl.conrad.angio.preprocessing.segmentation.morphological.tools.StructuringElement;
import edu.stanford.rsl.conrad.angio.util.apodization.BlackmanWindow;

public class TiledInpainting {
	
	private Grid3D origNonFilled = null;
	private Grid3D original = null;
	private Grid3D mask = null;
	private Grid3D result = null;
	
	private boolean SMOOTH_RESULT = true;
	private static final int averageKernelSize = 5;
	private static final int halfKernelSize = averageKernelSize / 2;
	
	private int usedSize = 32;//pSize/2;
	private int borderSize = 64;//pSize/4;
	private int pSize = 0;
	
	private Grid2D apoWindow = null;
	private boolean USE_APODIZATION = true;
	private boolean ERODE_MASK = true;
	private int openingSize = 0;
	private int dilationSize = 2;
	
	private static final int numIterations = 250;
	private Number[] param = null;
	
	private boolean has_defect = false;
	
	
	public static void main(String[] agrs){

		String[] projFiles = new String[2];
		String[] maskFiles = new String[2];
		String[] inpaintedFiles = new String[2];
		String[] dsaFiles = new String[2];
		
		String dir   = ".../";
		
		projFiles[0] = dir + "beats_7/contrast.tif";	
		maskFiles[0] = dir + "beats_7/seg.tif";	
		inpaintedFiles[0]  = dir + "beats_7/inpainted.tif";
		dsaFiles[0]  = dir + "beats_7/dsa.tif";
		
		projFiles[1] = dir + "static/contrast.tif";
		maskFiles[1] = dir + "static/seg.tif";
		inpaintedFiles[1]  = dir + "static/inpainted.tif";
		dsaFiles[1]  = dir + "static/dsa.tif";
		
//		projFiles[1] = dir + "Noise_Level_2/c6_b020/contrast.tif";
//		maskFiles[1] = dir + "Noise_Level_2/c6_b020/seg.tif";
//		inpaintedFiles[1]  = dir + "Noise_Level_2/c6_b020/inpainted.tif";
//		dsaFiles[1]  = dir + "Noise_Level_2/c6_b020/dsa.tif";
//		
//		projFiles[2] = dir + "Noise_Level_1/c5_b075/contrast.tif";
//		maskFiles[2] = dir + "Noise_Level_1/c5_b075/seg.tif";
//		inpaintedFiles[2]  = dir + "Noise_Level_1/c5_b075/inpainted.tif";
//		dsaFiles[2]  = dir + "Noise_Level_1/c5_b075/dsa.tif";
//		
//		projFiles[3] = dir + "Noise_Level_2/c5_b075/contrast.tif";
//		maskFiles[3] = dir + "Noise_Level_2/c5_b075/seg.tif";
//		inpaintedFiles[3]  = dir + "Noise_Level_2/c5_b075/inpainted.tif";
//		dsaFiles[3]  = dir + "Noise_Level_2/c5_b075/dsa.tif";
		
		for(int i = 0; i < projFiles.length; i++){
			String projFile = projFiles[i];
			String maskFile = maskFiles[i];
			String outFile = inpaintedFiles[i];
			String dsaFile = dsaFiles[i];
			
			System.out.println("On "+projFile);
		
			Grid3D proj = ImageUtil.wrapImagePlus(IJ.openImage(projFile));
			Grid3D mask = ImageUtil.wrapImagePlus(IJ.openImage(maskFile));
			
			new ImageJ();
			
			TiledInpainting inpaint = new TiledInpainting(proj, mask, 32, 64);
			Grid3D painted = inpaint.run();
			
			Grid3D masked = inpaint.getMaskedOutImage();
			
			IJ.saveAsTiff(ImageUtil.wrapGrid(painted, ""), outFile);
			IJ.saveAsTiff(ImageUtil.wrapGrid(masked, ""), dsaFile);
		}	
		
		//proj.show();
		//painted.show();
		//masked.show();
		
	}
	
	public TiledInpainting(Grid3D orig, Grid3D m){
		assert(sameSize(orig,m)) : new Exception("Image and mask must have the same size!");
		this.mask = erodeMask(m);
		this.origNonFilled = (Grid3D)orig.clone();
		this.original = applyMask(orig, mask);
		this.pSize = usedSize + 2*borderSize;
		
		Number[] p = new SpectralInterpolation().getParameters();
		p[0] = numIterations;
		p[1] = pSize;
		this.param = p;
	}
	
	public TiledInpainting(Grid3D orig, Grid3D m, int usedSiz, int borderSiz){
		assert(sameSize(orig,m)) : new Exception("Image and mask must have the same size!");
		this.mask = erodeMask(m);
		this.origNonFilled = (Grid3D)orig.clone();
		this.original = applyMask(orig, mask);
		this.usedSize = usedSiz;
		this.borderSize = borderSiz;
		this.pSize = usedSize + 2*borderSize;
		
		Number[] p = new SpectralInterpolation().getParameters();
		p[0] = numIterations;
		p[1] = pSize;
		this.param = p;
	}
	
	public Grid3D run(){
		System.out.println("Starting spectral inpainting on " + mask.getSize()[2] + " slices.");
		
		this.apoWindow = new BlackmanWindow(pSize, pSize).getWindow();
		
		int[] gSize = original.getSize();
		Grid3D appliedRun1 = new Grid3D(gSize[0],gSize[1],gSize[2]);
		appliedRun1.setSpacing(original.getSpacing());
		Grid3D appliedRun2 = new Grid3D(gSize[0],gSize[1],gSize[2]);
		appliedRun2.setSpacing(original.getSpacing());
		
		int nWindowsX = gSize[0] / usedSize;
		int nWindowsY = gSize[1] / usedSize;
		
		if(Configuration.getGlobalConfiguration() == null){
			Configuration.loadConfiguration();
		}
		ExecutorService executorService = Executors.newFixedThreadPool(
				Integer.valueOf(Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.MAX_THREADS)));
		Collection<Future<?>> futures = new LinkedList<Future<?>>();
				
		for(int sli = 0; sli < gSize[2]; sli++){
			final int k = sli;
			futures.add(
				executorService.submit(new Runnable() {
					@Override
					public void run() {						
				
						System.out.println("Working on slice "+String.valueOf(k+1)+" of "+gSize[2]+".");
						Grid2D imgSlice = original.getSubGrid(k);
						Grid2D maskSlice = mask.getSubGrid(k);
						// run 1
						for(int tx = 0; tx < nWindowsX; tx++){
							int tileStartX = tx * usedSize;
							for(int ty = 0; ty < nWindowsY; ty++){
								int tileStartY = ty * usedSize;
								Grid3D imgTile = getTileAtStartIdx(imgSlice,tileStartX, tileStartY);
								boolean[][][] maskTile = getMaskTileAtStartIdx(maskSlice, tileStartX, tileStartY);
								if(has_defect){
									if(USE_APODIZATION){
										imgTile = applyWindow(imgTile);
									}
									SpectralInterpolation specInt = new SpectralInterpolation(maskTile);
									specInt.setParameters(param);
									Grid3D appToTile = specInt.applyToGrid(imgTile);
									if(USE_APODIZATION){
										appToTile = removeWindow(appToTile);
									}
									tileToVolume(appliedRun1, appToTile, tileStartX, tileStartY, k);
								}else{
									tileToVolume(appliedRun1, imgTile, tileStartX, tileStartY, k);
								}
							}
						}
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
		
		for(int sli = 0; sli < gSize[2]; sli++){
			final int k = sli;
			futures.add(
				executorService.submit(new Runnable() {
					@Override
					public void run() {				
						System.out.println("Working on slice "+String.valueOf(k+1)+" of "+gSize[2]+".");
						Grid2D imgSlice = original.getSubGrid(k);
						Grid2D maskSlice = mask.getSubGrid(k);
						
						// run 2
						for(int tx = 0; tx < nWindowsX-1; tx++){
							int tileStartX = tx * usedSize + usedSize/2;
							for(int ty = 0; ty < nWindowsY-1; ty++){
								int tileStartY = ty * usedSize + usedSize/2;
								Grid3D imgTile = getTileAtStartIdx(imgSlice,tileStartX, tileStartY);
								boolean[][][] maskTile = getMaskTileAtStartIdx(maskSlice, tileStartX, tileStartY);
								if(has_defect){
									if(USE_APODIZATION){
										imgTile = applyWindow(imgTile);
									}
									SpectralInterpolation specInt = new SpectralInterpolation(maskTile);
									specInt.setParameters(param);
									Grid3D appToTile = specInt.applyToGrid(imgTile);
									if(USE_APODIZATION){
										appToTile = removeWindow(appToTile);
									}
									tileToVolume(appliedRun2, appToTile, tileStartX, tileStartY, k);
								}else{
									tileToVolume(appliedRun2, imgTile, tileStartX, tileStartY, k);
								}
							}
						}
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
					
		Grid3D applied = mergeRuns(appliedRun1, appliedRun2);
		applied = averageInpainted(applied);
		System.out.println("Done.");
		this.result = applied;
		return applied;
	}
	
	private Grid3D mergeRuns(Grid3D run1, Grid3D run2){
		int[] s = run1.getSize();
		Grid3D merged = (Grid3D)run1.clone();
		for(int k = 0; k < s[2]; k++){
			Grid2D sub1 = run1.getSubGrid(k);
			Grid2D sub2 = run2.getSubGrid(k);
			for(int i = 0; i < s[0]; i++){
				for(int j = 0; j < s[1];j++){
					if(mask.getAtIndex(i, j, k) != 0){
						float val = sub1.getAtIndex(i, j) + sub2.getAtIndex(i, j);
						merged.setAtIndex(i, j, k, val/2);
					}
				}
			}
		}
		return merged;
	}
	
	private Grid3D averageInpainted(Grid3D inpainted){
		int[] s = inpainted.getSize();
		RankFilters rf = new RankFilters();
		Grid3D avrg = new Grid3D(inpainted);
		for(int k = 0; k < s[2]; k++){
			Grid2D sub = inpainted.getSubGrid(k);
			for(int i = 0; i < s[0]; i++){
				for(int j = 0; j < s[1];j++){
					if(mask.getAtIndex(i, j, k) != 0){
						float val = sub.getAtIndex(i, j);
						if(SMOOTH_RESULT){
							val = filterAtPoint(sub, i, j);
						}
						avrg.setAtIndex(i, j, k, val);
					}
				}
			}
			ImagePlus ip = new ImagePlus("temp", ImageUtil.wrapGrid2D((Grid2D)mask.getSubGrid(k).clone()));
			ip = Morphology.dilate(ip, new StructuringElement("circle", dilationSize, false));
			Grid2D dilatedMask = ImageUtil.wrapFloatProcessor((FloatProcessor)ip.getProcessor().convertToFloat());
			
			if(SMOOTH_RESULT){
				FloatProcessor fpMed = ImageUtil.wrapGrid2D((Grid2D)avrg.getSubGrid(k).clone());
				rf.rank(fpMed, averageKernelSize, RankFilters.MEDIAN);
				Grid2D medianFiltered = ImageUtil.wrapFloatProcessor(fpMed);
				for(int i = 0; i < s[0]; i++){
					for(int j = 0; j < s[1];j++){
						if(dilatedMask.getAtIndex(i, j) != 0){
							float val = medianFiltered.getAtIndex(i, j);
							avrg.setAtIndex(i, j, k, val);
						}
					}
				}
			}
		}
		return avrg;
	}
	
	private float filterAtPoint(Grid2D sub, int i, int j){
		float val = 0;
		for(int x = 0; x < averageKernelSize; x++){
			int xx = i - halfKernelSize + x;
			if(xx > -1 && xx < sub.getSize()[0]){
				for(int y = 0; y < averageKernelSize; y++){
					int yy = j - halfKernelSize + y;
					if(yy > -1 && yy < sub.getSize()[1]){
						val += sub.getAtIndex(xx, yy);
					}
				}
			}
		}
		
		return val/averageKernelSize/averageKernelSize;
	}
	
	public Grid3D getMaskedOutImage(){
		assert(result != null) : new Exception("Run Inpainting first!");
		int[] s = original.getSize();
		Grid3D maskedOut = new Grid3D(s[0],s[1],s[2]);
		maskedOut.setSpacing(original.getSpacing());
		for(int k = 0; k < s[2]; k++){
			for(int i = 0; i < s[0]; i++){
				for(int j = 0; j < s[1];j++){
					if(mask.getAtIndex(i, j, k) != 0){
						float val = origNonFilled.getAtIndex(i, j, k) - result.getAtIndex(i, j, k);
						val = Math.max(val, 0);
						maskedOut.setAtIndex(i, j, k, val);
					}
				}
			}
		}
		return maskedOut;
	}

	
	private Grid3D removeWindow(Grid3D g){
		Grid3D rem = (Grid3D)g.clone();
		for(int k = 0; k < g.getSize()[2]; k++){
			for(int i = 0; i < usedSize; i++){
				int idxx = borderSize + i;
				for(int j = 0; j < usedSize; j++){
					int idxy = borderSize + j;
					rem.multiplyAtIndex(idxx, idxy, k, 1/apoWindow.getAtIndex(idxx, idxy));
				}
			}
		}
		return rem;
	}
	
	private Grid3D applyWindow(Grid3D g){
		int[] s = g.getSize();
		Grid3D apodized = new Grid3D(s[0],s[1],s[2]);
		
		for(int k = 0; k < s[2]; k++){
			for(int i = 0; i < s[0]; i++){
				for(int j = 0; j < s[1];j++){
					float val = g.getAtIndex(i, j, k)*apoWindow.getAtIndex(i, j);
					apodized.setAtIndex(i, j, k, val);
				}
			}
		}
		return apodized;		
	}
	
	/**
	 * Calculates the tile of the image grid of size pSize X pSize, the index passed to the method however is 
	 * the index of the first pixel that will be used, i.e. TileStartIndex+pSize/4 . The tile will range from 
	 * TileStartIndex to TileStartIndex+pSize . We assume mirroring boundary conditions.
	 * @param g
	 * @param x
	 * @param y
	 * @return
	 */
	private Grid3D getTileAtStartIdx(Grid2D g, int x, int y){
		Grid3D tile = new Grid3D(pSize,pSize,1);
		int[] gSize = g.getSize();
		
		for(int i = 0; i < pSize; i++){
			int ii = x-borderSize + i;
			if(ii < 0){
				ii = borderSize - 1 - i;
			}else if(ii >= gSize[0]){
				ii = gSize[0] - 1 - (ii - gSize[0]);
			}
			for(int j = 0; j < pSize; j++){
				int jj = y-borderSize + j;
				if(jj < 0){
					jj = borderSize - 1 - j;
				}else if(jj >= gSize[1]){
					jj = gSize[1] - 1 - (jj - gSize[1]);
				}
				float val = g.getAtIndex(ii, jj);
				tile.setAtIndex(i, j, 0, val);
			}
			
		}
		return tile;
	}
	
	private void tileToVolume(Grid3D volume, Grid3D tile, int x, int y, int z){
		for(int i = 0; i < usedSize; i++){
			int idxx = x + i;
			if(idxx < volume.getSize()[0]){
				for(int j = 0; j < usedSize; j++){
					int idxy = y + j;
					if(idxy < volume.getSize()[1]){
						volume.setAtIndex(idxx, idxy, z, tile.getAtIndex(borderSize + i, borderSize + j, 0));
					}
				}
			}
		}		
	}
	
	private boolean[][][] getMaskTileAtStartIdx(Grid2D g, int x, int y){
		boolean[][][] tile = new boolean[1][pSize][pSize];
		int[] gSize = g.getSize();
		boolean defective = false;
		
		for(int i = 0; i < pSize; i++){
			int ii = x-borderSize + i;
			if(ii < 0){
				ii = borderSize - 1 - i;
			}else if(ii >= gSize[0]){
				ii = gSize[0] - 1 - (ii - gSize[0]);
			}
			for(int j = 0; j < pSize; j++){
				int jj = y-borderSize + j;
				if(jj < 0){
					jj = borderSize - 1 - j;
				}else if(jj >= gSize[1]){
					jj = gSize[1] - 1 - (jj - gSize[1]);
				}
				float val = g.getAtIndex(ii, jj);
				if(val != 0){
					defective = true;
					tile[0][j][i] = true;
				}
			}
			
		}
		this.has_defect = defective;
		return tile;
	}
		
	private Grid3D erodeMask(Grid3D m){
		if(ERODE_MASK){
			int[] s = m.getSize();
			
			ImagePlus imp = ImageUtil.wrapGrid((Grid3D)m.clone(), "");
			StackConverter sC = new StackConverter(imp);
			sC.convertToGray8();
			
			StructuringElement structuringElementOpening = new StructuringElement("circle", openingSize, false);
			StructuringElement structuringElementDilation = new StructuringElement("circle", dilationSize, false);
			
			ImageStack stack = imp.getImageStack();
			for(int k = 0; k < s[2]; k++){
				ImageProcessor tempIp = stack.getProcessor(k+1);
				ImagePlus ip = new ImagePlus("temp", tempIp);
				ip = Morphology.open(ip, structuringElementOpening);
				ip = Morphology.dilate(ip, structuringElementDilation);
				stack.setProcessor(ip.getChannelProcessor(), k+1);
				
			}
			imp.setStack(stack);
			return ImageUtil.wrapImagePlus(imp);
		}else{
			return m;
		}
	}
	
	private Grid3D applyMask(Grid3D g, Grid3D m){
		Grid3D masked = (Grid3D) g.clone();
		int[] s = m.getSize();
		
		for(int k = 0; k < s[2]; k++){
			float mean = 0;//calculateSliceMean(g.getSubGrid(k), m.getSubGrid(k));
			for(int i = 0; i < s[0]; i++){
				for(int j = 0; j < s[1]; j++){
					if(m.getAtIndex(i, j, k) != 0){
						masked.setAtIndex(i, j, k, mean);
					}
				}
			}
		}
		return masked;
	}
	
		
	private boolean sameSize(Grid3D g1, Grid3D g2){
		int[] s1 = g1.getSize();
		int[] s2 = g2.getSize();
		for(int i = 0; i < s1.length; i++){
			if(s1[i]!=s2[i]){
				return false;
			}
		}
		return true;
	}
	
	
	public void setErodeMask(boolean val){
		this.ERODE_MASK = val;
	}
	
	public void setSmoothResult(boolean val){
		this.SMOOTH_RESULT = val;
	}
	
}


