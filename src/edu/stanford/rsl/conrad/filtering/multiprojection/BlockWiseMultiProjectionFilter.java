package edu.stanford.rsl.conrad.filtering.multiprojection;

import ij.process.FloatProcessor;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.filtering.multiprojection.blocks.ImageProcessingBlock;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.UserUtil;


/**
 * 
 * Class for simplified parallelization of MultiProjectionFilters. The idea is to divide the problem into blocks which can be processed in parallel independent of each other. Performance scales much better than parallel processing of volumes in each step on CPUs.
 * Similar to the processing performed for parallelization by GPU.
 * 
 * @author akmaier
 *
 */
public abstract class BlockWiseMultiProjectionFilter extends MultiProjectionFilter {

	protected ImageProcessingBlock modelBlock;
	protected ImageProcessingBlock [][][] blocks;
	protected int numBlocks = CONRAD.getNumberOfThreads();
	protected int blocksX = 0;
	protected int blocksY = 0;
	protected int blocksZ = 0;
	protected int blockSizeX = 0;
	protected int blockSizeY = 0;
	protected int blockSizeZ = 0;
	protected int width = 0;
	protected int height = 0;
	protected int[] blockOverlap = {0, 0, 0};
	protected boolean initBlocks = false;
	protected int nInputImages = 0;
	protected CountDownLatch latch;

	public BlockWiseMultiProjectionFilter(){
		context = 55;
	}

	protected synchronized void initBlocks (){
		if (!initBlocks){
			blocksX = 1;
			blocksY = 1;
			blocksZ = 1;
			int blockSpread = numBlocks;
			while (blockSpread % 2 == 0){
				if (blockSpread % 2 == 0){
					blockSpread /= 2;
					blocksX *= 2;
				}
				if (blockSpread % 2 == 0){
					blockSpread /= 2;
					blocksY *= 2;
				}
				if (blockSpread % 2 == 0){
					blockSpread /= 2;
					blocksZ *= 2;
				}
			}
			int distributed = blocksX * blocksY * blocksZ;
			double factor = Math.ceil(((double) distributed) / ((double) numBlocks));
			blocksX *= factor;
			distributed = blocksX * blocksY * blocksZ;
			latch = new CountDownLatch(distributed);
			width = inputQueue.get(0).getWidth();
			height = inputQueue.get(0).getHeight();
			nInputImages = inputQueue.size();
			blockSizeX = (int) Math.ceil(((double)width) / blocksX);
			blockSizeY = (int) Math.ceil(((double)height) / blocksY);
			blockSizeZ = (int) Math.ceil(((double)nInputImages) / blocksZ);
			if (debug > 0) System.out.println("Processing as " + blocksX + "x" + blocksY + "x" + blocksZ +" ("+distributed+") grid. Block dimension is " + blockSizeX + "x" + blockSizeY + "x"+ blockSizeZ);
			blockSizeX = (int) Math.ceil((((double)width) / blocksX) );
			blockSizeY = (int) Math.ceil((((double)height) / blocksY) );
			blockSizeZ = (int) Math.ceil((((double)nInputImages) / blocksZ) );
			blocks = new ImageProcessingBlock[blocksX][blocksY][blocksZ];
			initBlocks = true;
		}
	}


	/**
	 * 
	 */
	private static final long serialVersionUID = -8702779450295098767L;

	@Override
	protected void processProjectionData(int projectionNumber) throws Exception {
		if (debug > 2) System.out.println("called: " + projectionNumber + " " + nInputImages);
		if (isLastBlock(projectionNumber)) {
			initBlocks();
			int contextX = blockOverlap[0];
			int blockXMaxIndex = blockSizeX + 2 * contextX;
			if (blocksX == 1) {
				contextX = 0;
				blockXMaxIndex = blockSizeX;
			}
			int contextY = blockOverlap[1];
			int blockYMaxIndex = blockSizeY + 2 * contextY;
			if (blocksY == 1) {
				contextY = 0;
				blockYMaxIndex = blockSizeY;
			}
			int contextZ = blockOverlap[2];
			int blockZMaxIndex = blockSizeZ + 2 * contextZ;
			if (blocksZ == 1) {
				contextZ = 0;
				blockZMaxIndex = blockSizeZ;
			}
			int zCoordinate = (int)Math.floor(((double)projectionNumber) / blockSizeZ);
			for (int z = 0; z <=zCoordinate; z ++) {
				if (blocks[0][0][z] == null) { // set up new batch of blocks
					for (int x = 0; x < blocksX; x++){
						for (int y = 0; y < blocksY; y++){
							blocks[x][y][z] = modelBlock.clone();
							blocks[x][y][z].setLatch(latch);
							int offsetZ = 0;
							if (z > 0) offsetZ = contextZ;
							if (((z)*blockSizeZ)+blockZMaxIndex > nInputImages) {
								offsetZ = ((z)*blockSizeZ)+blockZMaxIndex - nInputImages;
							}
							if (debug > 1) System.out.println("z offset: " + offsetZ +" "+ z+" " + blockZMaxIndex);
							Grid3D block = new Grid3D(blockXMaxIndex, blockYMaxIndex, blockZMaxIndex);
							
							for (int k = 0; k< blockZMaxIndex; k++) {
								Grid2D referenceSlice = inputQueue.get((z*blockSizeZ) + k - offsetZ);
								float [] pixels = new float [blockXMaxIndex * blockYMaxIndex];
								int offsetX = 0;
								int offsetY = 0;
								if (x > 0) offsetX = contextX;
								if (((x)*blockSizeX) + blockXMaxIndex > referenceSlice.getWidth()){
									offsetX  = ((x)*blockSizeX) + blockXMaxIndex - referenceSlice.getWidth();
								}
								if (y > 0) offsetY = contextY;
								if (((y)*blockSizeY) + blockYMaxIndex > referenceSlice.getHeight()){
									offsetY  = ((y)*blockSizeY) + blockYMaxIndex - referenceSlice.getHeight();
								}
								for (int j = 0; j < blockYMaxIndex; j++){
									for (int i = 0; i < blockXMaxIndex; i++){
										pixels[(j*blockXMaxIndex) + i] = referenceSlice.getPixelValue((blockSizeX * (x)) + i - offsetX, (blockSizeY * (y)) +  j - offsetY);
									}
								}
								block.setSubGrid(k, new Grid2D(pixels, blockXMaxIndex, blockYMaxIndex));
							}
							
							if (debug > 2) block.show();
							blocks[x][y][z].setInputBlock(block);
						}
					}
					if (debug > 0) System.out.println("Spawned blocks for z coordinate" + z);
					// remove images from input stack if no longer required
					for (int k = 0; k< blockSizeZ; k++) {
						if ((z*blockSizeZ) + k < (nInputImages - ((context *2) + 1))){
							if ((((z*blockSizeZ) + k < (nInputImages - (blockZMaxIndex))))||(z == blocksZ -1)) {
								int index = (z*blockSizeZ) + k -contextZ;
								if ((index < nInputImages)&&(index >= 0)){
									inputQueue.remove(index);
								}
							}
						}
					}
				}
			}
			if (debug > 2) System.out.println("executed: "+ projectionNumber + " " + nInputImages);
			if (projectionNumber == nInputImages - 1){ // last projection;
				ExecutorService e = Executors.newFixedThreadPool(numBlocks);
				CONRAD.setUseGarbageCollection(false);
				// remove remaining images from input queue;
				for (int z = (nInputImages - ((context *2) + 1)); z < nInputImages; z++){
					inputQueue.remove(z);
				}
				for (int z = 0; z < blocksZ; z ++) {
					for (int x = 0; x < blocksX; x++){
						for (int y = 0; y < blocksY; y++){
							e.submit(blocks[x][y][z]);
						}
					}
				}
				// wait for everything to be finished.
				if (debug > 0) System.out.println("Waiting for threads ...");
				while (latch.getCount() > 0){
					Thread.sleep(CONRAD.INVERSE_SPEEDUP);
					//System.out.println(latch.getCount());
				}
				e.shutdownNow();
				CONRAD.setUseGarbageCollection(true);
				e = null;
				if (debug > 0) System.out.println("Assembling Data ...");
				// put everything again together.
				for (int z = 0; z < blocksZ; z ++) {
					int offsetZ = 0;
					if (z > 0) offsetZ = contextZ;
					if (((z)*blockSizeZ)+blockZMaxIndex > nInputImages) {
						offsetZ = ((z)*blockSizeZ)+blockZMaxIndex - nInputImages;
					}
					for (int k = offsetZ; k < blockZMaxIndex; k++) {
						float [] pixels = new float [width * height];

						for (int x = 0; x < blocksX; x++){
							int offsetX = 0;
							if (x > 0) offsetX = contextX;
							if (((x)*blockSizeX) + blockXMaxIndex > width){
								offsetX  = ((x)*blockSizeX) + blockXMaxIndex - width;
							}
							for (int y = 0; y < blocksY; y++){
								int offsetY = 0;
								if (y > 0) offsetY = contextY;
								if (((y)*blockSizeY) + blockYMaxIndex > height){
									offsetY  = ((y)*blockSizeY) + blockYMaxIndex - height;
								}
								
								Grid2D current = blocks[x][y][z].getOutputBlock().getSubGrid(k);
								for (int j = offsetY; j < blockYMaxIndex; j++){
									for (int i = offsetX; i < blockXMaxIndex; i++){
										pixels[(((blockSizeY * (y)) +  j - offsetY)*width) + (blockSizeX * (x)) + i - offsetX] = current.getPixelValue(i, j);
									}
								}

							}
						}
						FloatProcessor filtered = new FloatProcessor(width, height);
						filtered.setPixels(pixels);
						sink.process(new Grid2D((float[])filtered.getPixels(), filtered.getWidth(), filtered.getHeight()), (z*blockSizeZ) + k - offsetZ);
					}
				}
			}
		}
	}

	@Override
	public void configure() throws Exception{
		numBlocks = UserUtil.queryInt("Number of simultaneous processing blocks: ", numBlocks);
		
		double[] overlap = { this.blockOverlap[0], this.blockOverlap[1], this.blockOverlap[2]};
		overlap = UserUtil.queryArray("Block overlap", overlap);
		this.blockOverlap[0] = (int) overlap[0];
		this.blockOverlap[1] = (int) overlap[1];
		this.blockOverlap[2] = (int) overlap[2];
	}

	@Override
	public void prepareForSerialization(){
		super.prepareForSerialization();
		blocks = null;
		latch = null;
		if (modelBlock != null) modelBlock.prepareForSerialization();
	}
}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
