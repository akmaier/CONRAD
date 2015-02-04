/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.io;


import ij.ImagePlus;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.pipeline.BufferedProjectionSink;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageGridBuffer;


/**
 * Class to model an ImagePlus projection source, i.e. stream projections from an instance of ImagePlus.
 * 
 * @author akmaier
 * @see ImagePlus
 *
 */
public class ImagePlusDataSink extends BufferedProjectionSink {

	/**
	 * 
	 */
	private static final long serialVersionUID = -931042578800415977L;
	private ImageGridBuffer imageBuffer;
	private Grid3D stack = null;
	private boolean closed = false;
	private boolean init = false;
	private boolean debug = false;


	@Override
	public String getName() {
		return "Image Plus Projection Data Sink";
	}

	@Override
	public Grid3D getResult() {
		//System.out.println("get result called: " + closed);
		while (!closed){
			try {
				Thread.sleep(CONRAD.INVERSE_SPEEDUP);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		//System.out.println("Image returned." + closed);
		adjustViewRange();
		return stack;
	}

	private synchronized void updateStack(Grid2D image, int projectionNumber){
		if (stack == null) {
			int stackSize = imageBuffer.size();
			stack = new Grid3D(image.getWidth(), image.getHeight(), stackSize, false);
		}
		stack.setSubGrid(projectionNumber, image);
	}

	@Override
	public void process(Grid2D projection, int projectionNumber)
	throws Exception {
		if ((!init) || (imageBuffer == null)){
			if (debug) System.out.println("ImagePlusDataSink: got projection " + projectionNumber);
			imageBuffer = new ImageGridBuffer();
			init = true;
			closed = false;
		}
		if (debug) System.out.println("ImagePlusDataSink: got projection " + projectionNumber + " buffersize " + imageBuffer.size());
		imageBuffer.add(projection, projectionNumber);
	}

	@Override
	public String toString() {
		return getName();
	}

	@Override
	public void configure() throws Exception {
		configured = true;
	}

	@Override
	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}

	@Override
	public void prepareForSerialization(){
		super.prepareForSerialization();
		stack = null;
		init = false;
		closed = false;
		imageBuffer = null;
	}

	@Override
	public void close() throws Exception {
		if (!closed) {
			if (debug) System.out.println("ImagePlusDataSink: Closing Stream at " + imageBuffer.size());
			Grid2D [] process = imageBuffer.toArray();
			int newWidth = process[0].getWidth();
			int newHeight = process[0].getHeight();
			for (int i=1; i < process.length; i++){
				int check = process[i].getWidth();
				newWidth = (check > newWidth) ? check: newWidth;
				check = process[i].getHeight();
				newHeight = (check > newHeight)? check: newHeight;
			}
			// make sure the thread pool is done
			for (int i = 0; i < process.length; i++){
				Grid2D ip = process[i];
				if (process[i] == null) {
					System.out.println("ImagePlusDataSink: Image " + i + " was null");
				}
				if ((ip.getHeight() != newHeight)|| (ip.getWidth() != newWidth)) {
					Grid2D newIP = new Grid2D(newWidth, newHeight);
					for(int k = 0; k < ip.getHeight();k++){
						for (int l= 0; l < ip.getWidth();l++){
							newIP.putPixelValue(l, k, ip.getPixelValue(l, k));
						}
					}
					ip = newIP;
				}
				updateStack(ip, i);
			}
			closed = true;
		}
	}

	@Override
	public void setConfiguration(Configuration config){
		// Nothing to do here.
	}

}
