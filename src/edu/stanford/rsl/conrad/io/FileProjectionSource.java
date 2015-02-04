package edu.stanford.rsl.conrad.io;

import java.awt.image.ColorModel;
import java.awt.image.IndexColorModel;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.zip.GZIPInputStream;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.pipeline.IndividualImagePipelineFilteringTool;
import edu.stanford.rsl.conrad.pipeline.ProjectionSource;
import edu.stanford.rsl.conrad.utils.CONRAD;
import ij.IJ;
import ij.LookUpTable;
import ij.Macro;
import ij.Prefs;
import ij.io.FileInfo;
import ij.io.ImageReader;
import ij.io.RandomAccessStream;
import ij.process.ByteProcessor;
import ij.process.ColorProcessor;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;

/**
 * Class to model an abstract projection source which accesses the file system to stream the data.
 * 
 * @author akmaier
 *
 */
public abstract class FileProjectionSource implements ProjectionSource {

	protected FileInfo fi = null;
	protected long skip;
	protected int currentIndex = -1;
	protected boolean showProgress;
	protected ImageReader reader;
	protected InputStream is;
	protected ColorModel cm;

	


	public static ProjectionSource [] getProjectionSources(){
		ProjectionSource [] sources = {new TiffProjectionSource(), new ZipProjectionSource(), new SEQProjectionSource(), new DicomProjectionSource(), new NRRDProjectionSource(), new DennerleinProjectionSource()};
		return sources;
	}

	public static ProjectionSource openProjectionStream(String filename) throws IOException {
		ProjectionSource [] sources = getProjectionSources();
		ProjectionSource source = null;
		for (int i = 0; i < sources.length; i++){
			try {
				sources[i].initStream(filename);
				source = sources[i];
				break;
			} catch (Exception e) {
				// May throw some exception, if it is the wrong file type.
				CONRAD.log(sources[i] +" failed because " +e.getLocalizedMessage());
				// e.printStackTrace();
			}
		}
		if (source == null) {
			throw new IOException("No matching file format found.");
		}
		return source;	
	}

	public synchronized void getNextProjection(IndividualImagePipelineFilteringTool tool){
		
		Grid2D grid = getNextProjection();
		if (grid != null){
			tool.setImageProcessor(grid);
			tool.setImageIndex(getCurrentProjectionNumber());
		} else {
			tool.setImageProcessor(null);
			tool.setImageIndex(-1);
		}
	}

	public void setShowProgress(boolean showProgress) {
		this.showProgress = showProgress;
	}

	public boolean isShowProgress() {
		return showProgress;
	}

	public int getCurrentProjectionNumber(){
		return currentIndex;
	}

	public Grid2D getNextProjection() {
		ImageProcessor ip = null;
		Object pixels = reader.readPixels(is, skip);
		if (pixels == null) {
			return null;
		}
		if (pixels instanceof byte[])
			ip = new ByteProcessor(fi.width, fi.height, (byte[]) pixels, cm);
		else if (pixels instanceof short[])
			ip = new ShortProcessor(fi.width, fi.height, (short[]) pixels, cm);
		else if (pixels instanceof int[])
			ip = new ColorProcessor(fi.width, fi.height, (int[]) pixels);
		else if (pixels instanceof float[]) {
			// no conversion needed
		} else
			throw new IllegalArgumentException("Unknown stack type");
			
		skip = fi.gapBetweenImages;
		currentIndex++;
		if (showProgress){
			IJ.showProgress((0.0 + currentIndex) / fi.nImages);
		}
		if(ip != null)
			pixels = ip.toFloat(0, null).getPixels();
		Grid2D grid = new Grid2D((float [])pixels, fi.width, fi.height);
		
		return grid;
	}


	protected void init() {
		cm = createColorModel(fi);
		//if (fi.nImages>1){
			initStack();
		//}
	}


	/** Returns an IndexColorModel for the image specified by this FileInfo. */
	protected ColorModel createColorModel(FileInfo fi) {
		if (fi.fileType==FileInfo.COLOR8 && fi.lutSize>0)
			return new IndexColorModel(8, fi.lutSize, fi.reds, fi.greens, fi.blues);
		else
			return LookUpTable.createGrayscaleColorModel(fi.whiteIsZero);
	}



	/** Opens a stack of images. */
	protected void initStack() {
		skip = fi.getOffset();
		try {
			reader = new ImageReader(fi);
			is = createInputStream(fi);
			if (is==null) return ;
		}
		catch (Exception e) {
			CONRAD.log("" + e);
			e.printStackTrace();
		}
		catch(OutOfMemoryError e) {
			e.printStackTrace();
			IJ.outOfMemory(fi.fileName);
		}
	}

	/** Returns an InputStream for the image described by this FileInfo. */
	protected InputStream createInputStream(FileInfo fi) throws IOException, MalformedURLException {
		InputStream is = null;
		boolean gzip = fi.fileName!=null && (fi.fileName.endsWith(".gz")||fi.fileName.endsWith(".GZ"));
		if (fi.inputStream!=null)
			is = fi.inputStream;
		else if (fi.url!=null && !fi.url.equals(""))
			is = new URL(fi.url+fi.fileName).openStream();
		else {
			if (fi.directory.length()>0 && !fi.directory.endsWith(Prefs.separator))
				fi.directory += Prefs.separator;
			File f = new File(fi.directory + fi.fileName);
			if (gzip) fi.compression = FileInfo.COMPRESSION_UNKNOWN;
			if (f==null || f.isDirectory() || !validateFileInfo(f, fi))
				is = null;
			else
				is = new FileInputStream(f);
		}
		if (is!=null) {
			if (fi.compression>=FileInfo.LZW)
				is = new RandomAccessStream(is);
			else if (gzip)
				is = new GZIPInputStream(is, 50000);
		}
		return is;
	}

	protected static boolean validateFileInfo(File f, FileInfo fi) {
		long offset = fi.getOffset();
		long length = 0;
		if (fi.width<=0 || fi.height<0) {
			error("Width or height <= 0.", fi, offset, length);
			return false;
		}
		if (offset>=0 && offset<1000L)
			return true;
		if (offset<0L) {
			error("Offset is negative.", fi, offset, length);
			return false;
		}
		if (fi.fileType==FileInfo.BITMAP || fi.compression!=FileInfo.COMPRESSION_NONE)
			return true;
		length = f.length();
		long size = fi.width*fi.height*fi.getBytesPerPixel();
		size = fi.nImages>1?size:size/4;
		if (fi.height==1) size = 0; // allows plugins to read info of unknown length at end of file
		if (offset+size>length) {
			error("Offset + image size > file length.", fi, offset, length);
			return false;
		}
		return true;
	}

	protected static void error(String msg, FileInfo fi, long offset, long length) {
		IJ.error("FileOpener", "FileInfo parameter error. \n"
				+msg + "\n \n"
				+"  Width: " + fi.width + "\n"
				+"  Height: " + fi.height + "\n"
				+"  Offset: " + offset + "\n"
				+"  Bytes/pixel: " + fi.getBytesPerPixel() + "\n"
				+(length>0?"  File length: " + length + "\n":"")
		);
	}



	/** Reads the pixel data from an image described by a FileInfo object. */
	protected Object readPixels(FileInfo fi) {
		Object pixels = null;
		try {
			InputStream is = createInputStream(fi);
			if (is==null)
				return null;
			ImageReader reader = new ImageReader(fi);
			pixels = reader.readPixels(is);
			is.close();
		}
		catch (Exception e) {
			if (!Macro.MACRO_CANCELED.equals(e.getMessage()))
				IJ.handleException(e);
		}
		return pixels;
	}

	public void close() throws IOException{
		is.close();
	}

	/**
	 * converts a byte sequence to integer
	 * @param values the byte array
	 * @param index the index in 32bit integer metric
	 * @return the integer
	 */
	public long convertToUnsignedInt(byte[] values, int index) {
		int offset = index * 4;  
		//System.out.println(values[offset +0] + " " + values[offset +1] + " " + values[offset +2] + " " +values[offset +3]);
		int val = values[offset+3];
		if (val < 0) val += 256;
		long value = val;
		value = value << 8;
		val = values[offset+2];
		if (val < 0) val += 256;
		value += val;
		value = value << 8;
		val = values[offset+1];
		if (val < 0) val += 256;
		value += val;
		value = value << 8;
		val = values[offset];
		if (val < 0) val += 256;
		value += val;
		return value;
	}
	
	/**
	 * converts a byte sequence to integer
	 * @param values the byte array
	 * @param index the index in 32bit integer metric
	 * @return the integer
	 */
	public int convertToUnsignedShort(byte[] values, int index) {
		int offset = index * 2;  
		//System.out.println(values[offset +0] + " " + values[offset +1] + " " + values[offset +2] + " " +values[offset +3]);
		int val = values[offset+1];
		if (val < 0) val += 256;
		int value = val;
		value = value << 8;
		val = values[offset];
		if (val < 0) val += 256;
		value += val;
		return value;
	}
	

}

/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/