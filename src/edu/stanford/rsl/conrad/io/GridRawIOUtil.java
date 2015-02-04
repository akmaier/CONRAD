/*
 * Copyright (C) 2010-2014  Andreas Maier, Kerstin Müller
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.io;

import ij.ImagePlus;
import ij.io.FileInfo;
import ij.io.FileOpener;
import ij.io.ImageWriter;
import ij.process.FloatProcessor;
import ij.process.ImageConverter;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.utils.ImageUtil;

/**
 * Class to write and read raw data from files using ImageJ API.
 * The result is read to Grid structures and returned.
 * 
 * @author Kerstin Müller
 *
 */
public abstract class GridRawIOUtil {

	/**
	 * The default float 32, little endian format for CONRAD.
	 * @return the FileInfo object
	 */
	public static FileInfo getDefaultFloat32LittleEndianFileInfo(){
		FileInfo fI = new FileInfo();
		fI.fileFormat = FileInfo.RAW;
		fI.fileType = FileInfo.GRAY32_FLOAT;
		fI.height = 256;
		fI.width = 256;
		fI.nImages = 1;
		fI.intelByteOrder = true;
		fI.directory = "";
		fI.fileName = "";
		return fI;
	}

	/**
	 * The default float 32, big endian format for CONRAD.
	 * @return the FileInfo object
	 */
	public static FileInfo getDefaultFloat32BigEndianFileInfo(){
		FileInfo fI = new FileInfo();
		fI.fileFormat = FileInfo.RAW;
		fI.fileType = FileInfo.GRAY32_FLOAT;
		fI.height = 256;
		fI.width = 256;
		fI.nImages = 1;
		fI.intelByteOrder = false;
		fI.directory = "";
		fI.fileName = "";
		return fI;
	}

	/**
	 * The default float 32, little endian format for CONRAD.
	 * @param grid the 3D grid to process
	 * @return the FileInfo object
	 */
	public static FileInfo getDefaultFloat32LittleEndianFileInfo(Grid3D grid){
		FileInfo fI = new FileInfo();
		fI.fileFormat = FileInfo.RAW;
		fI.fileType = FileInfo.GRAY32_FLOAT;
		fI.height = grid.getSize()[1];
		fI.width = grid.getSize()[0];
		fI.nImages = grid.getSize()[2];
		fI.intelByteOrder = true;
		fI.directory = "";
		fI.fileName = "";
		return fI;
	}

	/**
	 * The default float 32, big endian format for CONRAD.
	 * @param grid the 3D grid to process
	 * @return the FileInfo object
	 */
	public static FileInfo getDefaultFloat32BigEndianFileInfo(Grid3D grid){
		FileInfo fI = new FileInfo();
		fI.fileFormat = FileInfo.RAW;
		fI.fileType = FileInfo.GRAY32_FLOAT;
		fI.height = grid.getSize()[1];
		fI.width = grid.getSize()[0];
		fI.nImages = grid.getSize()[2];
		fI.intelByteOrder = false;
		fI.directory = "";
		fI.fileName = "";
		return fI;
	}

	/**
	 * The default float 32, little endian format for CONRAD.
	 * @param grid the 2D grid to process
	 * @return the FileInfo object
	 */
	public static FileInfo getDefaultFloat32LittleEndianFileInfo(Grid2D grid){
		FileInfo fI = new FileInfo();
		fI.fileFormat = FileInfo.RAW;
		fI.fileType = FileInfo.GRAY32_FLOAT;
		fI.height = grid.getSize()[1];
		fI.width = grid.getSize()[0];
		fI.nImages = 1;
		fI.intelByteOrder = true;
		fI.directory = "";
		fI.fileName = "";
		return fI;
	}

	/**
	 * The default float 32, big endian format for CONRAD.
	 * @param grid the 2D grid to process
	 * @return the FileInfo object
	 */
	public static FileInfo getDefaultFloat32BigEndianFileInfo(Grid2D grid){
		FileInfo fI = new FileInfo();
		fI.fileFormat = FileInfo.RAW;
		fI.fileType = FileInfo.GRAY32_FLOAT;
		fI.height = grid.getSize()[1];
		fI.width = grid.getSize()[0];
		fI.nImages = 1;
		fI.intelByteOrder = false;
		fI.directory = "";
		fI.fileName = "";
		return fI;
	}
	
	/**
	 * The default float 32, little endian format for CONRAD.
	 * @param grid the 1D grid to process
	 * @return the FileInfo object
	 */
	public static FileInfo getDefaultFloat32LittleEndianFileInfo(Grid1D grid){
		FileInfo fI = new FileInfo();
		fI.fileFormat = FileInfo.RAW;
		fI.fileType = FileInfo.GRAY32_FLOAT;
		fI.height = 1;
		fI.width = grid.getSize()[0];
		fI.nImages = 1;
		fI.intelByteOrder = true;
		fI.directory = "";
		fI.fileName = "";
		return fI;
	}

	/**
	 * The default float 32, big endian format for CONRAD.
	 * @param grid the 1D grid to process
	 * @return the FileInfo object
	 */
	public static FileInfo getDefaultFloat32BigEndianFileInfo(Grid1D grid){
		FileInfo fI = new FileInfo();
		fI.fileFormat = FileInfo.RAW;
		fI.fileType = FileInfo.GRAY32_FLOAT;
		fI.height = 1;
		fI.width = grid.getSize()[0];
		fI.nImages = 1;
		fI.intelByteOrder = false;
		fI.directory = "";
		fI.fileName = "";
		return fI;
	}

	/**
	 * Method to write a Grid to raw disk space.
	 * @param grid the grid to write
	 * @param fileInfo the file into that describes the raw data file format
	 * @param filename the filename to save to 
	 * @throws IOException may occur, if we run out of diskspace or have an invalid filename
	 */
	public static void saveRawDataGrid(NumericGrid grid, FileInfo fileInfo, String filename) throws IOException
	{

		FileOutputStream fos = new FileOutputStream(filename);
		saveRawDataGrid(grid, fileInfo, fos);
		fos.close();
	}

	/**
	 * Method to write an Array of Grids to raw disk space.
	 * @param grid the grids to write
	 * @param fileInfo the file into that describes the raw data file format
	 * @param filename the filename to save to 
	 * @throws IOException may occur, if we run out of diskspace or have an invalid filename
	 */
	public static void saveRawDataGrid(NumericGrid[] grid, FileInfo fileInfo, String filename) throws IOException
	{

		FileOutputStream fos = new FileOutputStream(filename);
		for (int i=0; i<grid.length; i++){
			saveRawDataGrid(grid[i], fileInfo, fos);
		}
		fos.close();
	}

	/**
	 * Method to write a Grid to raw disk space.
	 * @param grid the grid to write
	 * @param fileInfo the file into that describes the raw data file format
	 * @param os the stream to write to 
	 * @throws IOException may occur, if we run out of diskspace or have an invalid filename
	 */
	public static void saveRawDataGrid(NumericGrid grid, FileInfo fileInfo, OutputStream os) throws IOException{
		if (grid instanceof Grid1D){
			saveRawDataGrid((Grid1D)grid, fileInfo, os);
			return;
		}
		if (grid instanceof Grid2D){
			saveRawDataGrid((Grid2D)grid, fileInfo, os);
			return;
		}
		if (grid instanceof Grid3D){
			saveRawDataGrid((Grid3D)grid, fileInfo, os);
			return;
		}
		throw new RuntimeException("This subtype of grid is not implemented.");
	}

	/**
	 * Method to write a Grid1D to raw disk space.
	 * @param grid the grid to write
	 * @param fileInfo the file into that describes the raw data file format
	 * @param os the stream to write to 
	 * @throws IOException may occur, if we run out of diskspace or have an invalid filename
	 */
	public static void saveRawDataGrid(Grid1D grid, FileInfo fileInfo, OutputStream os) throws IOException{
		ImageWriter writer = new ImageWriter(fileInfo);
		fileInfo.pixels = grid.getBuffer();
		writer.write(os);
	}

	/**
	 * Method to write a Grid2D to raw disk space.
	 * @param grid the grid to write
	 * @param fileInfo the file into that describes the raw data file format
	 * @param os the stream to write to 
	 * @throws IOException may occur, if we run out of diskspace or have an invalid filename
	 */
	public static void saveRawDataGrid(Grid2D grid, FileInfo fileInfo, OutputStream os) throws IOException{
		ImageWriter writer = new ImageWriter(fileInfo);
		fileInfo.pixels = grid.getBuffer();
		writer.write(os);
	}

	/**
	 * Method to write a Grid3D to raw disk space.
	 * @param grid the grid to write
	 * @param fileInfo the file into that describes the raw data file format
	 * @param os the stream to write to 
	 * @throws IOException may occur, if we run out of diskspace or have an invalid filename
	 */
	public static void saveRawDataGrid(Grid3D grid, FileInfo fileInfo, OutputStream os) throws IOException{
		fileInfo.nImages = 1;
		for (int i=0; i < grid.getBuffer().size(); i++){
			saveRawDataGrid(grid.getBuffer().get(i), fileInfo, os);
		}
		fileInfo.nImages = grid.getSize()[2];
	}

	/**
	 * Method to load a Grid from raw data.
	 * @param fileInfo describes the file format
	 * @param filename the file location
	 * @return the grid
	 */
	public static NumericGrid loadFromRawData(FileInfo fileInfo, String filename){
		String sep = System.getProperty("file.separator");
		String [] fileSplit = filename.split("\\"+ sep);
		String file = fileSplit[fileSplit.length-1];
		String path = fileSplit[0];
		for (int i = 1; i < fileSplit.length-1;i++){
			path += sep + fileSplit[i];
		}
		fileInfo.fileName = file;
		fileInfo.directory = path;
		FileOpener fO = new FileOpener(fileInfo);
		ImagePlus image = fO.open(false);
		
		if (!(image.getProcessor() instanceof FloatProcessor)) {
			ImageConverter converter = new ImageConverter(image);
			converter.convertToGray32();
		}
		NumericGrid grid = null;
		if (fileInfo.nImages>1){
			 grid = ImageUtil.wrapImagePlus(image);
		} else {
			grid = ImageUtil.wrapFloatProcessor((FloatProcessor) image.getProcessor());
		}
		return grid;
	}

}
