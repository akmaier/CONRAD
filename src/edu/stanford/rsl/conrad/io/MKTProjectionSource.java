/*
 * Copyright (C) 2014 Andreas Maier, Christian Jaremenko
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.io;


import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.CONRAD;
import ij.IJ;
import ij.io.FileInfo;

/**
 * Class to stream Cellvizio's MTK Format.  
 * 
 * @author akmaier
 *
 */
public class MKTProjectionSource extends FileProjectionSource {

	protected boolean debug = true;
	private boolean directoryMode;
	private ArrayList<File> fileList;

	@Override
	public void initStream (String filename) throws IOException {
		File file = new File(filename);
		if (file.isDirectory()){
			// multi file format
			fileList = new ArrayList<File>();
			String [] files = file.list();
			Arrays.sort(files);
			for (String f: files){
				if (f.substring(f.length()-4).compareToIgnoreCase(".mkt")==0){
					fileList.add(new File(file.getAbsolutePath() + "/" +f));
				}
			}
			if (fileList.size() == 0) throw new IOException("Directory does not contain .viv files");
			fi = getHeaderInfo(fileList.get(0).getAbsolutePath());
			directoryMode = true;
		} else {
			// Code for single file
			fi = getHeaderInfo(filename);
		}
		init();
	}

	/**
	 * Reads the header information from the file into a fileinfo object
	 * @param filename the filename
	 * @return the FileInfo
	 * @throws IOException
	 */
	public FileInfo getHeaderInfo( String filename ) throws IOException {
		if (IJ.debugMode) CONRAD.log("Entering MKT_Reader.getHeaderInfo():");
		FileInfo fi = new FileInfo();
		File file = new File(filename);
		fi.fileName=file.getName();
		fi.directory = file.getParent() + "/";
		// NB Need RAF in order to ensure that we know file offset
		fi.fileType = FileInfo.GRAY16_SIGNED;  // just assume this for the mo    
		fi.fileFormat = FileInfo.RAW;
		fi.nImages = 1;
		// read header info
		FileInputStream fis = new FileInputStream(file);
		byte [] unknownHeader = new byte[10];
		byte [] size = new byte[4];
		fis.read(unknownHeader);
		fis.read(size);
		long sizeInByte = ByteBuffer.wrap(size).getInt();
		fis.close();
		fi.offset = 16;
		fi.gapBetweenImages=32;
		//System.out.println("Read " + fi.offset);
		if (fi.offset != 16){
			throw new IOException("Wrong Header Size; Not an MKT File.");
		}
		fi.width = 576;
		if((sizeInByte/(2*fi.width))%2!=0){
			fi.width=512;
			fi.height = (int) (sizeInByte/(2*fi.width));
		} else {
			fi.height = (int) (sizeInByte/(2*fi.width));
		}
		
		/* Here we had a look at the 16 byte header and this is what we found so far:
		 * Width	Height	Size	Size (Byte)	Found	Size B Hex
		 * 576		580		334080	668160		320A	A3200
		 * 576		578		332928	665856		290A	A2900
		 * 576		576		331776	663552		200A	A2000
		 * 512		512		262144	524288		0008	80000
		*/
		fi.nImages = 1204;

		if (true) {
			System.out.println("MTK Reading image with " + fi.nImages + " frames with " + fi.width + "x" + fi.height + " resolution");
		}
		fi.compression = FileInfo.COMPRESSION_NONE;
		
		fi.intelByteOrder = true;

		return (fi);
	}

	private Grid2D readSingleImage(){
		Grid2D revan = super.getNextProjection();
		if (revan != null) {
			try {
				Thread.sleep(CONRAD.INPUT_QUEUE_DELAY);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			Grid2D uncompressed = new Grid2D(revan.getWidth(), revan.getHeight());
			// uncompression
			int min = Integer.MAX_VALUE;
			int max = Integer.MIN_VALUE;
			for (int i = 0; i< revan.getWidth(); i++){
				for (int j = 0; j< revan.getHeight(); j++){
					int val = (int) (revan.getPixelValue(i, j));
					if (val > max) max = val;
					if (val < min) min = val;


					uncompressed.putPixelValue(i,j,val);
					
				}
			}
			//System.out.println("Range: " + min + " " + max);
			return uncompressed;
		} else {
			return null;
		}

	}

	@Override
	public Grid2D getNextProjection(){
		if (!directoryMode) {
			return readSingleImage();
		} else {
			try{
				fi = getHeaderInfo(fileList.get(currentIndex+1).getPath());
				initStack();
				return readSingleImage();
			} catch (IOException e) {
				e.printStackTrace();				
				return null;
			} catch (IndexOutOfBoundsException e){
				return null;
			}
		}
	}



	public   byte[] convertInt2Bytes(int value) {
		byte[] buf = new byte[4];
		buf[3] = (byte) ((value & 0xFF000000)>>24);
		buf[2] = (byte) ((value & 0x00FF0000)>>16);
		buf[1] = (byte) ((value & 0x0000FF00)>>8);
		buf[0] = (byte) (value & 0x000000FF);
		return buf;
	}

	public   byte[] convertInt2UnShort(int value) {
		byte[] buf = new byte[2];
		buf[1] = (byte) ((value & 0x0000FF00)>>8);
		buf[0] = (byte) (value & 0x000000FF);
		return buf;
	}

}
