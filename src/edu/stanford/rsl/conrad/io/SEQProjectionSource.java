/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.io;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.CONRAD;


import ij.IJ;
import ij.ImagePlus;
import ij.io.FileInfo;
import ij.process.ImageProcessor;

/**
 * Class to stream SEQ Format. Created with a lot of help from Jared Starman.
 * 
 * @author akmaier
 *
 */
public class SEQProjectionSource extends FileProjectionSource {

	protected boolean debug = true;
	protected float [] lut;
	protected int [] Inlut; 
	private boolean directoryMode;
	private ArrayList<File> fileList;

	private void initLUT() {
		lut = new float[65537];
		for (int i=0;i<= 16383;i++){
			lut[i+1] = ((float)i);
		}
		for (int i=0;i<= 16383;i++){
			lut[i+1+16384] = ((float)i)*2;
		}
		for (int i=0;i<= 16383;i++){
			lut[i+1+32768] = ((float)i)*4;
		}
		for (int i=0;i<= 16383;i++){
			lut[i+1+49152] = ((float)i)*8;
		}
		if (fi!=null && fi.width>0 && fi.height>0) {
			init();
		} else {
			throw new RuntimeException("Format does not match: width = " + fi.width + " height = " + fi.height + " offset = " + fi.offset);
		}

	}

	@Override
	public void initStream (String filename) throws IOException {
		File file = new File(filename);
		if (file.isDirectory()){
			// multi file format
			fileList = new ArrayList<File>();
			String [] files = file.list();
			Arrays.sort(files);
			for (String f: files){
				if (f.substring(f.length()-4).compareToIgnoreCase(".viv")==0){
					fileList.add(new File(file.getAbsolutePath() + "/" +f));
				}
			}
			if (fileList.size() == 0) throw new IOException("Directory does not contain .viv files");
			fi = getHeaderInfo(fileList.get(0).getAbsolutePath());
			initLUT();
			directoryMode = true;
		} else {
			// Code for single file
			fi = getHeaderInfo(filename);
			initLUT();
		}
	}

	/**
	 * Reads the header information from the file into a fileinfo object
	 * @param filename the filename
	 * @return the FileInfo
	 * @throws IOException
	 */
	public FileInfo getHeaderInfo( String filename ) throws IOException {
		if (IJ.debugMode) CONRAD.log("Entering Nrrd_Reader.readHeader():");
		FileInfo fi = new FileInfo();
		File file = new File(filename);
		fi.fileName=file.getName();
		fi.directory = file.getParent() + "/";
		// NB Need RAF in order to ensure that we know file offset
		RandomAccessFile input = new RandomAccessFile(fi.directory+fi.fileName,"r");
		fi.fileType = FileInfo.GRAY8;  // just assume this for the mo    
		fi.fileFormat = FileInfo.RAW;
		fi.nImages = 1;

		// parse the header file, until reach an empty line//	boolean keepReading=true;
		byte [] values = new byte[4];
		input.read(values);
		fi.offset = (int) convertToUnsignedInt(values,0);
		//System.out.println("Read " + fi.offset);
		if (fi.offset != 2048){
			throw new IOException("Wrong Header Size; Not an SEQ File.");
		}
		byte [] header = new byte [(fi.offset) -4];
		input.read(header);
		fi.width = (int) convertToUnsignedInt(header, 3);
		fi.height = (int) convertToUnsignedInt(header, 4);
		fi.nImages = (int) convertToUnsignedInt(header, 6);

		if (true) {
			System.out.println("SEQ Reading image with " + fi.nImages + " frames with " + fi.width + "x" + fi.height + " resolution");
		}
		int dataType = (int) convertToUnsignedInt(header, 458);
		if((dataType & 512)>0){
			fi.compression = FileInfo.COMPRESSION_UNKNOWN;			
		}
		else {
			fi.compression = FileInfo.COMPRESSION_NONE;

		}

		fi.fileType=FileInfo.GRAY16_UNSIGNED;
		fi.intelByteOrder = true;


		input.close();
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

					if (fi.compression == FileInfo.COMPRESSION_UNKNOWN){
						uncompressed.putPixelValue(i,j,lut[val+1]);
					}
					else {
						uncompressed.putPixelValue(i,j,val);
					}
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

	public void saveViva(ImagePlus imp, String path) {   
		if (imp.getNSlices()<1) return;

		ImageProcessor improc = imp.getProcessor();

		File file = new File(path);   
		DataOutputStream dos = null;  
		initInLUT();
		try {   
			dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)));    
			FileInfo fi = imp.getFileInfo();   
			dos.write(convertInt2Bytes(2048));  
			dos.writeChars("RevK04");
			dos.write(convertInt2Bytes(fi.width));
			dos.write(convertInt2Bytes(fi.height));
			dos.write(convertInt2Bytes(2));
			dos.write(convertInt2Bytes(fi.nImages));
			byte [] a = {-93,46,90,-36,-50,-73,17,-45,-94,-24,1,16,75,-58,-41,-51,0,0,0,0};
			dos.write(a);
			dos.write(ByteBuffer.allocate(1784).array());
			int dataType = 0;
			if (improc.getMax()>65535) dataType = 514;
			dos.write(convertInt2Bytes(dataType));
			dos.write(ByteBuffer.allocate(208).array());       	

			byte[] bBuff = new byte [imp.getWidth()*imp.getHeight()*2];         
			for (int j=0;j<imp.getHeight();j++){
				for (int k=0;k<imp.getWidth();k++){
					float val = (float) (improc.getPixelValue(k, j)); 	
					if (dataType == 514){  
						bBuff[j*imp.getWidth()*2+k*2] = convertInt2UnShort(Inlut[(int)val+1])[0];
						bBuff[j*imp.getWidth()*2+k*2+1] = convertInt2UnShort(Inlut[(int)val+1])[1];
					}
					else {
						bBuff[j*imp.getWidth()*2+k*2] = convertInt2UnShort((int)val)[0];
						bBuff[j*imp.getWidth()*2+k*2+1] = convertInt2UnShort((int)val)[1];
					}
				}
			}
			dos.write(bBuff);  
			dos.flush();  

			dos.close();   

		} catch (Exception e) {   
			e.printStackTrace();   
		}  
	}   


	private  void initInLUT() {
		Inlut = new int [131073];
		for (int i=0;i<= 16383;i++){
			Inlut[i+1] = (int)i;
		}
		for (int i=16384;i<= 32767;i++){
			Inlut[i+1] = (int)(i/2 + 16384);
		}
		for (int i=32768;i<= 49151;i++){
			Inlut[i+1] = (int)(i/4 + 32768);
		}
		for (int i=49152;i<= 131071;i++){
			Inlut[i+1] = (int)(i/8 + 49152);
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
