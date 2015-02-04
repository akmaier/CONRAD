/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.io;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;

import edu.stanford.rsl.conrad.utils.CONRAD;


import ij.IJ;
import ij.io.FileInfo;

public class NRRDProjectionSource extends FileProjectionSource {

	public final String uint8Types="uchar, unsigned char, uint8, uint8_t";
	public final String int16Types="short, short int, signed short, signed short int, int16, int16_t";
	public final String uint16Types="ushort, unsigned short, unsigned short int, uint16, uint16_t";
	public final String int32Types="int, signed int, int32, int32_t";
	public final String uint32Types="uint, unsigned int, uint32, uint32_t";
	private String notes = "";
	
	private boolean detachedHeader=false;
	protected String headerPath=null;
	protected String imagePath=null;
	protected String imageName=null;	
	
	public void initStream (String filename) throws IOException {
		fi = getHeaderInfo(filename); 
		if (fi!=null && fi.width>0 && fi.height>0) {
			init();
		} else {
			throw new IOException("Format does not match: width = " + fi.width + " height = " + fi.height + " offset = " + fi.offset);
		}
	}
	
	public FileInfo getHeaderInfo( String filename ) throws IOException {

		if (IJ.debugMode) CONRAD.log("Entering Nrrd_Reader.readHeader():");
		FileInfo fi = new FileInfo();
		File file = new File(filename);
		fi.fileName=file.getName();
		fi.directory = file.getParent() + "/";
		// NB Need RAF in order to ensure that we know file offset
		RandomAccessFile input = new RandomAccessFile(fi.directory+fi.fileName,"r");

		String thisLine,noteType,noteValue, noteValuelc;

		fi.fileType = FileInfo.GRAY8;  // just assume this for the mo    
		fi.fileFormat = FileInfo.RAW;
		fi.nImages = 1;
		int dimension = 0;

		// parse the header file, until reach an empty line//	boolean keepReading=true;
		while(true) {
			thisLine=input.readLine();
			if(thisLine==null || thisLine.equals("")) {
				if(!detachedHeader) fi.longOffset = input.getFilePointer();
				break;
			}		
			notes+=thisLine+"\n";
			if(thisLine.indexOf("#")==0) continue; // ignore comments

			noteType=getFieldPart(thisLine,0).toLowerCase(); // case irrelevant
			noteValue=getFieldPart(thisLine,1);
			noteValuelc=noteValue.toLowerCase();
			String firstNoteValue=getSubField(thisLine,0);

			if (IJ.debugMode) CONRAD.log("NoteType:"+noteType+", noteValue:"+noteValue);

			if (noteType.equals("data file")||noteType.equals("datafile")) {
				// This is a detached header file
				// There are 3 kinds of specification for the data files
				// 	1.	data file: <filename>
				//	2.	data file: <format> <min> <max> <step> [<subdim>]
				//	3.	data file: LIST [<subdim>]
				if(firstNoteValue.equals("LIST")) {
					// TOFIX - type 3
					throw new IOException("Nrrd_Reader: not yet able to handle datafile: LIST specifications");
				} else if(!getSubField(thisLine,1).equals("")) {
					// TOFIX - type 2
					throw new IOException("Nrrd_Reader: not yet able to handle datafile: sprintf file specifications");
				} else {
					// Type 1 specification
					File imageFile;
					// Relative or absolute
					if(noteValue.indexOf("/")==0) {
						// absolute
						imageFile=new File(noteValue);
						// TOFIX could also check local directory if absolute path given
						// but dir does not exist
					} else {
						//CONRAD.log("fi.directory = "+fi.directory);					
						imageFile=new File(fi.directory,noteValue);
					}
					//CONRAD.log("image file ="+imageFile);

					if(imageFile.exists()) {
						fi.directory=imageFile.getParent();
						fi.fileName=imageFile.getName();
						imagePath=imageFile.getPath();
						detachedHeader=true;
					} else {
						throw new IOException("Unable to find image file ="+imageFile.getPath());
					}
				}										
			}

			if (noteType.equals("dimension")) {
				dimension=Integer.valueOf(noteValue).intValue();
				if(dimension>3) throw new IOException("Nrrd_Reader: Dimension>3 not yet implemented!");
			}
			if (noteType.equals("sizes")) {
				for(int i=0;i<dimension;i++) {
					int integer =Integer.valueOf(getSubField(thisLine,i)).intValue();
					if(i==0) fi.width=integer;
					if(i==1) fi.height=integer;
					if(i==2) fi.nImages=integer;
				}
			}
			if (noteType.equals("type")) {
				if (uint8Types.indexOf(noteValuelc)>=0) {
					fi.fileType=FileInfo.GRAY8;
				} else if(uint16Types.indexOf(noteValuelc)>=0) {
					fi.fileType=FileInfo.GRAY16_SIGNED;
				} else if(int16Types.indexOf(noteValuelc)>=0) {
					fi.fileType=FileInfo.GRAY16_UNSIGNED;
				} else if(uint32Types.indexOf(noteValuelc)>=0) {
					fi.fileType=FileInfo.GRAY32_UNSIGNED;
				} else if(int32Types.indexOf(noteValuelc)>=0) {
					fi.fileType=FileInfo.GRAY32_INT;
				} else if(noteValuelc.equals("float")) {
					fi.fileType=FileInfo.GRAY32_FLOAT;
				} else if(noteValuelc.equals("double")) {
					fi.fileType=FileInfo.GRAY64_FLOAT;
				} else {
					throw new IOException("Unimplemented data type ="+noteValue);
				}
			}
			if (noteType.equals("byte skip")||noteType.equals("byteskip")) fi.longOffset=Long.valueOf(noteValue).longValue();
			if (noteType.equals("endian")) {
				if(noteValuelc.equals("little")) {
					fi.intelByteOrder = true;
				} else {
					fi.intelByteOrder = false;
				}
			}

			if (noteType.equals("encoding")) {
				if(noteValuelc.equals("gz")) noteValuelc="gzip";
				//fi.encoding=noteValuelc;
			}	
		}

		if(!detachedHeader) fi.longOffset = input.getFilePointer();
		input.close();
		return (fi);
	}
	
	String getFieldPart(String str, int fieldIndex) {
		str=str.trim(); // trim the string
		String[] fieldParts=str.split(":\\s+");
		if(fieldParts.length<2) return(fieldParts[0]);
		//CONRAD.log("field = "+fieldParts[0]+"; value = "+fieldParts[1]+"; fieldIndex = "+fieldIndex);

		if(fieldIndex==0) return fieldParts[0];
		else return fieldParts[1];
	}
	
	String getSubField(String str, int fieldIndex) {
		String fieldDescriptor=getFieldPart(str,1);
		fieldDescriptor=fieldDescriptor.trim(); // trim the string

		if (IJ.debugMode) CONRAD.log("fieldDescriptor = "+fieldDescriptor+"; fieldIndex = "+fieldIndex);

		String[] fields_values=fieldDescriptor.split("\\s+");

		if (fieldIndex>=fields_values.length) {
			return "";
		} else {
			String rval=fields_values[fieldIndex];
			if(rval.startsWith("\"")) rval=rval.substring(1);
			if(rval.endsWith("\"")) rval=rval.substring(0, rval.length()-1);
			return rval;
		}
	}

}
