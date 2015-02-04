// Nrrd_Writer
// -----------
// ImageJ plugin to save a file in Gordon Kindlmann's NRRD 
// or 'nearly raw raster data' format, a simple format which handles
// coordinate systems and data types in a very general way
// See http://teem.sourceforge.net/nrrd/
// and http://flybrain.stanford.edu/nrrd/

// (c) Gregory Jefferis 2007
// Department of Zoology, University of Cambridge
// jefferis@gmail.com
// All rights reserved
// Source code released under Lesser Gnu Public License v2

// v0.1 2007-04-02
// - First functional version can write single channel image (stack)
// to raw/gzip encoded monolithic nrrd file
// - Writes key spatial calibration information	including
//   spacings, centers, units, axis mins

// TODO
// - Support for multichannel images, time data
// - option to write a detached header instead of detached nrrd file

// NB this class can be used to create detached headers for other file types
// See 

import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.io.FileInfo;
import ij.io.ImageWriter;
import ij.io.SaveDialog;
import ij.measure.Calibration;
import ij.plugin.PlugIn;

import java.io.*;
import java.util.Date;
import java.util.zip.GZIPOutputStream;

                          
public class Nrrd_Writer implements PlugIn {

	private static final String plugInName = "Nrrd Writer";
	private static final String noImages = plugInName+"...\n"+ "No images are open.";
	private static final String supportedTypes =
		plugInName+"..." + "Supported types:\n\n" +
				"32-bit Grayscale float : FLOAT\n" +
				"(32-bit Grayscale integer) : LONG\n" +
				"16-bit Grayscale integer: INT\n" +
				"(16-bit Grayscale unsigned integer) : UINT\n"+
				"8-bit Grayscale : BYTE\n"+
				"8-bit Colour LUT (converted to greyscale): BYTE\n";
				
	public static final int NRRD_VERSION = 4;	
	private String imgTypeString=null;	
	String nrrdEncoding="raw";
	// See http://teem.sourceforge.net/nrrd/format.html#centers
	static final String defaultNrrdCentering="node";	
	
	String setNrrdEncoding(String enc) throws IOException {
		enc=enc.toLowerCase();
		if (enc.equals("raw")) nrrdEncoding="raw";
		else if (enc.equals("gz") || enc.equals("gzip")) nrrdEncoding="gzip";
		else if (enc.equals("bz2") || enc.equals("bzip2")) throw new IOException("bzip2 encoding not yet supported");
		else if (enc.equals("txt") || enc.equals("text") || enc.equals("ascii")) throw new IOException("text encoding not yet supported");
		else if (enc.equals("hex") ) throw new IOException("hex encoding not yet supported");
		else throw new IOException("Unknown encoding "+enc);
		return nrrdEncoding;
	}
						
	public void run(String arg) {
		ImagePlus imp = WindowManager.getCurrentImage();
		if (imp == null) {
			IJ.showMessage(noImages);
			return;
		}

		String name = arg;
		if (arg == null || arg.equals("")) {
			name = imp.getTitle();
		}
		
		SaveDialog sd = new SaveDialog(plugInName+"...", name, ".nrrd");
		String file = sd.getFileName();
		if (file == null) return;
		String directory = sd.getDirectory();
		save(imp, directory, file);
	}

	public void save(ImagePlus imp, String directory, String file) {
		if (imp == null) {
			IJ.showMessage(noImages);
			return;
		}
		FileInfo fi = imp.getFileInfo();

		// Make sure that we can save this kind of image
		if(imgTypeString==null) {
			imgTypeString=imgType(fi.fileType);
			if (imgTypeString.equals("unsupported")) {
				IJ.showMessage(supportedTypes);
				return;
			}
		}		
		// Set the fileName stored in the file info record to the
		// file name that was passed in or chosen in the dialog box
		fi.fileName=file;
		fi.directory=directory;
		
		// Actually write out the image
		try {
			writeImage(fi,imp.getCalibration()); 
		} catch (IOException e) {
			IJ.error("An error occured writing the file.\n \n" + e);
			IJ.showStatus("");
		}
	}
	void writeImage(FileInfo fi, Calibration cal) throws IOException {
		FileOutputStream out = new FileOutputStream(new File(fi.directory, fi.fileName));
		// First write out the full header
		Writer bw = new BufferedWriter(new OutputStreamWriter(out));
		// Blank line terminates header
		bw.write(makeHeader(fi,cal)+"\n");
		// Flush rather than close
		bw.flush();		

		// Then the image data
		ImageWriter writer = new ImageWriter(fi);
		if(nrrdEncoding.equals("gzip")) {
			GZIPOutputStream zStream = new GZIPOutputStream(new BufferedOutputStream( out ));
			writer.write(zStream);
			zStream.close();
		} else {
			writer.write(out);
			out.close();
		}
		IJ.showStatus("Saved "+ fi.fileName);
	}

	public static String makeDetachedHeader(FileInfo fi,Calibration cal, boolean withDataFile) {
		// this static method can also be used externally to generate 
		// a basic nrrd detached header
		// Right now it will only work for single channel images
		// NB You can add further fields to this basic header but 
		// You MUST add your own blank line at the end
		StringWriter out=new StringWriter();
		out.write(makeHeader(fi,cal));
		out.write("byte skip: "+(fi.longOffset>0?fi.longOffset:fi.offset)+"\n");
		if(withDataFile) out.write("data file: "+fi.fileName+"\n");
		return out.toString();
	}
		
	public static String makeHeader(FileInfo fi,Calibration cal) {
		// NB You can add further fields to this basic header but 
		// You MUST add your own blank line at the end

		// See http://teem.sourceforge.net/nrrd/format.html
		/* 
		 * type: uchar
		 * dimension: 3
		 * sizes: 3 640 480
		 * encoding: raw
		 */
		StringWriter out=new StringWriter();
		
		out.write("NRRD000"+NRRD_VERSION+"\n");
		out.write("# Created by Nrrd_Writer at "+(new Date())+"\n");

		// Fetch and write the data type
		out.write("type: "+imgType(fi.fileType)+"\n");
		// Fetch and write the encoding
		out.write("encoding: "+getEncoding(fi)+"\n");
		
		if(fi.intelByteOrder) out.write("endian: little\n");
		else out.write("endian: big\n");
		
		int dimension=(fi.nImages==1)?2:3;		
		
		out.write("dimension: "+dimension+"\n");
		out.write(dimmedLine("sizes",dimension,fi.width+"",fi.height+"",fi.nImages+""));
		if(cal!=null)
			out.write(dimmedLine("spacings",dimension,cal.pixelWidth+"",cal.pixelHeight+"",cal.pixelDepth+""));		
		// GJ: It's my understanding that ImageJ operates on a 'node' basis
		// See http://teem.sourceforge.net/nrrd/format.html#centers
		out.write(dimmedLine("centers",dimension,defaultNrrdCentering,defaultNrrdCentering,"node"));		
		String units;
		if(cal!=null) units=cal.getUnit();
		else units=fi.unit;
		if(units.equals("µm")) units="microns";
		if(!units.equals("")) out.write(dimmedQuotedLine("units",dimension,units,units,units));		

		// Only write axis mins if origin info has at least one non-zero
		// element
		if(cal!=null && (cal.xOrigin!=0 || cal.yOrigin!=0 || cal.zOrigin!=0) ) {
			out.write(dimmedLine("axis mins",dimension,(-cal.xOrigin*cal.pixelWidth)+"",
								 (-cal.yOrigin*cal.pixelHeight)+"",(-cal.zOrigin*cal.pixelDepth)+""));
			System.out.println(" ");
		}
		return out.toString();
	}
		
	public static String imgType(int fiType) {
		switch (fiType) {
			case FileInfo.GRAY32_FLOAT:
				return "float";
			case FileInfo.GRAY32_INT:
				return "int32";
			case FileInfo.GRAY32_UNSIGNED:
				return "uint32";
			case FileInfo.GRAY16_SIGNED:
				return "int16";	
			case FileInfo.GRAY16_UNSIGNED:
				return "uint16";
		
			case FileInfo.COLOR8:
			case FileInfo.GRAY8:
				return "uint8";
			default:
				return "unsupported";
		}
	}
	
	public static String getEncoding(FileInfo fi) {
		NrrdFileInfo nfi;
		
		if (IJ.debugMode) IJ.log("fi :"+fi);
		
		try {
			nfi=(NrrdFileInfo) fi;
			if (IJ.debugMode) IJ.log("nfi :"+nfi);
			if(nfi.encoding!=null && !nfi.encoding.equals("")) return (nfi.encoding);
		} catch (Exception e) { }
		
		switch(fi.compression) {
			case NrrdFileInfo.GZIP: return("gzip");
			case NrrdFileInfo.BZIP2: return null;
			default:
			break;
		}
		// These aren't yet supported
		switch(fi.fileFormat) {
			case NrrdFileInfo.NRRD_TEXT:
			case NrrdFileInfo.NRRD_HEX:
			return(null);
			default:
			break;
		}
		// The default!
		return "raw";
	}
			
	private static String dimmedQuotedLine(String tag,int dimension,String x1,String x2,String x3) {
		x1="\""+x1+"\"";
		x2="\""+x2+"\"";
		x3="\""+x3+"\"";
		return dimmedLine(tag, dimension,x1, x2, x3);
	}
	
	private static String dimmedLine(String tag,int dimension,String x1,String x2,String x3) {
		String rval=null;
		if(dimension==2) rval=tag+": "+x1+" "+x2+"\n";
		else if(dimension==3) rval=tag+": "+x1+" "+x2+" "+x3+"\n";
		return rval;
	}	
}

class NrrdFileInfo extends FileInfo {
	public int dimension=0;
	public int[] sizes;
	public String encoding="";
	public String[] centers=null;
	
	// Additional compression modes for fi.compression
	public static final int GZIP = 1001;
	public static final int ZLIB = 1002;
	public static final int BZIP2 = 1003;
	
	// Additional file formats for fi.fileFormat
	public static final int NRRD = 1001;
	public static final int NRRD_TEXT = 1002;
	public static final int NRRD_HEX = 1003;

}
