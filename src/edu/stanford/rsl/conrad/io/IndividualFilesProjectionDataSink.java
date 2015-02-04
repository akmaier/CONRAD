/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.io;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.text.NumberFormat;
import java.util.Locale;

import javax.swing.JOptionPane;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.pipeline.BufferedProjectionSink;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;

/**
 * Class to model a directory as projection data sink. The sink creates a new file for each projection in the configured raw format.
 * 
 * @author akmaier
 *
 */
public class IndividualFilesProjectionDataSink extends
BufferedProjectionSink {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2991638524617273945L;
	private String directory = null;
	private String prefix = null;
	private String format = null;
	private int width = 0;
	private int height = 0;
	private boolean littleEndian = true;
	private boolean closed = false;
	public final static String UnsignedShort = "Unsigned Short";
	public final static String SignedShort = "Signed Short";
	public final static String Float32Bit = "Float";

	@Override
	public String getName() {
		if (configured) {
			return "Write files in "+  format + " to " + directory;
		} else {
			return "Write files to directory";
		}
	}

	@Override
	public Grid3D getResult() {
		while (!closed) {
			try {
				Thread.sleep(CONRAD.INVERSE_SPEEDUP);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		width = 0;
		height = 0;
		return null;
	}

	private synchronized void setDimensions(Grid2D projection){
		width = projection.getWidth();
		height = projection.getHeight();	
		System.out.println("Setting new dimensions: " + width + " " + height);
	}

	@Override
	public void process(Grid2D projection, int projectionNumber)
	throws Exception {
		if (width == 0) setDimensions(projection);
		NumberFormat nf = NumberFormat.getNumberInstance(Locale.US);
		//NumberFormat nf = new NumberFormat("0000");
		nf.setMinimumFractionDigits(0);
		nf.setMaximumFractionDigits(0);
		nf.setMaximumIntegerDigits(4);
		nf.setMinimumIntegerDigits(4);
		nf.setGroupingUsed(false);
		String filename = directory + "/" + prefix + nf.format(projectionNumber);
		FileOutputStream writer = new FileOutputStream(filename);
		short [] shortPixels = null;
		if (format.equals(UnsignedShort) || format.equals(SignedShort) ) {
			shortPixels = new short [width * height];
		}
		float [] floatPixels = null;
		if (format.equals(Float32Bit)){
			floatPixels = new float [width * height];
		}
		for (int j = 0; j <  projection.getHeight(); j++){
			for (int i = 0; i < projection.getWidth(); i ++){
				float value = projection.getPixelValue(i, j);
				if (format.equals(UnsignedShort)) {
					shortPixels[i + (j*width)] = (short)((int) value);
				}
				if (format.equals(SignedShort)) {
					shortPixels[i + (j*width)] = (short)((int) value - 32768);
				}
				if (format.equals(Float32Bit)){
					floatPixels[i + (j*width)] = (float) value;
				}
			}
		}
		if (format.equals(UnsignedShort) || format.equals(SignedShort) ) {
			write16BitImage(writer, shortPixels);
		}
		if (format.equals(Float32Bit)){
			writeFloatImage(writer, floatPixels);
		}
		writer.flush();
		writer.close();
	}

	@Override
	public void setConfiguration(Configuration config) {

	}

	@Override
	public String toString() {
		return getName();
	}

	@Override
	public void configure() throws Exception {
		boolean success = true;
		width = 0;
		height = 0;
		directory = JOptionPane.showInputDialog("Enter directory:", directory);
		if (directory == null) success = false;
		prefix = JOptionPane.showInputDialog("Enter file prefix:", prefix);
		if (prefix == null) success = false;
		String [] formats = {Float32Bit, SignedShort, UnsignedShort};
		format = (String) JOptionPane.showInputDialog(null, "Select format:", "Format Selection", JOptionPane.INFORMATION_MESSAGE, null, formats, format);
		if (format == null) success = false;
		String [] endianess = {"Litte Endian", "Big Endian"};
		String endian = "Big Endian";
		if (littleEndian) endian = "Litte Endian";
		endian = (String) JOptionPane.showInputDialog(null, "Select endianess:", "Format Selection", JOptionPane.INFORMATION_MESSAGE, null, endianess, endian);
		if (endian == null) {
			success = false;
		} else {
			if (endian.equals("Litte Endian")){
				littleEndian = true;
			} else {
				littleEndian = false;
			}
		}
		configured = success;
	}
	
	public void configured(){
		configured = true;
	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@BOOK{Harlold06-JIO,\n" +
		"  author = {{Harold}, E. R.},\n" +
		"  title = {{Java I/O}},\n" +
		"  publisher = {O'Reilly Media, Inc.},\n" +
		"  address = {Sebastopol, CA, United States},\n" +
		"  year = {2006}\n" +
		"}";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		return "Harold ER, Java I/O, O'Reilly Media, Inc., Sebastopol, CA, United States, 2006.";
	}

	/**
	 * Same method as in ImageWriter
	 * @see ij.io.ImageWriter
	 * 
	 * @param out the Stream to write the ImageProcessor
	 * @param pixels the pixel data
	 * @throws IOException may occur
	 */
	void write16BitImage(OutputStream out, short[] pixels)  throws IOException {
		int bytesWritten = 0;
		int size = width*height*2;
		int count = 8192;
		byte[] buffer = new byte[count];

		while (bytesWritten<size) {
			if ((bytesWritten + count)>size)
				count = size - bytesWritten;
			int j = bytesWritten/2;
			int value;
			if (littleEndian)
				for (int i=0; i < count; i+=2) {
					value = pixels[j];
					buffer[i] = (byte)value;
					buffer[i+1] = (byte)(value>>>8);
					j++;
				}
			else
				for (int i=0; i < count; i+=2) {
					value = pixels[j];
					buffer[i] = (byte)(value>>>8);
					buffer[i+1] = (byte)value;
					j++;
				}
			out.write(buffer, 0, count);
			bytesWritten += count;
		}
	}

	/**
	 * Same method as in ImageWriter
	 * @see ij.io.ImageWriter
	 * 
	 * @param out the Stream to write the ImageProcessor
	 * @param pixels the pixel data
	 * @throws IOException may occur
	 */
	void writeFloatImage(OutputStream out, float[] pixels)  throws IOException {
		int bytesWritten = 0;
		int size = width*height*4;
		int count = 8192;
		byte[] buffer = new byte[count];
		int tmp;

		while (bytesWritten<size) {

			if ((bytesWritten + count)>size)
				count = size - bytesWritten;
			int j = bytesWritten/4;
			if (littleEndian)
				for (int i=0; i < count; i+=4) {
					tmp = Float.floatToRawIntBits(pixels[j]);
					buffer[i]   = (byte)tmp;
					buffer[i+1] = (byte)(tmp>>8);
					buffer[i+2] = (byte)(tmp>>16);
					buffer[i+3] = (byte)(tmp>>24);
					j++;
				}
			else
				for (int i=0; i < count; i+=4) {
					tmp = Float.floatToRawIntBits(pixels[j]);
					buffer[i]   = (byte)(tmp>>24);
					buffer[i+1] = (byte)(tmp>>16);
					buffer[i+2] = (byte)(tmp>>8);
					buffer[i+3] = (byte)tmp;
					j++;
				}
			out.write(buffer, 0, count);
			bytesWritten += count;
		}
	}
	
	@Override
	public void close (){
		closed = true;
	}

	/**
	 * @return the directory
	 */
	public String getDirectory() {
		return directory;
	}

	/**
	 * @param directory the directory to set
	 */
	public void setDirectory(String directory) {
		this.directory = directory;
	}

	/**
	 * @return the prefix
	 */
	public String getPrefix() {
		return prefix;
	}

	/**
	 * @param prefix the prefix to set
	 */
	public void setPrefix(String prefix) {
		this.prefix = prefix;
	}

	/**
	 * @return the format
	 */
	public String getFormat() {
		return format;
	}

	/**
	 * @param format the format to set
	 */
	public void setFormat(String format) {
		this.format = format;
	}

	/**
	 * @return the width
	 */
	public int getWidth() {
		return width;
	}

	/**
	 * @param width the width to set
	 */
	public void setWidth(int width) {
		this.width = width;
	}

	/**
	 * @return the height
	 */
	public int getHeight() {
		return height;
	}

	/**
	 * @param height the height to set
	 */
	public void setHeight(int height) {
		this.height = height;
	}

	/**
	 * @return the littleEndian
	 */
	public boolean isLittleEndian() {
		return littleEndian;
	}

	/**
	 * @param littleEndian the littleEndian to set
	 */
	public void setLittleEndian(boolean littleEndian) {
		this.littleEndian = littleEndian;
	}

}

