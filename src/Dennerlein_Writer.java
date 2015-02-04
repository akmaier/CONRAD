import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.io.FileInfo;
import ij.io.ImageWriter;
import ij.io.SaveDialog;
import ij.plugin.PlugIn;


public class Dennerlein_Writer implements PlugIn {

	public Dennerlein_Writer() {
		// TODO Auto-generated constructor stub
	}
	
	public static byte[] toBytes(short s) {
		return new byte[]{(byte)(s & 0x00FF),(byte)((s & 0xFF00)>>8)};     
	} 

	@Override
	public void run(String arg) {
		ImagePlus imp = WindowManager.getCurrentImage();
		if (imp == null) {
			IJ.showMessage("no image selected");
			return;
		}
		String name = arg;
		if (arg == null || arg.equals("")) {
			name = imp.getTitle();
		}
		SaveDialog sd = new SaveDialog("Write Dennerlein Format...", name, ".bin");
		String file = sd.getFileName();
		String directory = sd.getDirectory();
		if (file == null) return;
		FileInfo fi = imp.getFileInfo();
		// Make sure that we can save this kind of image
		fi.fileFormat = FileInfo.RAW;
		fi.compression = FileInfo.COMPRESSION_NONE;
		fi.fileType = FileInfo.GRAY32_FLOAT;
		fi.intelByteOrder = true;
		// Set the fileName stored in the file info record to the
		// file name that was passed in or chosen in the dialog box
		fi.fileName=file;
		fi.directory=directory;
		// TODO Auto-generated method stub
		FileOutputStream out;
		try {
			out = new FileOutputStream(new File(directory, file));
			out.write(toBytes((short) imp.getWidth()));
			out.write(toBytes((short) imp.getHeight()));
			out.write(toBytes((short) imp.getNSlices()));
			ImageWriter writer = new ImageWriter(fi);
			writer.write(out);
			out.close();
			IJ.showStatus("Saved "+ fi.fileName);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
