package edu.stanford.rsl.tutorial.basics;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import ij.ImageJ;
import ij.ImagePlus;
import ij.io.FileInfo;
import ij.io.FileOpener;

public class MHDImageLoader {



	public MHDImageLoader() {	


	}

	public Grid3D loadImage(String MHDfilename){
		int width = 512;
		int height = 512;
		int offset = 0;
		int nImages = 500;
		int fileType = FileInfo.GRAY16_UNSIGNED;
		boolean intelByteOrder = true;
		double offsets [] = null;
		double spacings []  = null;
		String datafile = "";

		try {
			BufferedReader br = new BufferedReader(new FileReader(MHDfilename));
			String line = "a";
			while (line != null){
				line = br.readLine();
				//System.out.println(line);
				if (line != null ) {
					String [] split = line.split(" = ");
					if (line.contains("BinaryDataByteOrderMSB")){
						boolean value = Boolean.parseBoolean(split[1]);
						intelByteOrder = !value;
					}
					if (line.contains("Offset")){
						String [] split2 = split[1].split(" ");
						offsets = new double [split2.length];
						for (int i=0; i < split2.length; i++){
							offsets[i] = Double.parseDouble(split2[i]);
						}
					}
					if (line.contains("ElementSpacing")){
						String [] split2 = split[1].split(" ");
						spacings = new double [split2.length];
						for (int i=0; i < split2.length; i++){
							spacings[i] = Double.parseDouble(split2[i]);
						}
					}
					if (line.contains("DimSize")){
						String [] split2 = split[1].split(" ");
						width = Integer.parseInt(split2[0]);
						height = Integer.parseInt(split2[1]);
						nImages = Integer.parseInt(split2[2]);
					}
					if (line.contains("ElementType")){
						if (split[1].equals("MET_USHORT")) fileType = FileInfo.GRAY16_UNSIGNED;
					}
					if (line.contains("ElementDataFile")){
						datafile = split[1];
					}
				}
			}
			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}


		FileInfo fi = new FileInfo();
		fi.width = width;
		fi.height = height;
		fi.offset = offset;
		fi.nImages = nImages;
		fi.fileType = fileType;
		fi.intelByteOrder = intelByteOrder;
		fi.fileFormat = FileInfo.RAW;
		fi.fileName = datafile;
		fi.directory = new File(MHDfilename).getParent();
		
		ImagePlus img = new FileOpener(fi).open(false);
		
		Grid3D grid = ImageUtil.wrapImagePlus(img, false, true);

		//grid.setOrigin(offsets);
		grid.setSpacing(spacings);
		return grid;
	}

	public static void main(String [] args){
		new ImageJ();
		MHDImageLoader loader = new MHDImageLoader();
		
		Grid3D image = loader.loadImage(args[0]);
		image.show("Loaded Data");
	}

}
