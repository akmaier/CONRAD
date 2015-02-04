


import ij.*;
import ij.io.OpenDialog;

import java.io.File;
import java.io.IOException;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.io.FileProjectionSource;
import edu.stanford.rsl.conrad.io.ImagePlusDataSink;
import edu.stanford.rsl.conrad.io.MKTProjectionSource;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import ij.plugin.*;


public class MKT_Reader extends ImagePlus implements PlugIn {

	public void run(String arg) {
		boolean display = false;
		File test = new File(arg);
		display = !test.exists();
		OpenDialog od = new OpenDialog("Open mkt file...", arg);
		if(od!=null){
			String file = od.getFileName();
			if (file == null) return;
			String directory = od.getDirectory();
			directory = directory.replace('\\', '/'); // Windows safe   
	        if (!directory.endsWith("/")) directory += "/";   
			arg = directory + file;
		}
		
		try {
			FileProjectionSource fileSource = new MKTProjectionSource();
			
			fileSource.initStream(arg);
			ImagePlusDataSink sink = new ImagePlusDataSink();
			Grid2D imp = fileSource.getNextProjection();
			int i =0;
			while (imp != null){
				sink.process(imp, i);
				i++;
				imp = fileSource.getNextProjection();
			}
			sink.close();
			setStack(ImageUtil.wrapGrid3D(sink.getResult(),"").getStack());
			setTitle(new File(arg).getName());
			fileSource.close();	
			if(display) show();
	
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
