import ij.ImagePlus;
import ij.WindowManager;
import ij.io.SaveDialog;
import ij.plugin.PlugIn;
import edu.stanford.rsl.conrad.io.SEQProjectionSource;


public class SaveAs_Viva implements PlugIn {

	public void run(String arg) {   
        ImagePlus imp = WindowManager.getCurrentImage();   
        if (null == imp) return;   
        SaveDialog sd = new SaveDialog("Save Viva File", "copy_"+imp.getTitle(), ".viv");  
        if(sd.getDirectory()!=null & sd.getFileName()!=null){
            String dir = sd.getDirectory();   
            if (null == dir) return; // user canceled dialog   
            dir = dir.replace('\\', '/'); // Windows safe   
            if (!dir.endsWith("/")) dir += "/";   

            new SEQProjectionSource().saveViva(imp, dir + sd.getFileName());   
        }

    }   
  
  
}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/