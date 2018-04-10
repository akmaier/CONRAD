//
// Part of the HDF5 plugin for ImageJ
// written by: Olaf Ronneberger (ronneber@informatik.uni-freiburg.de)
// Copyright: GPL v2
//

import de.unifreiburg.informatik.lmb.HDF5ImageJ;
import ij.ImagePlus;
import ij.plugin.*;
import ij.plugin.filter.PlugInFilter;
import ij.gui.GenericDialog;
import ij.io.OpenDialog;
import ij.io.SaveDialog;
import ij.process.ImageProcessor;
import ij.Prefs;

public class HDF5_Simple_Writer implements PlugInFilter
{
  String _saveMode;  // either "replace" or "append"
  ImagePlus _imp;

  public int setup(String arg, ImagePlus imp) 
  {
    _saveMode = arg;
    _imp = imp;
    return DOES_8G + DOES_8C + DOES_16 + DOES_32 + DOES_RGB + NO_CHANGES;
  }
  
  public void run(ImageProcessor ip) 
  {
    String filename;
    if( _saveMode.equals("append")) 
    {
      OpenDialog sd = new OpenDialog("Save to HDF5 (append) ...", OpenDialog.getLastDirectory(), "");
      String directory = sd.getDirectory();
      String name = sd.getFileName();
      if (name == null)
          return;
      if (name == "")
          return;
      filename = directory + name;
    }
    else
    {
      SaveDialog sd = new SaveDialog("Save to HDF5 (new or replace)...", OpenDialog.getLastDirectory(), ".h5");
      String directory = sd.getDirectory();
      String name = sd.getFileName();
      if (name == null)
          return;
      if (name == "")
          return;
      filename = directory + name;
    }
    
    GenericDialog gd = new GenericDialog("Save HDF5");
    gd.addMessage("Data set name template (for hyperstacks should contain placeholders '{t}' and '{c}')" );
    String dsetNameTemplate = (String)Prefs.get("hdf5writervibez.nametemplate", "/t{t}/channel{c}");
    gd.addStringField( "dsetnametemplate", dsetNameTemplate, 128);


    gd.addMessage("Format string for frame number. Either printf syntax (e.g. '%d') or comma-separated list of names." );
    String formatTime = (String)Prefs.get("hdf5writervibez.timeformat", "%d");
    gd.addStringField( "formattime", formatTime, 128);

    gd.addMessage("Format string for channel number. Either printf syntax (e.g. '%d') or comma-separated list of names." );
    String formatChannel = (String)Prefs.get("hdf5writervibez.timeformat", "%d");
    gd.addStringField( "formatchannel", formatChannel, 128);

    gd.addMessage("Compression level (0-9)");
    int compressionLevel = (int)Prefs.get("hdf5writervibez.compressionlevel", 0);
    gd.addNumericField( "compressionlevel", compressionLevel, 0);
    gd.showDialog();
    if (gd.wasCanceled()) return;
    
    dsetNameTemplate = gd.getNextString();
    formatTime = gd.getNextString();
    formatChannel = gd.getNextString();
    compressionLevel = (int)(gd.getNextNumber());

    HDF5ImageJ.saveHyperStack( _imp, filename, dsetNameTemplate, 
                               formatTime, formatChannel, 
                               compressionLevel, _saveMode);
  }
}
