//
// Part of the HDF5 plugin for ImageJ
// written by: Olaf Ronneberger (ronneber@informatik.uni-freiburg.de)
// Copyright: GPL v2
//

import ij.plugin.*;
import ij.gui.GenericDialog;
import ij.io.OpenDialog;
import de.unifreiburg.informatik.lmb.HDF5ImageJ;
import ij.Prefs;


public class HDF5_Simple_Reader implements PlugIn 
{
  public void run(String arg) {
    OpenDialog od = new OpenDialog("Load HDF5","","");
    String filename = od.getPath();
    

    GenericDialog gd = new GenericDialog("Load HDF5");
    gd.addMessage("Comma-separated or space-separated list of dataset names.");
    
    String commaSeparatedDsetNames = (String)Prefs.get("hdf5readervibez.dsetnames", "");
    gd.addStringField( "datasetnames", commaSeparatedDsetNames, 128);
    int nFrames = (int)Prefs.get("hdf5readervibez.nframes", 1);
    gd.addNumericField( "nframes", nFrames, 0);
    int nChannels = (int)Prefs.get("hdf5readervibez.nchannels", 1);
    gd.addNumericField( "nchannels", nChannels, 0);
    gd.showDialog();
    if (gd.wasCanceled()) return;

    String datasetnames = gd.getNextString();
    int nframes =  (int)(gd.getNextNumber());
    int nchannels = (int)(gd.getNextNumber());

    HDF5ImageJ.loadDataSetsToHyperStack( filename, 
                                         datasetnames.split(","),
                                         nframes,
                                         nchannels);
    
  }
}