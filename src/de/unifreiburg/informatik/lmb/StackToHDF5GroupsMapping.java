//
// Part of the HDF5 plugin for ImageJ
// written by: Olaf Ronneberger (ronneber@informatik.uni-freiburg.de)
// Copyright: GPL v2
//
package de.unifreiburg.informatik.lmb;
public class StackToHDF5GroupsMapping
{
  public String uniqueName_;
  public String formatString_;
  public String formatT_;
  public String formatC_;
  
  public StackToHDF5GroupsMapping( String paramInOneLine) {
    String[] tokens = paramInOneLine.split("[\\s]*,[\\s]*");
    uniqueName_   = tokens[0];
    formatString_ = tokens[1];
    formatT_      = tokens[2];
    formatC_      = tokens[3];
  }
  

  public String toString() {
    return uniqueName_ + ","
        + formatString_ + ","
        + formatT_ + ","
        + formatC_;
  }
}

  
  