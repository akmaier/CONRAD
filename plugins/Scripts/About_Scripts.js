  importClass(java.io.File);
  path = IJ.getDirectory("plugins")+"Scripts/About Scripts";
  if (!(new File(path)).exists())
     IJ.error("\"About Scripts\" not found in ImageJ/plugins/Scripts.");
  IJ.open(path);

