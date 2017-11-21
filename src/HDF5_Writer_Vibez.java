//
// Part of the HDF5 plugin for ImageJ
// written by: Olaf Ronneberger (ronneber@informatik.uni-freiburg.de)
// Copyright: GPL v2
//

import ij.*;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.WindowManager;
import ij.gui.*;
import ij.gui.GenericDialog;
import ij.io.DirectoryChooser;
import ij.io.Opener;
import ij.io.OpenDialog;
import ij.io.SaveDialog;
import ij.plugin.*;
import ij.plugin.filter.*;
import ij.plugin.filter.PlugInFilter;
import ij.plugin.frame.PlugInFrame;
import ij.process.*;
import ij.process.ImageProcessor;
import ij.Prefs;


import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.io.File;


import javax.swing.*;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.table.AbstractTableModel;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.TableModel;
import javax.swing.text.*;
import javax.swing.tree.DefaultMutableTreeNode;

import ncsa.hdf.hdf5lib.exceptions.HDF5Exception;
import ch.systemsx.cisd.hdf5.HDF5Factory;
import ch.systemsx.cisd.hdf5.IHDF5Writer;
import ch.systemsx.cisd.hdf5.HDF5IntStorageFeatures;
import ch.systemsx.cisd.hdf5.HDF5FloatStorageFeatures;
import ch.systemsx.cisd.base.mdarray.MDByteArray;
import ch.systemsx.cisd.base.mdarray.MDShortArray;
import ch.systemsx.cisd.base.mdarray.MDFloatArray;
import ch.systemsx.cisd.hdf5.HDF5FactoryProvider;
import ch.systemsx.cisd.hdf5.IHDF5WriterConfigurator;
import de.unifreiburg.informatik.lmb.HDF5ImageJ;
import de.unifreiburg.informatik.lmb.StackToHDF5GroupsMapping;


public class HDF5_Writer_Vibez extends java.awt.Frame implements PlugInFilter, ActionListener 
{
  // private Variables
  //
  String _filename;
  String _saveMode;  // either "replace" or "append"
  
  ImagePlus _imp;
  BoxLayout  _mylayout;
  JComboBox  _compressionSelect;
  JComboBox  _presetSelect;
  JTextField _dsetNameTempl;
  JTextArea  _textAreaT;
  JTextArea  _textAreaC;
  JLabel  _dsetNamesPreview;
//  String[] _presetStrings;
  ArrayList<StackToHDF5GroupsMapping> _mappingPresets;
  
  
  public HDF5_Writer_Vibez()
        {
          super("Save to HDF5");
          _mappingPresets = new ArrayList<StackToHDF5GroupsMapping>();
          _mappingPresets.add(new StackToHDF5GroupsMapping(
                                  ",/t{t}/channel{c},%d,%d"));
          _mappingPresets.add(new StackToHDF5GroupsMapping(
                                  "Standard,/t{t}/channel{c},%d,%d"));
          _mappingPresets.add(new StackToHDF5GroupsMapping(
                                  "Standard (no time),/channel{c},%d,%d"));
          _mappingPresets.add(new StackToHDF5GroupsMapping(
                                  "LSM --> ViBE-Z step0 dorsal,/step0/raw/dorsal/tile{t}/{c},%d,laserset0/channel1 laserset1/channel1 laserset0/channel0 laserset1/channel0"));
          _mappingPresets.add(new StackToHDF5GroupsMapping(
                                  "LSM --> ViBE-Z step0 ventral,/step0/raw/ventral/tile{t}/{c},%d,laserset0/channel1 laserset1/channel1 laserset0/channel0 laserset1/channel0"));
          _mappingPresets.add(new StackToHDF5GroupsMapping(
                                  "Zen --> ViBE-Z step0 dorsal tile 0,/step0/raw/dorsal/tile{t}/{c},0,laserset0/channel0 laserset1/channel0 laserset0/channel1 laserset1/channel1"));
          _mappingPresets.add(new StackToHDF5GroupsMapping(
                                  "Zen --> ViBE-Z step0 dorsal tile 1,/step0/raw/dorsal/tile{t}/{c},1,laserset0/channel0 laserset1/channel0 laserset0/channel1 laserset1/channel1"));
          _mappingPresets.add(new StackToHDF5GroupsMapping(
                                  "Zen --> ViBE-Z step0 ventral tile 0,/step0/raw/ventral/tile{t}/{c},0,laserset0/channel0 laserset1/channel0 laserset0/channel1 laserset1/channel1"));
          _mappingPresets.add(new StackToHDF5GroupsMapping(
                                  "Zen --> ViBE-Z step0 ventral tile 1,/step0/raw/ventral/tile{t}/{c},1,laserset0/channel0 laserset1/channel0 laserset0/channel1 laserset1/channel1"));
        }


  public int setup(String arg, ImagePlus imp) 
        {
          _saveMode = arg;
//          IJ.log("save mode: "+_saveMode+"\n");
          _imp = imp;
          return DOES_8G + DOES_8C + DOES_16 + DOES_32 + DOES_RGB + NO_CHANGES;
        }

  public void run(ImageProcessor ip) 
  {
    
    // File Dialog
    //
    String hint = "";
    if( _saveMode.equals("append")) 
    {
      OpenDialog sd = new OpenDialog("Save to HDF5 (append) ...", OpenDialog.getLastDirectory(), "");
      String directory = sd.getDirectory();
      String name = sd.getFileName();
      if (name == null)
          return;
      if (name == "")
          return;
      _filename = directory + name;
          
      if( new File(_filename).exists())
      {
        setTitle("Append to Existing HDF5 File '"+name+"'");
        hint = "<font color='red'>(Append to existing HDF File '"+name+"')</font>"; 
      }
      else
      {
        setTitle("Save to New HDF5 File '"+name+"'");
      }
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
      _filename = directory + name;
          
      
      setTitle("Save to New HDF5 File '"+name+"'");
    }
      
    

    int nFrames = _imp.getNFrames();
    int nChannels = _imp.getNChannels();

    setLayout(new GridBagLayout());
    GridBagConstraints cs = new GridBagConstraints();
    int currentRow = 0;
    
    cs.anchor = GridBagConstraints.FIRST_LINE_START;

    //natural height, maximum width
    cs.fill = GridBagConstraints.HORIZONTAL;
    cs.weightx = 0;
    cs.weighty = 0;
    cs.gridx = 0;
    cs.gridy = currentRow;  
    cs.insets = new Insets(3,3,0,0);
    cs.gridwidth = 3;
    add(new JLabel("<html>"+hint+"<h2>Select Compression (Lossless gzip algorithm)</h2></html>"), cs);

    String[] compressionLevels = {
        "no compression",
        "1 (fastest, larger file)", 
        "2", "3", "4", "5", "6", "7", "8", 
        "9 (slowest, smallest file)"};
    _compressionSelect = new JComboBox( compressionLevels);
    int compressionLevel = (int)Prefs.get("hdf5writervibez.compressionlevel", 0);
    _compressionSelect.setSelectedIndex(compressionLevel);
    //   _compressionSelect.setActionCommand( "set_compression");
    //   _compressionSelect.addActionListener(this);
    JLabel compressionLabel = new JLabel("Compression Level: ");
    cs.fill = GridBagConstraints.NONE;
    cs.gridx = 0;
    cs.gridy = ++currentRow;   
    cs.gridwidth = 1;
    add(compressionLabel, cs);
    cs.fill = GridBagConstraints.NONE;
    cs.gridx = 1;
    cs.gridy = currentRow;   
    cs.gridwidth = 2;
    add(_compressionSelect, cs);

    //natural height, maximum width
    cs.fill = GridBagConstraints.HORIZONTAL;
    cs.weightx = 0;
    cs.weighty = 0;
    cs.gridx = 0;
    cs.gridy = ++currentRow;  
    cs.insets = new Insets(3,3,0,0);
    cs.gridwidth = 3;
    add(new JLabel("<html><h2>Specify the dataset names for each 3D stack in the HDF5 file</h2></html>"), cs);


    
    _presetSelect = new JComboBox();
    for (int i = 0; i < _mappingPresets.size(); ++i)
    {
      _presetSelect.addItem( _mappingPresets.get(i).uniqueName_);
    }
    

    _presetSelect.setSelectedIndex(0);
    _presetSelect.setActionCommand( "set_preset");
    _presetSelect.addActionListener(this);
    JLabel presetLabel = new JLabel("Presets: ");
    //   presetLabel.setLabelFor(_presetSelect);
    cs.fill = GridBagConstraints.NONE;
    cs.gridx = 0;
    cs.gridy = ++currentRow;   
    cs.gridwidth = 1;
    add(presetLabel, cs);
    cs.fill = GridBagConstraints.NONE;
    cs.gridx = 1;
    cs.gridy = currentRow;   
    cs.gridwidth = 2;
    add(_presetSelect, cs);
     
    
    _dsetNameTempl = new JTextField((String)Prefs.get("hdf5writervibez.nametemplate", "/t{t}/channel{c}"));
    _dsetNameTempl.setActionCommand("update_preview");
    _dsetNameTempl.addActionListener(this);
    JLabel textFieldLabel = new JLabel("Dataset Names Template: ");
//    textFieldLabel.setLabelFor(textField);
    cs.gridx = 0;
    cs.gridy = ++currentRow;   
    cs.gridwidth = 1;
    add(textFieldLabel, cs);
    cs.fill = GridBagConstraints.HORIZONTAL;
    cs.gridx = 1;
    cs.gridy = currentRow;   
    cs.gridwidth = 2;
    add(_dsetNameTempl, cs);
    
    cs.fill = GridBagConstraints.NONE;
    cs.gridx = 1;
    cs.gridy = ++currentRow;   
    cs.gridwidth = 1;
    add(new JLabel("Replace {t} with: (Your data has "+nFrames+" time points)"),cs);

    cs.gridx = 2;
    cs.gridy = currentRow;   
    cs.gridwidth = 1;
    add(new JLabel("Replace {c} with: (Your data has "+nChannels+" channels)"),cs);

    _textAreaT = new JTextArea((String)Prefs.get("hdf5writervibez.timeformat", "%d"));
    JScrollPane scrollPaneT = new JScrollPane(_textAreaT);
    scrollPaneT.setMinimumSize(new Dimension(1,100));
    cs.weightx = 1;
    cs.weighty = 1;
    cs.gridx = 1;
    cs.gridy = ++currentRow;   
    cs.gridwidth = 1;
    cs.fill = GridBagConstraints.BOTH;
    add(scrollPaneT,cs);
    
    _textAreaC = new JTextArea((String)Prefs.get("hdf5writervibez.channelformat", "%d"));
    JScrollPane scrollPaneC = new JScrollPane(_textAreaC);
    scrollPaneC.setMinimumSize(new Dimension(1,100));
    cs.gridx = 2;
    cs.gridy = currentRow;   
    cs.gridwidth = 1;
    add(scrollPaneC,cs);


    JButton b0 = new JButton("Update Preview");
    b0.setActionCommand("update_preview");
    b0.addActionListener(this);
    cs.weightx = 0;
    cs.weighty = 0;
    cs.gridx = 1;
    cs.gridy = ++currentRow;   
    cs.gridwidth = 2;
    cs.fill = GridBagConstraints.NONE;
    add(b0,cs);


    cs.fill = GridBagConstraints.NONE;
    cs.weightx = 0;
    cs.weighty = 1;
    cs.gridx = 0;
    cs.gridy = ++currentRow;   
    cs.gridwidth = 1;
    add(new JLabel("Resulting\nMapping:"),cs);

    _dsetNamesPreview = new JLabel();
    JScrollPane scrollPanePreview = new JScrollPane(_dsetNamesPreview);
    scrollPanePreview.setMinimumSize(new Dimension(1,100));
    cs.gridx = 1;
    cs.gridy = currentRow;   
    cs.gridwidth = 2;
    cs.fill = GridBagConstraints.BOTH;
    add(scrollPanePreview,cs);

    JButton b1 = new JButton("Save");
    b1.setActionCommand("save");
    b1.addActionListener(this);
    cs.weightx = 0;
    cs.weighty = 0;
    cs.gridx = 1;
    cs.gridy = ++currentRow;   
    cs.gridwidth = 1;
    cs.fill = GridBagConstraints.NONE;
    add(b1,cs);

    JButton b2 = new JButton("Cancel");
    b2.setActionCommand("cancel");
    b2.addActionListener(this);
    cs.gridx = 2;
    cs.gridy = currentRow;   
    cs.gridwidth = 1;
    add(b2,cs);

    pack();
     
    Dimension si = getSize();
    si.height=400;
    setSize( si);
    setVisible(true);
    
//    setPreset( 0);
    updateMapping();
    
  }
  
  public void actionPerformed(ActionEvent event) 
  {
    if (event.getActionCommand().equals("set_preset")) 
    {
      setPreset( _presetSelect.getSelectedIndex());
      
    }
    else if (event.getActionCommand().equals("update_preview")) 
    {
      updateMapping();
    }
    else if (event.getActionCommand().equals("save")) 
    {
      saveHDF5();
    }
    else if (event.getActionCommand().equals("cancel")) 
    {
      dispose();
    }
  }

  public void setPreset(int index)
  {
    StackToHDF5GroupsMapping m = _mappingPresets.get(index);
    _dsetNameTempl.setText( m.formatString_);
    String t1 = m.formatT_.replaceAll("[,\\s]+", "\n");
    String t2 = m.formatC_.replaceAll("[,\\s]+", "\n");
    _textAreaT.setText(t1);
    _textAreaC.setText(t2);
    updateMapping();
    
  }
  

  public void updateMapping()
  {
    String formatString= _dsetNameTempl.getText();
    String formatT = _textAreaT.getText();
    String formatC = _textAreaC.getText();
    int nFrames = _imp.getNFrames();
    int nChannels = _imp.getNChannels();
   
//   String[] substT = new String[nFrames]; 
//   if( formatT.startsWith("%"))
//   {
//     for( int t=0; t < nFrames; ++t)
//     {
//       substT[t] = String.format(formatT, t);
//     }
//   }
//   else
//   {
//     substT = formatT.split("[,\\s]+");
//   }
//   String[] substC = new String[nChannels]; 
//   if( formatC.startsWith("%"))
//   {
//     for( int c=0; c < nChannels; ++c)
//     {
//       substC[c] = String.format(formatC, c);
//     }
//   }
//   else
//   {
//     substC = formatC.split("[,\\s]+");
//   }

    String[] substT = HDF5ImageJ.createNameList( formatT, nFrames);
    String[] substC = HDF5ImageJ.createNameList( formatC, nChannels);
    
  
    String[] dataSetNames = null;
    String preview = new String();
    preview += "<html>";
    
    for( int t=0; t < nFrames; ++t)
    {
      for( int c=0; c < nChannels; ++c)
      {
        String dsetName = formatString;
        dsetName = dsetName.replace("{t}", substT[t]);
        dsetName = dsetName.replace("{c}", substC[c]);
        preview += "t:" + t + ",c:" + c + "  &rarr;  "+dsetName + "<br>";
      }
    }
    preview += "</html>";
    
    _dsetNamesPreview.setText(preview);
    
  }
  
  
  public void saveHDF5()
  {
    //
    // Create data set names from user input
    //
    String dsetNameTemplate = _dsetNameTempl.getText();
    String formatT      = _textAreaT.getText();
    String formatC      = _textAreaC.getText();
    int compressionLevel = _compressionSelect.getSelectedIndex();
 
    // store as preferences for next call
    Prefs.set("hdf5writervibez.nametemplate",dsetNameTemplate);
    Prefs.set("hdf5writervibez.timeformat",formatT);
    Prefs.set("hdf5writervibez.channelformat",formatC);
    Prefs.set("hdf5writervibez.compressionlevel", compressionLevel);

    HDF5ImageJ.saveHyperStack( _imp, _filename, dsetNameTemplate, 
                               formatT, formatC, compressionLevel, _saveMode);
    dispose();

   
//   int nFrames   = _imp.getNFrames();
//   int nChannels = _imp.getNChannels();
//   int nLevs     = _imp.getNSlices();
//   int nRows     = _imp.getHeight();
//   int nCols     = _imp.getWidth();
//
//   // Name stubs for time points
//   String[] substT = new String[nFrames]; 
//   if( formatT.startsWith("%"))
//   {
//     for( int t=0; t < nFrames; ++t)
//     {
//       substT[t] = String.format(formatT, t);
//     }
//   }
//   else
//   {
//     substT = formatT.split("\\s");
//   }
//   
//   // Name stubs for channels
//   String[] substC = new String[nChannels]; 
//   if( formatC.startsWith("%"))
//   {
//     for( int c=0; c < nChannels; ++c)
//     {
//       substC[c] = String.format(formatC, c);
//     }
//   }
//   else
//   {
//     substC = formatC.split("\\s");
//   }
// 
//   //
//   //  Open output file
//   //
//   try
//   {
//     IHDF5Writer writer;
//     if( _saveMode.equals( "append")) 
//     {
//       writer = HDF5Factory.configure(_filename).useSimpleDataSpaceForAttributes().writer();
//     }
//     else
//     {
//       writer = HDF5Factory.configure(_filename).useSimpleDataSpaceForAttributes().overwrite().writer();
//     }
//     
//
//     //  get element_size_um
//     //
//     ij.measure.Calibration cal = _imp.getCalibration();
//     float[] element_size_um = new float[3];
//     element_size_um[0] = (float) cal.pixelDepth;
//     element_size_um[1] = (float) cal.pixelHeight;
//     element_size_um[2] = (float) cal.pixelWidth;
//
//     //
//     //  loop through frames and channels
//     //
//     for( int t=0; t < nFrames; ++t)
//     {
//       for( int c=0; c < nChannels; ++c)
//       {
//         // format the data set name
//         //
//         String dsetName = formatString;
//         dsetName = dsetName.replace("{t}", substT[t]);
//         dsetName = dsetName.replace("{c}", substC[c]);
//
//         System.out.println( "t="+t+",c="+c+" --> "+dsetName);
//
//         // write Stack according to data type
//         // 
//         int imgColorType = _imp.getType();
//         int compressionLevel = _compressionSelect.getSelectedIndex();
//         Prefs.set("hdf5writervibez.compressionlevel", compressionLevel);
//         
//         if (imgColorType == ImagePlus.GRAY8 
//             || imgColorType == ImagePlus.COLOR_256 ) 
//         {
//           writeGray8Stack( _imp, t, c, writer, dsetName,
//                            HDF5IntStorageFeatures.createDeflationDelete(
//                                compressionLevel));
//         } 
//         else if (imgColorType == ImagePlus.GRAY16) 
//         {
//           writeGray16Stack( _imp, t, c, writer, dsetName,
//                            HDF5IntStorageFeatures.createDeflationDelete(
//                                compressionLevel));
//         }
//         else if (imgColorType == ImagePlus.GRAY32)
//         {
//           writeFloatStack( _imp, t, c, writer, dsetName,
//                            HDF5FloatStorageFeatures.createDeflationDelete(
//                                compressionLevel));
//         }
//         else if (imgColorType == ImagePlus.COLOR_RGB)
//         {
//           writeRGBStack( _imp, t, c, writer, dsetName,
//                            HDF5IntStorageFeatures.createDeflationDelete(
//                                compressionLevel));
//         }
//           
//         
//         //  add element_size_um attribute 
//         //
//         writer.float32().setArrayAttr( dsetName, "element_size_um", 
//                                        element_size_um);
//
//         
//       }
//     }
//     writer.close();
//     dispose();
//   }    
//
//   
//   catch (HDF5Exception err) 
//   {
//     IJ.error("Error while saving '" + _filename + "':\n"
//              + err);
//   } 
//   catch (Exception err) 
//   {
//     IJ.error("Error while saving '" + _filename + "':\n"
//              + err);
//   } 
//   catch (OutOfMemoryError o) 
//   {
//     IJ.outOfMemory("Error while saving '" + _filename + "'");
//   }
  }
  
    

}
