// Part of the HDF5 plugin for ImageJ
// written by: Olaf Ronneberger (ronneber@informatik.uni-freiburg.de)
// Copyright: GPL v2
//



import ij.CompositeImage;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.Prefs;
import ij.gui.GenericDialog;
import ij.io.OpenDialog;
import ij.plugin.PlugIn;
import ij.process.ColorProcessor;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;
import ij.process.ImageStatistics;
import ij.process.StackStatistics;
import ij.measure.Measurements;

import java.io.File;
import java.util.Date;
import java.util.List;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Vector;

import java.awt.*;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import javax.swing.*;
import javax.swing.tree.*;

import com.sun.tracing.dtrace.ArgsAttributes;

import javax.swing.JTable;
import javax.swing.table.AbstractTableModel;
import javax.swing.table.DefaultTableModel;
import javax.swing.JCheckBox;


import ch.systemsx.cisd.hdf5.HDF5DataSetInformation;
import ch.systemsx.cisd.hdf5.HDF5DataTypeInformation;

import ch.systemsx.cisd.hdf5.HDF5Factory;
import ch.systemsx.cisd.hdf5.HDF5LinkInformation;
import ch.systemsx.cisd.hdf5.IHDF5Reader;
import ch.systemsx.cisd.hdf5.IHDF5Writer;
import de.unifreiburg.informatik.lmb.HDF5ImageJ;
import de.unifreiburg.informatik.lmb.TableColumnAdjuster;
import ch.systemsx.cisd.hdf5.IHDF5ReaderConfigurator;
import ch.systemsx.cisd.base.mdarray.MDByteArray;
import ch.systemsx.cisd.base.mdarray.MDFloatArray;
import ch.systemsx.cisd.base.mdarray.MDShortArray;
import ncsa.hdf.hdf5lib.exceptions.HDF5Exception; 



public class HDF5_Reader_Vibez extends JFrame  implements PlugIn, ActionListener 
{
	class DataSetInfo
	{ 
		public String path;
		public String numericSortablePath;
		public String dimText;
		public String typeText;
		public String element_size_um_text;
		final int numPaddingSize = 10;

		public DataSetInfo( String p, String d, String t, String e) {
			setPath(p);
			dimText = d;
			typeText = t; 
			element_size_um_text = e;
		}

		public void setPath( String p) {
			path = p;
			numericSortablePath = "";
			String num = "";
			for( int i = 0; i < p.length(); ++i) {
				if (isNum(p.charAt(i))) {
					num += p.charAt(i);
				} else {
					if (num != "") {
						for (int j = 0; j < numPaddingSize - num.length(); ++j) {
							numericSortablePath += "0";
						}
						numericSortablePath += num;
						num = "";
					}
					numericSortablePath += p.charAt(i);
				}
			}
			if (num != "") {
				for (int j = 0; j < numPaddingSize - num.length(); ++j) {
					numericSortablePath += "0";
				}
				numericSortablePath += num;
			}
			IJ.log( path);
			IJ.log( numericSortablePath);
		}

		private boolean isNum( char c) {
			return c >= '0' && c <= '9';
		}
	}

	class DataSetInfoComparator implements Comparator<DataSetInfo> {
		public int compare( DataSetInfo a, DataSetInfo b) {
			return a.numericSortablePath.compareTo( b.numericSortablePath);
		}
	}

	//  Private Members
	// 
	private ArrayList<DataSetInfo> dataSets_;
	private JTable pathTable_;
	private String fullFileName_;
	//  private JCheckBox loadAsHyperstackCheckBox_;
	private JRadioButton[] loadAsRadioButtons_;
	private int loadAsMode_;

	private SpinnerNumberModel nChannelsSpinner_;
	private JTextField dsetLayoutTextField_;

	public void run(String arg) 
	{

		// Let User select the filename
		//
		String directory = "";
		String name = "";
		if (!arg.equals("")) {
			File file = new File(arg);
			name = file.getName();
			directory= file.getParent()+File.separator;
		} else {
			boolean tryAgain;
			String openMSG = "Open HDF5...";
			do {
				tryAgain = false;
				OpenDialog od;
				if (directory.equals(""))
					od = new OpenDialog(openMSG, "");
				else
					od = new OpenDialog(openMSG, directory, "");

				directory = od.getDirectory();
				name = od.getFileName();
				if (name == null)
					return;
				if (name == "")
					return;

				File testFile = new File(directory + name);
				if (!testFile.exists() || !testFile.canRead())
					return;

				if (testFile.isDirectory()) {
					directory = directory + name;
					tryAgain = true;
				}
			} while (tryAgain);
		}

		// Get All Dataset names
		//
		fullFileName_ = directory + name;
		IJ.showStatus("Loading HDF5 File: " + fullFileName_);
		IHDF5Reader reader = HDF5Factory.openForReading(fullFileName_);
		dataSets_ = new ArrayList<DataSetInfo>();
		recursiveGetInfo( reader, reader.object().getLinkInformation("/"));
		//    DefaultMutableTreeNode root = browse(reader, reader.object().getLinkInformation("/"));
		//    WindowUtilities.setNativeLookAndFeel();
		//    addWindowListener(new ExitListener());


		//    Container content = getContentPane();
		//    JTree tree = new JTree(root);
		//    content.add(new JScrollPane(tree), BorderLayout.CENTER);
		reader.close();

		// print all dataset infos for debuuging
		Collections.sort( dataSets_, new DataSetInfoComparator());
		IJ.log( "ALL DATASETS:");
		for (DataSetInfo info : dataSets_)
		{
			IJ.log(info.path + " (" + info.dimText + " " + info.typeText + ")");
		}



		// Create a JTable from the data set list
		Vector<Vector> tableData = new Vector<Vector>();
		for (int row = 0; row < dataSets_.size(); ++row) {
			Vector<String> line = new Vector<String>();
			line.addElement("<html>"+dataSets_.get(row).path.replace("/", "<font color='red'><strong>/</strong></font>")+"</html>");
			line.addElement("<html>"+dataSets_.get(row).dimText.replace("x", "<font color='red'>&times;</font>")+"</html>");
			line.addElement(dataSets_.get(row).typeText);
			line.addElement("<html>"+dataSets_.get(row).element_size_um_text.replace("x", "<font color='red'>&times;</font>")+"</html>");
			tableData.addElement( line);
		}

		String[] columnTitles = {"path", "size", "type"};
		Vector<String> columnNames = new Vector<String>();
		columnNames.addElement("data set path");
		columnNames.addElement("size");
		columnNames.addElement("type");
		columnNames.addElement("element size [um]");


		// make table non-editable
		DefaultTableModel tableModel = new DefaultTableModel(tableData, columnNames) {
			@Override
			public boolean isCellEditable(int row, int column) {
				//all cells false
				return false;
			}
		};
		pathTable_ = new JTable( tableModel); 

		setLayout(new GridBagLayout());
		GridBagConstraints cs = new GridBagConstraints();
		int currentRow = 0;

		cs.anchor = GridBagConstraints.FIRST_LINE_START;

		//natural height, maximum width
		cs.fill = GridBagConstraints.HORIZONTAL;
		cs.ipady = 10;      //make this component tall
		cs.weightx = 0;
		cs.weighty = 0;
		cs.gridx = 0;
		cs.gridy = currentRow;  
		cs.insets = new Insets(3,3,0,0);
		cs.gridwidth = 2;
		JLabel titleText = new JLabel("<html><h2>Select data sets</h2></html>");
		add(titleText, cs);

		//maximum height, maximum width
		cs.fill = GridBagConstraints.BOTH;
		cs.ipady = 100;      //make this component tall
		cs.weightx = 1;
		cs.weighty = 1;
		cs.gridx = 0;
		cs.gridy = ++currentRow;  
		cs.insets = new Insets(3,3,0,0);
		cs.gridwidth = 2;
		JScrollPane scrollPaneT = new JScrollPane(pathTable_);
		scrollPaneT.setMinimumSize(new Dimension(1,100));
		add( scrollPaneT, cs);

		pathTable_.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);
		TableColumnAdjuster tca = new TableColumnAdjuster(pathTable_);
		tca.adjustColumns();

		loadAsRadioButtons_ = new JRadioButton[5];
		loadAsRadioButtons_[0] = new JRadioButton("individual stacks");
		loadAsRadioButtons_[1] = new JRadioButton("individual hyperstacks (custom layout)");
		loadAsRadioButtons_[2] = new JRadioButton("hyperstack (multichannel)");
		loadAsRadioButtons_[3] = new JRadioButton("hyperstack (time series)");
		loadAsRadioButtons_[4] = new JRadioButton("hyperstack (multichannel time series)");
		int loadAsMode = (int)Prefs.get("hdf5readervibez.loadasmode", 0);
		loadAsRadioButtons_[loadAsMode].setSelected(true);
		ButtonGroup group = new ButtonGroup();
		for( int i = 0; i < 5; ++i) 
		{
			group.add(loadAsRadioButtons_[i]);
		}


		//    loadAsHyperstackCheckBox_ = new JCheckBox( "Combine Stacks to Hyperstack", true);
		cs.fill = GridBagConstraints.HORIZONTAL;
		cs.ipady = 0; 
		cs.weightx = 0;
		cs.weighty = 0;
		cs.gridx = 0;
		cs.gridy = ++currentRow;  
		cs.insets = new Insets(3,3,0,0);
		cs.gridwidth = 2;
		JLabel subtitleText = new JLabel("Load as ...");
		add(subtitleText, cs);
		for( int i = 0; i < 2; ++i)
		{
			cs.gridy = ++currentRow;  
			add(loadAsRadioButtons_[i], cs);
		}

		JLabel layoutText = new JLabel("       - data set layout:");
		cs.weightx = 1;
		cs.weighty = 0;
		cs.gridx = 0;
		cs.gridy = ++currentRow;  
		cs.insets = new Insets(3,3,0,0);
		cs.gridwidth = 1;
		//natural height, natural width
		cs.fill = GridBagConstraints.NONE;
		add(layoutText, cs);

		cs.fill = GridBagConstraints.HORIZONTAL;
		String dsetLayout = Prefs.get("hdf5readervibez.dsetLayout", "zyx");
		dsetLayoutTextField_ = new JTextField(dsetLayout, 6); 
		cs.gridx = 1;
		add(dsetLayoutTextField_, cs);

		cs.fill = GridBagConstraints.HORIZONTAL;
		cs.ipady = 0; 
		cs.weightx = 0;
		cs.weighty = 0;
		cs.gridx = 0;
		cs.gridy = ++currentRow;  
		cs.insets = new Insets(3,3,0,0);
		cs.gridwidth = 2;
		JLabel subtitleText2 = new JLabel("Combine to ...");
		cs.gridy = ++currentRow;  
		add(subtitleText2, cs);
		for( int i = 2; i < 5; ++i)
		{
			cs.gridy = ++currentRow;  
			add(loadAsRadioButtons_[i], cs);
		}

		JLabel spinnerText = new JLabel("       - Number of channels:");
		cs.weightx = 1;
		cs.weighty = 0;
		cs.gridx = 0;
		cs.gridy = ++currentRow;  
		cs.insets = new Insets(3,3,0,0);
		cs.gridwidth = 1;
		//natural height, natural width
		cs.fill = GridBagConstraints.NONE;
		add(spinnerText, cs);


		nChannelsSpinner_ = new SpinnerNumberModel(1, 1, 10, 1);
		JSpinner spinner = new JSpinner(nChannelsSpinner_);
		cs.gridx = 1;
		add(spinner, cs);
		int nChannels = (int)Prefs.get("hdf5readervibez.nchannels", 1);
		spinner.setValue(nChannels);


		JButton b1 = new JButton("Load");
		b1.setActionCommand("load");
		b1.addActionListener(this);
		cs.ipady = 0;     
		cs.weightx = 1;
		cs.weighty = 0;
		cs.gridx = 0;
		cs.gridy = ++currentRow;   
		cs.gridwidth = 1;
		//natural height, natural width
		cs.fill = GridBagConstraints.NONE;
		add(b1,cs);

		JButton b2 = new JButton("Cancel");
		b2.setActionCommand("cancel");
		b2.addActionListener(this);
		cs.gridx = 1;
		cs.gridy = currentRow;   
		cs.gridwidth = 1;
		add(b2,cs);

		pack();

		Dimension si = getSize();
		si.height = 400;
		si.width = pathTable_.getWidth()+40;
		setSize( si);
		setVisible(true);


	}



	//  private String dsInfoToTypeString( HDF5DataSetInformation dsInfo) {
	//    HDF5DataTypeInformation dsType = dsInfo.getTypeInformation();
	//    String typeText = "";
	//    
	//    if (dsType.isSigned() == false) {
	//      typeText += "u";
	//    }
	//    
	//    switch( dsType.getDataClass())
	//    {
	//      case INTEGER:
	//        typeText += "int" + 8*dsType.getElementSize();
	//        break;
	//      case FLOAT:
	//        typeText += "float" + 8*dsType.getElementSize();
	//        break;
	//      default:
	//        typeText += dsInfo.toString();
	//    }
	//    return typeText;
	//  }

	private void recursiveGetInfo(IHDF5Reader reader, HDF5LinkInformation link)
	{
		List<HDF5LinkInformation> members = reader.object().getGroupMemberInformation(link.getPath(), true);
		//    DefaultMutableTreeNode node = new DefaultMutableTreeNode(link.getName());

		for (HDF5LinkInformation info : members)
		{
			IJ.log(info.getPath() + ":" + info.getType());
			switch (info.getType())
			{
			case DATASET:
				HDF5DataSetInformation dsInfo = reader.object().getDataSetInformation(info.getPath());
				HDF5DataTypeInformation dsType = dsInfo.getTypeInformation();

				String dimText = "";
				if( dsInfo.getRank() == 0) 
				{
					dimText ="1";
				}
				else
				{
					dimText += dsInfo.getDimensions()[0];
					for( int i = 1; i < dsInfo.getRank(); ++i)
					{
						dimText += "x" + dsInfo.getDimensions()[i];
					}
				}


				String typeText = HDF5ImageJ.dsInfoToTypeString(dsInfo);

				// try to read element_size_um attribute
				String element_size_um_text = "unknown";
				try {
					float[] element_size_um = reader.float32().getArrayAttr(info.getPath(), "element_size_um");
					element_size_um_text = "" + element_size_um[0] + "x" 
							+ element_size_um[1] + "x" + element_size_um[2];

				}     
				catch (HDF5Exception err) {
					IJ.log("Warning: Can't read attribute 'element_size_um' from dataset '" + info.getPath() + "':\n"
							+ err );
				} 

				IJ.log(info.getPath() + ":" + dsInfo);

				dataSets_.add( new DataSetInfo( info.getPath(), dimText, typeText, 
						element_size_um_text));


				break;
			case SOFT_LINK:
				IJ.log(info.getPath() + "     -> " + info.tryGetSymbolicLinkTarget());
				//      node.add(new DefaultMutableTreeNode(info.getName() + "     -> " + info.tryGetSymbolicLinkTarget()));

				break;
			case GROUP:
				recursiveGetInfo( reader, info);
				//        node.add( browse(reader,info));

				break;
			default:
				break;
			}
		}

	}



	public void actionPerformed(ActionEvent event) 
	{
		if (event.getActionCommand().equals("load")) 
		{
			loadHDF5();
		}
		else if (event.getActionCommand().equals("cancel")) 
		{
			dispose();
		}
	}


	public void loadHDF5() {
		int[] selection = pathTable_.getSelectedRows();
		if (selection.length == 0) {
			IJ.error( "load HDF5", "You must select at least one data set");
			return;
		}

		int loadAsMode = 0;
		for( int i = 0; i < loadAsRadioButtons_.length; ++i)
		{
			if( loadAsRadioButtons_[i].isSelected()) loadAsMode = i;
		}
		Prefs.set("hdf5readervibez.loadasmode", loadAsMode);

		if (loadAsMode == 0) 
		{
			// load as multiple standard stacks

			for (int i : selection) {
				IJ.log( "i = " + i + dataSets_.get(i).path);
				String[] dsetNames = new String[1];
				dsetNames[0] = dataSets_.get(i).path;
				String type = dataSets_.get(i).typeText;
				HDF5ImageJ.loadDataSetsToHyperStack( fullFileName_, dsetNames, 1, 1);
			}
		}
		else if  (loadAsMode == 1) 
		{
			// load as multiple hyper stacks with custom layout

			for (int i : selection) {
				IJ.log( "i = " + i + dataSets_.get(i).path);
				String dsetLayout =  dsetLayoutTextField_.getText();
				Prefs.set("hdf5readervibez.dsetLayout", dsetLayout);

				HDF5ImageJ.loadCustomLayoutDataSetToHyperStack( fullFileName_, dataSets_.get(i).path, 
						dsetLayout);
			}

		}
		else
		{
			// load as Hyperstack
			String[] dsetNames = new String[selection.length];
			for( int i = 0; i < selection.length; ++i) {
				dsetNames[i] = dataSets_.get(selection[i]).path;
			}
			int nChannels = 1;
			if( loadAsMode == 2) nChannels = selection.length;
			if( loadAsMode == 3) nChannels = 1;
			if( loadAsMode == 4) 
			{
				nChannels = nChannelsSpinner_.getNumber().intValue();
			}            
			if (nChannels > dsetNames.length) {
				nChannels = dsetNames.length;
			}
			Prefs.set("hdf5readervibez.nchannels",nChannels);
			int nFrames = dsetNames.length/nChannels;
			Prefs.set("hdf5readervibez.nframes",nFrames);
			String commaSeparatedDsetNames = "";
			for( int i=0; i < dsetNames.length; ++i)
			{
				if( i > 0) commaSeparatedDsetNames += ",";
				commaSeparatedDsetNames += dsetNames[i];
			}
			Prefs.set("hdf5readervibez.dsetnames",commaSeparatedDsetNames);

			String type = dataSets_.get(selection[0]).typeText;
			HDF5ImageJ.loadDataSetsToHyperStack( fullFileName_, dsetNames, 
					nFrames, nChannels);

		} 
		dispose();
	}



	//
	//  int assignHDF5TypeToImagePlusBitdepth( String type, int rank) {
	//    int nBits = 0;
	//    if (type.equals("uint8")) {
	//      if( rank == 4) {
	//        nBits = 24;
	//      } else {
	//        nBits = 8;
	//      }
	//    } else if (type.equals("uint16") || type.equals("int16")) {
	//      nBits = 16;
	//    } else if (type.equals("float32") || type.equals("float64")) {
	//      nBits = 32;
	//    } else {
	//      IJ.error("Type '" + type + "' Not handled yet!");
	//    }
	//    return nBits;
	//  }
	//
	//
	//  ImagePlus loadDataSetToHyperStack( String filename, String[] dsetNames, int nFrames, int nChannels) {
	//    String dsetName = "";
	//    try
	//    {
	//      IHDF5ReaderConfigurator conf = HDF5Factory.configureForReading(filename);
	//      conf.performNumericConversions();
	//      IHDF5Reader reader = conf.reader();
	//      ImagePlus imp = null;
	//      int rank    = 0;
	//      int nLevels = 0;
	//      int nRows   = 0;
	//      int nCols   = 0;
	//      int nBits   = 0;
	//      double maxGray = 1;
	//      String typeText = "";
	//      for (int frame = 0; frame < nFrames; ++frame) {
	//        for (int channel = 0; channel < nChannels; ++channel) {
	//          // load data set
	//          //
	//          dsetName = dsetNames[frame*nChannels+channel];
	//          IJ.showStatus( "Loading " + dsetName);
	//          IJ.showProgress( frame*nChannels+channel+1, nFrames*nChannels);
	//          HDF5DataSetInformation dsInfo = reader.object().getDataSetInformation(dsetName);
	//          float[] element_size_um = {1,1,1};
	//          try {
	//            element_size_um = reader.float32().getArrayAttr(dsetName, "element_size_um");
	//          }     
	//          catch (HDF5Exception err) {
	//            IJ.log("Warning: Can't read attribute 'element_size_um' from file '" + filename 
	//                     + "', dataset '" + dsetName + "':\n"
	//                     + err + "\n" 
	//                     + "Assuming element size of 1 x 1 x 1 um^3");
	//          } 
	//
	//          // in first call create hyperstack
	//          //
	//          if (imp == null) {
	//            rank = dsInfo.getRank();
	//            typeText = dsInfoToTypeString(dsInfo);
	//            if (rank == 2) {
	//              nLevels = 1;
	//              nRows = (int)dsInfo.getDimensions()[0];
	//              nCols = (int)dsInfo.getDimensions()[1];
	//            } else if (rank == 3) {
	//              nLevels = (int)dsInfo.getDimensions()[0];
	//              nRows   = (int)dsInfo.getDimensions()[1];
	//              nCols   = (int)dsInfo.getDimensions()[2];
	//            } else if (rank == 4 && typeText.equals( "uint8")) {
	//              nLevels = (int)dsInfo.getDimensions()[0];
	//              nRows   = (int)dsInfo.getDimensions()[1];
	//              nCols   = (int)dsInfo.getDimensions()[2];
	//            } else {
	//              IJ.error( dsetName + ": rank " + rank + " of type " + typeText + " not supported (yet)");
	//              return null;
	//            }
	//            nBits = assignHDF5TypeToImagePlusBitdepth( typeText, rank);
	//
	//            
	//            imp = IJ.createHyperStack( filename + ": " + dsetName, 
	//                                       nCols, nRows, nChannels, nLevels, nFrames, nBits);
	//            imp.getCalibration().pixelDepth  = element_size_um[0];
	//            imp.getCalibration().pixelHeight = element_size_um[1];
	//            imp.getCalibration().pixelWidth  = element_size_um[2];
	//            imp.getCalibration().setUnit("micrometer");
	//            imp.setDisplayRange(0,255);
	//          }
	//          
	//          // copy slices to hyperstack
	//          int sliceSize = nCols * nRows;
	//          
	//          if (typeText.equals( "uint8") && rank < 4) {
	//            MDByteArray rawdata = reader.uint8().readMDArray(dsetName);
	//            for( int lev = 0; lev < nLevels; ++lev) {
	//              ImageProcessor ip = imp.getStack().getProcessor( imp.getStackIndex(
	//                  channel+1, lev+1, frame+1));
	//              System.arraycopy( rawdata.getAsFlatArray(), lev*sliceSize, 
	//                                (byte[])ip.getPixels(),   0, 
	//                                sliceSize);
	//            }            
	//            maxGray = 255;
	//          }  else if (typeText.equals( "uint8") && rank == 4) {  // RGB data
	//            MDByteArray rawdata = reader.uint8().readMDArray(dsetName);
	//            byte[] srcArray = rawdata.getAsFlatArray();
	//            
	//
	//            for( int lev = 0; lev < nLevels; ++lev) {
	//              ImageProcessor ip = imp.getStack().getProcessor( imp.getStackIndex(
	//                  channel+1, lev+1, frame+1));
	//              int[] trgArray = (int[])ip.getPixels();
	//              int srcOffset = lev*sliceSize*3;
	//              
	//              for( int rc = 0; rc < sliceSize; ++rc)
	//              {
	//                int red   = srcArray[srcOffset + rc*3];
	//                int green = srcArray[srcOffset + rc*3 + 1];
	//                int blue  = srcArray[srcOffset + rc*3 + 2];
	//                trgArray[rc] = (red<<16) + (green<<8) + blue;
	//              }
	//              
	//            }            
	//            maxGray = 255;
	//
	//          } else if (typeText.equals( "uint16")) {
	//            MDShortArray rawdata = reader.uint16().readMDArray(dsetName);
	//            for( int lev = 0; lev < nLevels; ++lev) {
	//              ImageProcessor ip = imp.getStack().getProcessor( imp.getStackIndex(
	//                  channel+1, lev+1, frame+1));
	//              System.arraycopy( rawdata.getAsFlatArray(), lev*sliceSize, 
	//                                (short[])ip.getPixels(),   0, 
	//                                sliceSize);
	//            }
	//            short[] data = rawdata.getAsFlatArray();
	//            for (int i = 0; i < data.length; ++i) {
	//              if (data[i] > maxGray) maxGray = data[i];
	//            }
	//          } else if (typeText.equals( "int16")) {
	//            MDShortArray rawdata = reader.int16().readMDArray(dsetName);
	//            for( int lev = 0; lev < nLevels; ++lev) {
	//              ImageProcessor ip = imp.getStack().getProcessor( imp.getStackIndex(
	//                  channel+1, lev+1, frame+1));
	//              System.arraycopy( rawdata.getAsFlatArray(), lev*sliceSize, 
	//                                (short[])ip.getPixels(),   0, 
	//                                sliceSize);
	//            }
	//            short[] data = rawdata.getAsFlatArray();
	//            for (int i = 0; i < data.length; ++i) {
	//              if (data[i] > maxGray) maxGray = data[i];
	//            }
	//          } else if (typeText.equals( "float32") || typeText.equals( "float64") ) {
	//            MDFloatArray rawdata = reader.float32().readMDArray(dsetName);
	//            for( int lev = 0; lev < nLevels; ++lev) {
	//              ImageProcessor ip = imp.getStack().getProcessor( imp.getStackIndex(
	//                  channel+1, lev+1, frame+1));
	//              System.arraycopy( rawdata.getAsFlatArray(), lev*sliceSize, 
	//                                (float[])ip.getPixels(),   0, 
	//                                sliceSize);
	//            }
	//            float[] data = rawdata.getAsFlatArray();
	//            for (int i = 0; i < data.length; ++i) {
	//              if (data[i] > maxGray) maxGray = data[i];
	//            }
	//          }
	//        }
	//      }                  
	//      reader.close();
	//      
	//      // aqdjust max gray
	//      for( int c = 1; c <= nChannels; ++c)
	//      {
	//        imp.setC(c);
	//        imp.setDisplayRange(0,maxGray);
	//      }
	//      
	//      imp.setC(1);
	//      imp.show();
	//      return imp;
	//    }
	//    
	//    catch (HDF5Exception err) 
	//    {
	//      IJ.error("Error while opening '" + filename 
	//               + "', dataset '" + dsetName + "':\n"
	//               + err);
	//    } 
	//    catch (Exception err) 
	//    {
	//      IJ.error("Error while opening '" + filename 
	//               + "', dataset '" + dsetName + "':\n"
	//               + err);
	//    } 
	//    catch (OutOfMemoryError o) 
	//    {
	//      IJ.outOfMemory("Load HDF5");
	//    }
	//    return null;
	//    
	//  }
	//  
	//







	//
	//
	//
	//
	//  ImagePlus loadDataSetToImagePlus_Byte( String filename, String dsetName)
	//        {
	//          try
	//          {
	//            IHDF5ReaderConfigurator conf = HDF5Factory.configureForReading(filename);
	//            conf.performNumericConversions();
	//            IHDF5Reader reader = conf.reader();
	//            MDByteArray rawdata = reader.uint8().readMDArray(dsetName);
	//            float[] element_size_um = reader.float32().getArrayAttr(dsetName, "element_size_um");
	//        
	//            reader.close();
	//          
	//            System.out.println( "dimensions: " 
	//                                + rawdata.dimensions()[0] + 
	//                                "," + rawdata.dimensions()[1] +
	//                                "," + rawdata.dimensions()[2]);
	//          
	//            // create a new image stack and fill in the data
	//            int nLevels = rawdata.dimensions()[0];
	//            int nRows = rawdata.dimensions()[1];
	//            int nCols = rawdata.dimensions()[2];
	//          
	//            ImageStack stack = new ImageStack(nCols, nRows, nLevels);
	//            long stackSize = nCols * nRows;
	//            byte[] flatArray = rawdata.getAsFlatArray();
	//            for( int lev = 0; lev < nLevels; ++lev)
	//            {
	//              byte[] slice = new byte[nRows*nCols];
	//            
	//              System.arraycopy( flatArray, lev*nRows*nCols, 
	//                                slice, 0, 
	//                                nRows*nCols);
	//              stack.setPixels(slice, lev+1);
	//            }
	//            ImagePlus imp = new ImagePlus( filename + ": " + dsetName, stack);
	//            imp.getCalibration().pixelDepth  = element_size_um[0];
	//            imp.getCalibration().pixelHeight = element_size_um[1];
	//            imp.getCalibration().pixelWidth  = element_size_um[2];
	//            imp.getCalibration().setUnit("micrometer");
	//            imp.setDisplayRange(0,255);
	//        
	//          
	//            imp.show();
	//            return imp;
	//          }
	//       
	//          catch (HDF5Exception err) 
	//          {
	//            IJ.error("Error while opening '" + filename 
	//                     + "', dataset '" + dsetName + "':\n"
	//                     + err);
	//          } 
	//          catch (Exception err) 
	//          {
	//            IJ.error("Error while opening '" + filename 
	//                     + "', dataset '" + dsetName + "':\n"
	//                     + err);
	//          } 
	//          catch (OutOfMemoryError o) 
	//          {
	//            IJ.outOfMemory("Load HDF5");
	//          }
	//          return null;
	//          
	//        }
	//
	//
	//  ImagePlus loadDataSetToHyperStack_Byte( String filename, String[] dsetNames, int nFrames, int nChannels)
	//        {
	//          String dsetName = "";
	//          try
	//          {
	//            IHDF5ReaderConfigurator conf = HDF5Factory.configureForReading(filename);
	//            conf.performNumericConversions();
	//            IHDF5Reader reader = conf.reader();
	//            ImagePlus imp = null;
	//            int nLevels = 0;
	//            int nRows   = 0;
	//            int nCols   = 0;
	//            for (int frame = 0; frame < nFrames; ++frame) {
	//              for (int channel = 0; channel < nChannels; ++channel) {
	//                // load data set
	//                //
	//                dsetName = dsetNames[frame*nChannels+channel];
	//                IJ.showStatus( "Loading " + dsetName);
	//                IJ.showProgress( frame*nChannels+channel+1, nFrames*nChannels);
	//                MDByteArray rawdata = reader.uint8().readMDArray(dsetName);
	//                float[] element_size_um = reader.float32().getArrayAttr(dsetName, "element_size_um");
	//
	//                // in first call create hyperstack
	//                //
	//                if( imp == null) {
	//                  System.out.println( "dimensions: " 
	//                                      + rawdata.dimensions()[0] + 
	//                                      "," + rawdata.dimensions()[1] +
	//                                      "," + rawdata.dimensions()[2]);
	//                  nLevels = rawdata.dimensions()[0];
	//                  nRows = rawdata.dimensions()[1];
	//                  nCols = rawdata.dimensions()[2];
	//                  imp = IJ.createHyperStack( filename + ": " + dsetName, 
	//                                             nCols,
	//                                             nRows, 
	//                                             nChannels, 
	//                                             nLevels, 
	//                                             nFrames,
	//                                             8);
	//                  imp.getCalibration().pixelDepth  = element_size_um[0];
	//                  imp.getCalibration().pixelHeight = element_size_um[1];
	//                  imp.getCalibration().pixelWidth  = element_size_um[2];
	//                  imp.getCalibration().setUnit("micrometer");
	//                  imp.setDisplayRange(0,255);
	//                }
	//                
	//                // copy slices to hyperstack
	//                byte[] flatArray = rawdata.getAsFlatArray();
	//                int sliceSize = nCols * nRows;
	//                for( int lev = 0; lev < nLevels; ++lev)
	//                {
	//                  ImageProcessor ip = imp.getStack().getProcessor( imp.getStackIndex(channel+1, lev+1, frame+1));
	//                  byte[] slice = (byte[])ip.getPixels();
	//                  System.arraycopy( flatArray, lev*sliceSize, 
	//                                    slice, 0, 
	//                                    sliceSize);
	//                }
	//              }
	//            }                  
	//            reader.close();
	//                      
	//            imp.show();
	//            return imp;
	//          }
	//       
	//          catch (HDF5Exception err) 
	//          {
	//            IJ.error("Error while opening '" + filename 
	//                     + "', dataset '" + dsetName + "':\n"
	//                     + err);
	//          } 
	//          catch (Exception err) 
	//          {
	//            IJ.error("Error while opening '" + filename 
	//                     + "', dataset '" + dsetName + "':\n"
	//                     + err);
	//          } 
	//          catch (OutOfMemoryError o) 
	//          {
	//            IJ.outOfMemory("Load HDF5");
	//          }
	//          return null;
	//          
	//        }
	//
	//
	//
	//  ImagePlus loadDataSetToImagePlus_Short( String filename, String dsetName)
	//        {
	//          try
	//          {
	//            IHDF5ReaderConfigurator conf = HDF5Factory.configureForReading(filename);
	//            conf.performNumericConversions();
	//            IHDF5Reader reader = conf.reader();
	//            MDShortArray rawdata = reader.int16().readMDArray(dsetName);
	//            float[] element_size_um = reader.float32().getArrayAttr(dsetName, "element_size_um");
	//        
	//            reader.close();
	//          
	//            System.out.println( "dimensions: " 
	//                                + rawdata.dimensions()[0] + 
	//                                "," + rawdata.dimensions()[1] +
	//                                "," + rawdata.dimensions()[2]);
	//          
	//            // create a new image stack and fill in the data
	//            int nLevels = rawdata.dimensions()[0];
	//            int nRows = rawdata.dimensions()[1];
	//            int nCols = rawdata.dimensions()[2];
	//          
	//            ImageStack stack = new ImageStack(nCols, nRows, nLevels);
	//            long stackSize = nCols * nRows;
	//            short[] flatArray = rawdata.getAsFlatArray();
	//            for( int lev = 0; lev < nLevels; ++lev)
	//            {
	//              short[] slice = new short[nRows*nCols];
	//            
	//              System.arraycopy( flatArray, lev*nRows*nCols, 
	//                                slice, 0, 
	//                                nRows*nCols);
	//              stack.setPixels(slice, lev+1);
	//            }
	//            ImagePlus imp = new ImagePlus( filename + ": " + dsetName, stack);
	//            imp.getCalibration().pixelDepth  = element_size_um[0];
	//            imp.getCalibration().pixelHeight = element_size_um[1];
	//            imp.getCalibration().pixelWidth  = element_size_um[2];
	//            imp.getCalibration().setUnit("micrometer");
	//
	//            short maxGray = 255;
	//            for (int i = 0; i < flatArray.length; ++i) {
	//              if( flatArray[i] > maxGray) maxGray = flatArray[i];
	//            }
	//            imp.setDisplayRange(0,maxGray);
	//        
	//          
	//            imp.show();
	//            return imp;
	//          }
	//       
	//          catch (HDF5Exception err) 
	//          {
	//            IJ.error("Error while opening '" + filename 
	//                     + "', dataset '" + dsetName + "':\n"
	//                     + err);
	//          } 
	//          catch (Exception err) 
	//          {
	//            IJ.error("Error while opening '" + filename 
	//                     + "', dataset '" + dsetName + "':\n"
	//                     + err);
	//          } 
	//          catch (OutOfMemoryError o) 
	//          {
	//            IJ.outOfMemory("Load HDF5");
	//          }
	//          return null;
	//          
	//        }
	//
	//    ImagePlus loadDataSetToHyperStack_Short( String filename, String[] dsetNames, int nFrames, int nChannels)
	//        {
	//          String dsetName = "";
	//          try
	//          {
	//            IHDF5ReaderConfigurator conf = HDF5Factory.configureForReading(filename);
	//            conf.performNumericConversions();
	//            IHDF5Reader reader = conf.reader();
	//            ImagePlus imp = null;
	//            int nLevels = 0;
	//            int nRows   = 0;
	//            int nCols   = 0;
	//            for (int frame = 0; frame < nFrames; ++frame) {
	//              for (int channel = 0; channel < nChannels; ++channel) {
	//                // load data set
	//                //
	//                dsetName = dsetNames[frame*nChannels+channel];
	//                IJ.showStatus( "Loading " + dsetName);
	//                IJ.showProgress( frame*nChannels+channel+1, nFrames*nChannels);
	//                MDShortArray rawdata = reader.int16().readMDArray(dsetName);
	//                float[] element_size_um = reader.float32().getArrayAttr(dsetName, "element_size_um");
	//
	//                // in first call create hyperstack
	//                //
	//                if( imp == null) {
	//                  System.out.println( "dimensions: " 
	//                                      + rawdata.dimensions()[0] + 
	//                                      "," + rawdata.dimensions()[1] +
	//                                      "," + rawdata.dimensions()[2]);
	//                  nLevels = rawdata.dimensions()[0];
	//                  nRows = rawdata.dimensions()[1];
	//                  nCols = rawdata.dimensions()[2];
	//                  imp = IJ.createHyperStack( filename + ": " + dsetName, 
	//                                             nCols,
	//                                             nRows, 
	//                                             nChannels, 
	//                                             nLevels, 
	//                                             nFrames,
	//                                             16);
	//                  imp.getCalibration().pixelDepth  = element_size_um[0];
	//                  imp.getCalibration().pixelHeight = element_size_um[1];
	//                  imp.getCalibration().pixelWidth  = element_size_um[2];
	//                  imp.getCalibration().setUnit("micrometer");
	//                  imp.setDisplayRange(0,4095);
	//                }
	//                
	//                // copy slices to hyperstack
	//                short[] flatArray = rawdata.getAsFlatArray();
	//                int sliceSize = nCols * nRows;
	//                for( int lev = 0; lev < nLevels; ++lev)
	//                {
	//                  ImageProcessor ip = imp.getStack().getProcessor( imp.getStackIndex(channel+1, lev+1, frame+1));
	//                  short[] slice = (short[])ip.getPixels();
	//                  System.arraycopy( flatArray, lev*sliceSize, 
	//                                    slice, 0, 
	//                                    sliceSize);
	//                }
	//              }
	//            }                  
	//            reader.close();
	//                      
	//            imp.show();
	//            return imp;
	//          }
	//       
	//          catch (HDF5Exception err) 
	//          {
	//            IJ.error("Error while opening '" + filename 
	//                     + "', dataset '" + dsetName + "':\n"
	//                     + err);
	//          } 
	//          catch (Exception err) 
	//          {
	//            IJ.error("Error while opening '" + filename 
	//                     + "', dataset '" + dsetName + "':\n"
	//                     + err);
	//          } 
	//          catch (OutOfMemoryError o) 
	//          {
	//            IJ.outOfMemory("Load HDF5");
	//          }
	//          return null;
	//          
	//        }
	//
	//ImagePlus loadDataSetToImagePlus_Float( String filename, String dsetName)
	//        {
	//          try
	//          {
	//            IHDF5ReaderConfigurator conf = HDF5Factory.configureForReading(filename);
	//            conf.performNumericConversions();
	//            IHDF5Reader reader = conf.reader();
	//            MDFloatArray rawdata = reader.float32().readMDArray(dsetName);
	//            float[] element_size_um = reader.float32().getArrayAttr(dsetName, "element_size_um");
	//        
	//            reader.close();
	//          
	//            System.out.println( "dimensions: " 
	//                                + rawdata.dimensions()[0] + 
	//                                "," + rawdata.dimensions()[1] +
	//                                "," + rawdata.dimensions()[2]);
	//          
	//            // create a new image stack and fill in the data
	//            int nLevels = rawdata.dimensions()[0];
	//            int nRows = rawdata.dimensions()[1];
	//            int nCols = rawdata.dimensions()[2];
	//          
	//            ImageStack stack = new ImageStack(nCols, nRows, nLevels);
	//            long stackSize = nCols * nRows;
	//            float[] flatArray = rawdata.getAsFlatArray();
	//            for( int lev = 0; lev < nLevels; ++lev)
	//            {
	//              float[] slice = new float[nRows*nCols];
	//            
	//              System.arraycopy( flatArray, lev*nRows*nCols, 
	//                                slice, 0, 
	//                                nRows*nCols);
	//              stack.setPixels(slice, lev+1);
	//            }
	//            ImagePlus imp = new ImagePlus( filename + ": " + dsetName, stack);
	//            imp.getCalibration().pixelDepth  = element_size_um[0];
	//            imp.getCalibration().pixelHeight = element_size_um[1];
	//            imp.getCalibration().pixelWidth  = element_size_um[2];
	//            imp.getCalibration().setUnit("micrometer");
	//
	//            float maxGray = 1;
	//            for (int i = 0; i < flatArray.length; ++i) {
	//              if( flatArray[i] > maxGray) maxGray = flatArray[i];
	//            }
	//            imp.setDisplayRange(0,maxGray);
	//        
	//          
	//            imp.show();
	//            return imp;
	//          }
	//       
	//          catch (HDF5Exception err) 
	//          {
	//            IJ.error("Error while opening '" + filename 
	//                     + "', dataset '" + dsetName + "':\n"
	//                     + err);
	//          } 
	//          catch (Exception err) 
	//          {
	//            IJ.error("Error while opening '" + filename 
	//                     + "', dataset '" + dsetName + "':\n"
	//                     + err);
	//          } 
	//          catch (OutOfMemoryError o) 
	//          {
	//            IJ.outOfMemory("Load HDF5");
	//          }
	//          return null;
	//          
	//        }
	//
	//

}
