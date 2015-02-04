package edu.stanford.rsl.apps.gui;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.io.FileInfo;
import ij.io.FileOpener;

import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.Transferable;
import java.awt.datatransfer.UnsupportedFlavorException;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JTextField;
import javax.swing.TransferHandler;
import javax.swing.UIManager;

import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.utils.Configuration;

/**
 * User interface to read raw data. Allows also the reading of DICOM data in a zip file.
 * @author akmaier
 *
 */
public class RawDataOpener extends JFrame implements ActionListener {
	/**
	 * 
	 */
	private static final long serialVersionUID = -1952776160986620856L;
	private JButton jButtonLittle;
	private JButton jButtonBig;
	private JLabel jLabel2;
	private JLabel jLabel3;
	private JLabel jLabel4;
	private JTextField jOffset;
	private JTextField jTextFieldStack;
	private JButton jButtonDouble;
	private JTextField jTextFieldHeight;
	private JTextField jTextFieldWidth;
	private JLabel jLabel1;
	private JButton jButtonSShort;
	private JButton jButtonShort;
	private JButton jButtonFloat;

	private class OpenerTransferHandler extends TransferHandler {
		/**
		 * 
		 */
		private static final long serialVersionUID = 2183195105160823169L;
		RawDataOpener field;

		public OpenerTransferHandler(RawDataOpener field){
			super();
			this.field = field;
		}

		public boolean canImport(TransferHandler.TransferSupport support) {
			if (!support.isDataFlavorSupported(DataFlavor.javaFileListFlavor)) {
				return false;
			}
			boolean copySupported = (COPY & support.getSourceDropActions()) == COPY;
			if (!copySupported) {
				return false;
			}
			support.setDropAction(COPY);
			return true;
		}
		
		@SuppressWarnings("unchecked")
		public boolean importData(TransferHandler.TransferSupport support) {
			if (!canImport(support)) {
				return false;
			}
			Transferable t = support.getTransferable();
			try {
				java.util.List<File> l =
					(java.util.List<File>) t.getTransferData(DataFlavor.javaFileListFlavor);
				FileInfo fi = field.getFileInfo();
				if (l.size() > 1) {
					File [] filenames = new File [l.size()];
					for (int i = 0; i < l.size(); i++){
						filenames[i] = l.get(i);
					}	
					openFileList(filenames, fi).show();
				} else {
					File file = l.get(0);
					if (file.isDirectory()){
						//System.out.println(file.listFiles().length);
						openFileList(file.listFiles(), fi).show();
					} else {
						if (file.getName().endsWith(".zip")) {
							// ImageJ does not allow to override the reading of ZIP files.
							// Hence, we do this here.
							String className ="ZIP_Reader";
							String path = file.getAbsolutePath();
							Object o = IJ.runPlugIn(className, path);
							if (o instanceof ImagePlus) {
								// plugin extends ImagePlus class
								ImagePlus imp = (ImagePlus)o;
								imp.show();
							}
						} else {
							openImage(file, fi).show();
						}
					}
				}
			} catch (UnsupportedFlavorException e) {
				return false;
			} catch (IOException e) {
				return false;
			}
			return true;
		}


	}
	
	private ImagePlus openFileList(File [] files, FileInfo fi){
		ImagePlus image = null;
		if (fi.nImages > 1){
			ImageStack stack = null;
			for (File file: files){
				ImagePlus img = openImage(file, fi);
				if (stack == null) stack = new ImageStack(img.getWidth(), img.getHeight());
				for (int i =1; i <= img.getStackSize(); i++){
					stack.addSlice("slice " + i, img.getStack().getProcessor(i));
				}
			}
			image = new ImagePlus();
			image.setStack(files[0].getName(), stack);
			image.setOpenAsHyperStack(true);
			image.setDimensions(1, fi.nImages, files.length);
		} else {
			image = openImageSequence(files, fi);
		}
		return image;
	}

	public static RawDataOpener opener = null;

	public static RawDataOpener getRawDataOpener(){
		if (opener == null){
			new RawDataOpener();
		}
		return opener;
	}

	public ImagePlus openImage(File file, FileInfo fi){
		fi.directory = file.getParent();
		fi.fileName = file.getName();
		FileOpener open = new FileOpener(fi);
		return open.open(false);
	}

	public ImagePlus openImageSequence(File [] filenames, FileInfo fi){
		//Arrays.sort(filenames);
		ImageStack stack = new ImageStack(fi.width, fi.height);
		for (int i = 0; i < filenames.length; i++){
			File file = filenames[i];
			stack.addSlice("Image " + i, openImage(file, fi).getChannelProcessor());
		}
		ImagePlus stackImage = new ImagePlus();
		stackImage.setStack(filenames[0].getName(), stack);
		return stackImage;
	}

	public RawDataOpener(){
		super("Raw Data Opener");
		initGUI();
		opener = this;
	}

	private void initGUI() {
		try {
			{
				TransferHandler th = new OpenerTransferHandler(this);
				GridBagLayout thisLayout = new GridBagLayout();
				thisLayout.rowWeights = new double[] {0.1, 0.1, 0.1, 0.1, 0.0};
				thisLayout.rowHeights = new int[] {20, 20, 20, 20, 7};
				thisLayout.columnWeights = new double[] {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
				thisLayout.columnWidths = new int[] {20, 20, 20, 20, 20, 20, 20, 20};
				getContentPane().setLayout(thisLayout);
				UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
				{
					jButtonLittle = new JButton();
					getContentPane().add(jButtonLittle, new GridBagConstraints(4, 2, 4, 2, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.BOTH, new Insets(0, 0, 0, 0), 0, 0));
					jButtonLittle.setText("little Endian");
					jButtonLittle.addActionListener(this);
					jButtonLittle.setTransferHandler(th);
				}
				{
					jButtonBig = new JButton();
					getContentPane().add(jButtonBig, new GridBagConstraints(4, 0, 4, 2, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.BOTH, new Insets(0, 0, 0, 0), 0, 0));
					jButtonBig.setText("big Endian");
					jButtonBig.addActionListener(this);
					jButtonBig.setTransferHandler(th);
				}
				{
					jButtonFloat = new JButton();
					getContentPane().add(jButtonFloat, new GridBagConstraints(0, 2, 4, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.BOTH, new Insets(0, 0, 0, 0), 0, 0));
					jButtonFloat.setText("Float 32 Bit");
					jButtonFloat.addActionListener(this);
					jButtonFloat.setTransferHandler(th);
				}
				{
					jButtonDouble = new JButton();
					getContentPane().add(jButtonDouble, new GridBagConstraints(2, 1, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.BOTH, new Insets(0, 0, 0, 0), 0, 0));
					jButtonDouble.setText("Float 64 Bit");
					jButtonDouble.addActionListener(this);
					jButtonDouble.setTransferHandler(th);
				}
				{
					jButtonShort = new JButton();
					getContentPane().add(jButtonShort, new GridBagConstraints(0, 1, 4, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.BOTH, new Insets(0, 0, 0, 0), 0, 0));
					jButtonShort.setText("Unsigned Short");
					jButtonShort.addActionListener(this);
					jButtonShort.setTransferHandler(th);
				}
				{
					jLabel1 = new JLabel("Width:");
					jLabel2 = new JLabel("Height:");
					Trajectory geometry = Configuration.getGlobalConfiguration().getGeometry();
					if (geometry != null) {
						jTextFieldWidth = new JTextField("" + geometry.getDetectorWidth());
						jTextFieldHeight = new JTextField("" + geometry.getDetectorHeight());
					} else {
						jTextFieldWidth = new JTextField("620");
						jTextFieldHeight = new JTextField("480");	
					}
					jTextFieldWidth.setHorizontalAlignment(JTextField.RIGHT);
					jTextFieldHeight.setHorizontalAlignment(JTextField.RIGHT);
					jButtonSShort = new JButton();
					jButtonSShort.setText("Signed Short");
					jButtonSShort.addActionListener(this);
					jButtonSShort.setTransferHandler(th);
					jTextFieldStack = new JTextField();
					jTextFieldStack.setText("1");
					jTextFieldStack.setHorizontalAlignment(JTextField.RIGHT);
					getJOffset().setHorizontalAlignment(JTextField.RIGHT);
					jLabel3 = new JLabel();
					jLabel3.setText("Stack Size:");
					getContentPane().add(jButtonSShort, new GridBagConstraints(0, 0, 4, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.BOTH, new Insets(0, 0, 0, 0), 0, 0));
					getContentPane().add(jLabel1, new GridBagConstraints(0, 4, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
					getContentPane().add(jLabel2, new GridBagConstraints(2, 4, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
					getContentPane().add(jTextFieldWidth, new GridBagConstraints(1, 4, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.HORIZONTAL, new Insets(0, 0, 0, 0), 0, 0));
					getContentPane().add(jTextFieldHeight, new GridBagConstraints(3, 4, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.HORIZONTAL, new Insets(0, 0, 0, 0), 0, 0));
					getContentPane().add(jButtonDouble, new GridBagConstraints(0, 3, 4, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.BOTH, new Insets(0, 0, 0, 0), 0, 0));
					getContentPane().add(getJOffset(), new GridBagConstraints(7, 4, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.BOTH, new Insets(0, 0, 0, 0), 0, 0));
					getContentPane().add(getJLabel4(), new GridBagConstraints(6, 4, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
					getContentPane().add(jLabel3, new GridBagConstraints(4, 4, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
					getContentPane().add(jTextFieldStack, new GridBagConstraints(5, 4, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.BOTH, new Insets(0, 0, 0, 0), 0, 0));
				}
			}
			pack();
		} catch(Exception e) {
			e.printStackTrace();
		}
	}

	public FileInfo getFileInfo(){
		FileInfo fi = new FileInfo();
		fi.fileFormat = FileInfo.RAW;
		fi.fileType = FileInfo.GRAY8; 
		fi.height = Integer.parseInt(this.jTextFieldHeight.getText());
		fi.width = Integer.parseInt(this.jTextFieldWidth.getText());
		fi.nImages = Integer.parseInt(this.jTextFieldStack.getText());
		fi.offset = Integer.parseInt(getJOffset().getText());
		if (this.jButtonBig.isSelected()){
			fi.intelByteOrder = false;
		}
		if (this.jButtonLittle.isSelected()){
			fi.intelByteOrder = true;
		}
		if (this.jButtonShort.isSelected()){
			fi.fileType=FileInfo.GRAY16_UNSIGNED;
		}
		if (this.jButtonSShort.isSelected()){
			fi.fileType=FileInfo.GRAY16_SIGNED;
		}
		if (this.jButtonFloat.isSelected()){
			fi.fileType=FileInfo.GRAY32_FLOAT;
		}
		if (this.jButtonDouble.isSelected()){
			fi.fileType=FileInfo.GRAY64_FLOAT;
		}		
		return fi;
	}

	public void actionPerformed(ActionEvent e) {
		if (e.getSource() != null){
			Object source = e.getSource();
			if (source.equals(this.jButtonLittle)){
				this.jButtonLittle.setSelected(true);
				this.jButtonBig.setSelected(false);
			}
			if (source.equals(this.jButtonBig)){
				this.jButtonLittle.setSelected(false);
				this.jButtonBig.setSelected(true);

			}
			if (source.equals(this.jButtonShort)){
				this.jButtonShort.setSelected(true);
				this.jButtonSShort.setSelected(false);
				this.jButtonFloat.setSelected(false);
				this.jButtonDouble.setSelected(false);
			}
			if (source.equals(this.jButtonSShort)){
				this.jButtonShort.setSelected(false);
				this.jButtonSShort.setSelected(true);
				this.jButtonFloat.setSelected(false);
				this.jButtonDouble.setSelected(false);				
			}
			if (source.equals(this.jButtonFloat)){
				this.jButtonShort.setSelected(false);
				this.jButtonSShort.setSelected(false);
				this.jButtonFloat.setSelected(true);
				this.jButtonDouble.setSelected(false);
			}
			if (source.equals(this.jButtonDouble)){
				this.jButtonShort.setSelected(false);
				this.jButtonSShort.setSelected(false);
				this.jButtonFloat.setSelected(false);
				this.jButtonDouble.setSelected(true);
			}
		}

	}

	public JButton getjButtonLittle() {
		return jButtonLittle;
	}

	public void setjButtonLittle(JButton jButtonLittle) {
		this.jButtonLittle = jButtonLittle;
	}

	public JButton getjButtonBig() {
		return jButtonBig;
	}

	public void setjButtonBig(JButton jButtonBig) {
		this.jButtonBig = jButtonBig;
	}

	public JButton getjButtonDouble() {
		return jButtonDouble;
	}

	public void setjButtonDouble(JButton jButtonDouble) {
		this.jButtonDouble = jButtonDouble;
	}

	public JButton getjButtonSShort() {
		return jButtonSShort;
	}

	public void setjButtonSShort(JButton jButtonSShort) {
		this.jButtonSShort = jButtonSShort;
	}

	public JButton getjButtonShort() {
		return jButtonShort;
	}

	public void setjButtonShort(JButton jButtonShort) {
		this.jButtonShort = jButtonShort;
	}

	public JButton getjButtonFloat() {
		return jButtonFloat;
	}

	public void setjButtonFloat(JButton jButtonFloat) {
		this.jButtonFloat = jButtonFloat;
	}

	private JTextField getJOffset() {
		if(jOffset == null) {
			jOffset = new JTextField();
			jOffset.setText("0");
		}
		return jOffset;
	}

	private JLabel getJLabel4() {
		if(jLabel4 == null) {
			jLabel4 = new JLabel();
			jLabel4.setText("Offset:");
		}
		return jLabel4;
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
