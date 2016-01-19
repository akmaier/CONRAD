/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.apps.gui;

import ij.ImagePlus;

import java.awt.Color;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.io.File;
import java.io.IOException;

import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.Clip;
import javax.sound.sampled.DataLine;
import javax.swing.AbstractAction;
import javax.swing.JButton;
import javax.swing.BorderFactory;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.text.DefaultEditorKit;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.io.FileProjectionSource;
import edu.stanford.rsl.conrad.io.ImagePlusProjectionDataSource;
import edu.stanford.rsl.conrad.pipeline.BufferedProjectionSink;
import edu.stanford.rsl.conrad.pipeline.ParallelImageFilterPipeliner;
import edu.stanford.rsl.conrad.pipeline.ProjectionSource;
import edu.stanford.rsl.conrad.reconstruction.ReconstructionFilter;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.DicomConfigurationUpdater;
import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;
import edu.stanford.rsl.conrad.utils.FileUtil;
import edu.stanford.rsl.conrad.utils.GUIUtil;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.RegKeys;


/**
 * This code was edited or generated using CloudGarden's Jigloo
 * SWT/Swing GUI Builder, which is free for non-commercial
 * use. If Jigloo is being used commercially (ie, by a corporation,
 * company or business for any purpose whatever) then you
 * should purchase a license for each developer using Jigloo.
 * Please visit www.cloudgarden.com for details.
 * Use of Jigloo implies acceptance of these licensing terms.
 * A COMMERCIAL LICENSE HAS NOT BEEN PURCHASED FOR
 * THIS MACHINE, SO JIGLOO OR THIS CODE CANNOT BE USED
 * LEGALLY FOR ANY CORPORATE OR COMMERCIAL PURPOSE.
 */
public class ReconstructionPipelineFrame extends JFrame implements ActionListener, UpdateableGUI {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -2788658929679662808L;
	private JLabel jLabel1;
	private JTextField jProjectionsTextField;
	private JPanel jPanel1;
	private JButton jLoadDataButton;
	private JButton jEditConfigurationButton;
	private JButton jEditPipelineButton;
	private JButton jChooseBackprojector;
	private JLabel jPipelineLabel;
	private JButton jReconstructButton;
	private JButton jProjectionDataChooseButton;
	private JLabel jProjectionsLabel;
	private GUICompatibleObjectVisualizationPanel sinkPanel;
	private BufferedProjectionSink sink;
	private boolean debug = false;
	private ConfigurePipelineFrame pipelineFrame = null;
	private ConfigurationFrame configFrame = null;
	private ParallelImageFilterPipeliner filteringPipeline;
	private Clip soundClip = null;


	public ReconstructionPipelineFrame() {
		sink = Configuration.getGlobalConfiguration().getSink();
		initGUI();
		updateGUI();
		
		// Load the sound file and setup playback
		if (Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.SOUND_FILE) != null) {
			String name = Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.SOUND_FILE);
			if (name.isEmpty())
				return;
			
			AudioInputStream as = null;
			try {
				File in = new File(name);
				as = AudioSystem.getAudioInputStream(in);
			} catch (Exception ex) {
				System.out.println("Failed to load sound file " + name);
				return;
			}
			
			DataLine.Info info = new DataLine.Info(Clip.class, as.getFormat());
			if (!AudioSystem.isLineSupported(info)) {
				System.out.println("Audio output is not supported for this format");
				return;
			}
			
			try {
				soundClip = (Clip)AudioSystem.getLine(info);
				soundClip.open(as);
			} catch (Exception ex) {
				soundClip = null;
				System.out.println("Failed to setup sound playback (sound card probably busy)");
				return;
			}
		}
	}

	public void updateGUI(){
		sinkPanel.updateGUI();
		repaint();
	}
	
    /**
     * Create an Edit menu to support cut/copy/paste.
     */
    @SuppressWarnings("serial")
	public JMenuBar createMenuBar () {
        JMenuItem menuItem = null;
        JMenuBar menuBar = new JMenuBar();
        
        JMenu mainMenu = new JMenu("Configuration");
        mainMenu.setMnemonic(KeyEvent.VK_C);
        
        
        menuItem = new JMenuItem(new AbstractAction(){
			public void actionPerformed(ActionEvent e) {
				try {
					String filename = FileUtil.myFileChoose(".xml", false);
					Configuration.setGlobalConfiguration(Configuration.loadConfiguration(filename));
					if (configFrame != null) configFrame.exit();
					if (pipelineFrame != null) pipelineFrame.exit();
				} catch (Exception e1) {
					System.out.println(e1.getLocalizedMessage());
				}
			}
        });
        menuItem.setText("Load");
        menuItem.setMnemonic(KeyEvent.VK_L);
        mainMenu.add(menuItem);

        menuItem = new JMenuItem(new AbstractAction(){
			public void actionPerformed(ActionEvent e) {
				try {
					String filename = FileUtil.myFileChoose(".xml", true);
					if (!filename.endsWith(".xml")) filename += ".xml";
					Configuration.saveConfiguration(Configuration.getGlobalConfiguration(), filename);
					Configuration.loadConfiguration(filename);
				} catch (Exception e1) {
					System.out.println(e1.getLocalizedMessage());
				}
			}
        });
        menuItem.setText("Save");
        menuItem.setMnemonic(KeyEvent.VK_S);
        mainMenu.add(menuItem);

        menuBar.add(mainMenu);
        
        mainMenu = new JMenu("Edit");
        mainMenu.setMnemonic(KeyEvent.VK_E);

        menuItem = new JMenuItem(new DefaultEditorKit.CutAction());
        menuItem.setText("Cut");
        menuItem.setMnemonic(KeyEvent.VK_T);
        mainMenu.add(menuItem);

        menuItem = new JMenuItem(new DefaultEditorKit.CopyAction());
        menuItem.setText("Copy");
        menuItem.setMnemonic(KeyEvent.VK_C);
        mainMenu.add(menuItem);

        menuItem = new JMenuItem(new DefaultEditorKit.PasteAction());
        menuItem.setText("Paste");
        menuItem.setMnemonic(KeyEvent.VK_P);
        mainMenu.add(menuItem);

        menuBar.add(mainMenu);
        
        
        return menuBar;
    }


	private void initGUI() {
		Configuration config = Configuration.getGlobalConfiguration();
		try {
			{
				{
					// Set Look & Feel
					try {
						javax.swing.UIManager
						.setLookAndFeel("com.sun.java.swing.plaf.windows.WindowsLookAndFeel");
					} catch (Exception e) {
						//e.printStackTrace();
					}
				}
				getContentPane().setBackground(Color.WHITE);
				this.setSize(640, 480);
				this.setTitle("Reconstruction Pipeline " + CONRAD.VersionString);
				this.setJMenuBar(createMenuBar());
				GridBagLayout thisLayout = new GridBagLayout();
				thisLayout.rowWeights = new double[] {0.1, 0.1, 0.1, 0.1, 0.1 };
				thisLayout.rowHeights = new int[] {50, 50, 50, 50, 7 };
				thisLayout.columnWeights = new double[] { 0.1, 0.1, 0.1, 0.1,
						0.1 };
				thisLayout.columnWidths = new int[] { 7, 7, 223, 7, 7 };
				getContentPane().setLayout(thisLayout);
				{
					jLabel1 = new JLabel();
					getContentPane().add(jLabel1, new GridBagConstraints(1, 0, 3, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
					jLabel1
					.setText(CONRAD.CONRADDefinition);
				}
				jPanel1 = new JPanel();
				jPanel1.setBorder(BorderFactory.createTitledBorder("Input"));
				jPanel1.setBackground(Color.WHITE);
				GridBagLayout jPanel1Layout = new GridBagLayout();
				getContentPane().add(
						jPanel1,
						new GridBagConstraints(1, 1, 3, 1, 0.0, 0.0,
								GridBagConstraints.CENTER,
								GridBagConstraints.BOTH, new Insets(0, 0,
										0, 0), 0, 0));
				jPanel1Layout.rowWeights = new double[] { 0.1, 0.1 };
				jPanel1Layout.rowHeights = new int[] { 40, 20 };
				jPanel1Layout.columnWeights = new double[] {0.1, 0.1, 0.1};
				jPanel1Layout.columnWidths = new int[] {80, 80, 80};
				jPanel1.setLayout(jPanel1Layout);
				
				{
					jProjectionsTextField = new JTextField();
					jPanel1.add(
							jProjectionsTextField,
							new GridBagConstraints(1, 0, 1, 1, 0.0, 0.0,
									GridBagConstraints.NORTH,
									GridBagConstraints.HORIZONTAL, new Insets(
											0, 0, 0, 0), 0, 0));
					jProjectionsTextField.setText(config.getRecentFileOne());
					jProjectionsTextField.addActionListener(this);
					GUIUtil.enableDragAndDrop(jProjectionsTextField);
					jProjectionsLabel = new JLabel();
					jPanel1.add(
							jProjectionsLabel,
							new GridBagConstraints(0, 0, 1, 1, 0.0, 0.0,
									GridBagConstraints.NORTH,
									GridBagConstraints.NONE, new Insets(0, 0,
											0, 0), 0, 0));
					jProjectionsLabel.setText("Projection Data:");
					jProjectionDataChooseButton = new JButton();
					jPanel1.add(jProjectionDataChooseButton, new GridBagConstraints(2, 0, 1, 1, 0.0, 0.0, GridBagConstraints.NORTH, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
					jProjectionDataChooseButton.setText("Choose");
					jProjectionDataChooseButton.addActionListener(this);
					jLoadDataButton = new JButton();
					jPanel1.add(jLoadDataButton, new GridBagConstraints(1, 1, 1, 1, 0.0, 0.0, GridBagConstraints.NORTH, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
					jLoadDataButton.setText("Use Current ImagePlus");
					jLoadDataButton.addActionListener(this);
				}
				jPanel1 = new JPanel();
				jPanel1.setBorder(BorderFactory.createTitledBorder("Processing"));
				jPanel1.setBackground(Color.WHITE);
				jPanel1Layout = new GridBagLayout();
				getContentPane().add(
						jPanel1,
						new GridBagConstraints(1, 2, 3, 1, 0.0, 0.0,
								GridBagConstraints.CENTER,
								GridBagConstraints.BOTH, new Insets(0, 0,
										0, 0), 0, 0));
				jPanel1Layout.rowWeights = new double[] { 0.1, 0.1, 0.1 };
				jPanel1Layout.rowHeights = new int[] { 50, 0, 0 };
				jPanel1Layout.columnWeights = new double[] {0.1,  0.1};
				jPanel1Layout.columnWidths = new int[] {80, 80};
				jPanel1.setLayout(jPanel1Layout);
				{
					{
						jEditPipelineButton = new JButton();
						jPanel1.add(jEditPipelineButton, new GridBagConstraints(0, 0, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
						jEditPipelineButton.setText("Edit Pipeline");
						jEditPipelineButton.addActionListener(this);
					}
					{
						jEditConfigurationButton = new JButton();
						jPanel1.add(jEditConfigurationButton, new GridBagConstraints(1, 0, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
						jEditConfigurationButton.setText("Edit Configuration");
						jEditConfigurationButton.addActionListener(this);
					}

				}
				jPanel1 = new JPanel();
				jPanel1.setBorder(BorderFactory.createTitledBorder("Output"));
				jPanel1.setBackground(Color.WHITE);
				jPanel1Layout = new GridBagLayout();
				getContentPane().add(
						jPanel1,
						new GridBagConstraints(1, 3, 3, 1, 0.0, 0.0,
								GridBagConstraints.CENTER,
								GridBagConstraints.BOTH, new Insets(0, 0,
										0, 0), 0, 0));
				jPanel1Layout.rowWeights = new double[] { 0.1, 0.1, 0.1 };
				jPanel1Layout.rowHeights = new int[] { 30, 80, 0 };
				jPanel1Layout.columnWeights = new double[] {0.1, 0.1, 0.1};
				jPanel1Layout.columnWidths = new int[] {80, 80, 80};
				jPanel1.setLayout(jPanel1Layout);
				{
					sinkPanel = new GUICompatibleObjectVisualizationPanel(sink);
					jPanel1.add(sinkPanel, new GridBagConstraints(0, 1, 3, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.BOTH, new Insets(0, 0, 0, 0), 0, 0));
					sinkPanel.setParentFrame(this);
				
					jChooseBackprojector = new JButton();
					jPanel1.add(jChooseBackprojector, new GridBagConstraints(1, 0, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
					jChooseBackprojector.setText("Choose Output");
					jChooseBackprojector.addActionListener(this);
				}
			}
			jPipelineLabel = new JLabel();
			getContentPane().add(jPipelineLabel, new GridBagConstraints(2, 0, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(30, 0, 0, 0), 0, 0));
			jPipelineLabel.setText("Reconstruction Pipeline");
			{
				jReconstructButton = new JButton();
				getContentPane().add(jReconstructButton, new GridBagConstraints(1, 4, 3, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
				jReconstructButton.setText("Reconstruct");
				jReconstructButton.addActionListener(this);
			}
			


		} catch (Exception e) {
			e.printStackTrace();
		}
	}


	ImagePlus volumePreview = null;
	ProjectionSource pSource = null;

	public void actionPerformed(ActionEvent e) {
		Object source = e.getSource();
		if (source != null){
			//jChooseBackprojector
			if (source.equals(this.jProjectionDataChooseButton)){
				try {
					String file = FileUtil.myFileChoose("", false);
					this.jProjectionsTextField.setText(file);
					if (Configuration.getGlobalConfiguration().getImportFromDicomAutomatically()){
						DicomConfigurationUpdater update = new DicomConfigurationUpdater();
						Configuration config = Configuration.getGlobalConfiguration();
						update.setConfig(config);
						update.setFilename(this.jProjectionsTextField.getText());
						update.readConfiguration();
						config.setGeometry(Configuration.loadGeometrySource(config));
						Configuration.setGlobalConfiguration(config);
					}
				} catch (Exception e1) {
					// TODO Auto-generated catch block
					//e1.printStackTrace();
				}
			}
			if (source.equals(this.jEditPipelineButton)){
				if (pipelineFrame == null){
					pipelineFrame = new ConfigurePipelineFrame();
					pipelineFrame.setParentFrame(this);
					pipelineFrame.setVisible(true);
					pipelineFrame.setLocation(CONRAD.getWindowTopCorner());
				} else {
					if (pipelineFrame.isExited()){
						pipelineFrame = new ConfigurePipelineFrame();
						pipelineFrame.setParentFrame(this);
						pipelineFrame.setVisible(true);		
					} else {
						if (! pipelineFrame.isVisible()) {
							pipelineFrame.setVisible(true);
						}
					}
				}
			}
			if (source.equals(this.jEditConfigurationButton)){
				if (configFrame == null){
					configFrame = new ConfigurationFrame();
					
					configFrame.setParentFrame(this);
					configFrame.setVisible(true);
					configFrame.setLocation(CONRAD.getWindowTopCorner());
				} else {
					if (configFrame.isExited()){
						configFrame = new ConfigurationFrame();
						configFrame.setParentFrame(this);
						configFrame.setVisible(true);		
					} else {
						if (! configFrame.isVisible()) {
							configFrame.setVisible(true);
						}
					}
				}
			}
			if (source.equals(this.jLoadDataButton)){
				pSource = new ImagePlusProjectionDataSource();
				((ImagePlusProjectionDataSource)pSource).initStream(null);
			}
			if (source.equals(this.jChooseBackprojector)){
				Configuration config = Configuration.getGlobalConfiguration(); 
				BufferedProjectionSink [] sinks = BufferedProjectionSink.getProjectionDataSinks();
				BufferedProjectionSink selected = (BufferedProjectionSink) JOptionPane.showInputDialog(this, "Please select the reconstruction algorithm: ", "Select Reconstruction Algorithm", JOptionPane.INFORMATION_MESSAGE, null, sinks, sink);
				if (selected != null) {
					sink = selected;
					sink.setConfiguration(config);
					sinkPanel.setVisualizedObject(sink);
					config.setSink(sink);
				}
				updateGUI();
			}
			if (source.equals(this.jReconstructButton)){
				if (pSource == null){
					try {
						pSource = FileProjectionSource.openProjectionStream(jProjectionsTextField.getText());
					} catch (IOException e1) {
						if (debug) e1.printStackTrace();
						JOptionPane.showMessageDialog(this, "Could not open projection source.");
					}
				}
				if (pSource != null){
					if (sink.isConfigured()) {
						ImageFilteringTool [] filters = Configuration.getGlobalConfiguration().getFilterPipeline();
						boolean isConfigured = true;
						boolean recoTest = false;
						
						for (int i = 0; i < filters.length; i++){
							if (!filters[i].isConfigured()) isConfigured = false;
							if(filters[i] instanceof ReconstructionFilter) recoTest = true;
						}
						final boolean isReconstruction = recoTest;
						if (isConfigured) {
							volumePreview = ImageUtil.wrapGrid3D(sink.getProjectionVolume(), "Preview");
							if (volumePreview != null) volumePreview.show();
							filteringPipeline = new ParallelImageFilterPipeliner(pSource, filters, sink);
							Thread thread = new Thread(new Runnable(){
								public void run(){
									long time = System.currentTimeMillis();
									try {
										filteringPipeline.project();
									} catch (Exception e) {
										e.printStackTrace();
									}
									Grid3D result = sink.getResult();
									
									time = System.currentTimeMillis() - time;
									if (volumePreview == null) {
										volumePreview = ImageUtil.wrapGrid3D(result, "Result of " + pSource);
										
										
										ImageUtil.applyConradImageCalibration(volumePreview, isReconstruction);
										
										
										if (volumePreview != null){
											volumePreview.show();
										}
									}
									if (volumePreview != null){
										File filename = new File(jProjectionsTextField.getText());
										volumePreview.setTitle(filename.getName());
										if (pSource instanceof ImagePlusProjectionDataSource){
											ImagePlusProjectionDataSource s = (ImagePlusProjectionDataSource) pSource;
											volumePreview.setTitle("Reconstruction of " + s.getTitle());
										}
									}
									filteringPipeline = null;
									pSource = null;
									CONRAD.log("Reconstruction time (sec):" + time);
									try {
										sink = sink.getClass().newInstance();
										sinkPanel.setVisualizedObject(sink);
										updateGUI();
									} catch (InstantiationException e) {
										// TODO Auto-generated catch block
										e.printStackTrace();
									} catch (IllegalAccessException e) {
										// TODO Auto-generated catch block
										e.printStackTrace();
									}
									DoubleArrayUtil.visualizeBufferedArrays("Current Weighting");
									
									// Play the sound if so desired
									if (soundClip != null) {
										soundClip.setFramePosition(0);
										soundClip.start();
									}
								}
							});
							thread.start();

						} else {
							JOptionPane.showMessageDialog(this, "Not all filters of the reconstruction pipeline are configured. Please correct this.");
						}


					} else {
						JOptionPane.showMessageDialog(this, "Reconstruction Algorithm is not configured.");
					}
				}
			}
			updateGUI();
		}
	}

	public static void main(String [] args){
		CONRAD.setup();
		ReconstructionPipelineFrame oscar = new ReconstructionPipelineFrame();
		oscar.setVisible(true);
		oscar.setLocation(CONRAD.getWindowTopCorner());
	}

}
