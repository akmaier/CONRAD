package edu.stanford.rsl.apps.gui;
import ij.IJ;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.BorderFactory;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextPane;
import javax.swing.border.TitledBorder;

import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.pipeline.BufferedProjectionSink;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;

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
public class GUICompatibleObjectVisualizationPanel extends JPanel implements ActionListener, UpdateableGUI {
	/**
	 * 
	 */
	private static final long serialVersionUID = 126994236281825512L;
	private JButton jConfigureButton;
	private JCheckBox jIsConfiguredCheckBox;
	private JScrollPane jScrollPane1;
	private JTextPane jCitationTextField;
	private Citeable citation;
	private GUIConfigurable config;
	private TitledBorder border;
	private UpdateableGUI parentFrame = null;
	
	public TitledBorder getBorder() {
		return border;
	}

	public GUICompatibleObjectVisualizationPanel(ImageFilteringTool tool){
		citation = tool;
		config = tool;
		initGUI();
		updateGUI();
	}
	
	public GUICompatibleObjectVisualizationPanel(GUIConfigurable config, Citeable citation){
		this.citation = citation;
		this.config = config;
		initGUI();
		updateGUI();
	}
	
	public GUICompatibleObjectVisualizationPanel(BufferedProjectionSink sink) {
		this.citation = sink;
		this.config = sink;
		initGUI();
		updateGUI();
	}

	private void initGUI() {
		try {
			{
				border = BorderFactory.createTitledBorder("Title");
				this.setBorder(border);
				GridBagLayout thisLayout = new GridBagLayout();
				thisLayout.rowWeights = new double[] {0.0, 0.1};
				thisLayout.rowHeights = new int[] {12, 7};
				thisLayout.columnWeights = new double[] {0.0, 0.0, 0.1};
				thisLayout.columnWidths = new int[] {344, 109, 7};
				this.setLayout(thisLayout);
				{
					jConfigureButton = new JButton();
					jConfigureButton.addActionListener(this);
					this.add(jConfigureButton, new GridBagConstraints(1, 0, 1, 2, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
					jConfigureButton.setText("configure...");
				}
				{
					jIsConfiguredCheckBox = new JCheckBox();
					this.add(jIsConfiguredCheckBox, new GridBagConstraints(2, 0, 1, 2, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
					jIsConfiguredCheckBox.setText("configured?");
					jIsConfiguredCheckBox.setEnabled(false);
				}
				{
					jScrollPane1 = new JScrollPane();
					this.add(jScrollPane1, new GridBagConstraints(0, 1, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.BOTH, new Insets(0, 0, 0, 0), 0, 0));
					{
						jCitationTextField = new JTextPane();
						jScrollPane1.setViewportView(jCitationTextField);
						jCitationTextField.setEditable(false);
						jCitationTextField.setBackground(Color.WHITE);
					}
				}
				Dimension d = new Dimension(550, 100); 
				this.setSize(d);
				this.setMinimumSize(d);
				this.setMaximumSize(d);
				setBackground(Color.WHITE);
			}
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	public void setBackground(Color color){
		super.setBackground(color);
		if (jConfigureButton != null) jConfigureButton.setBackground(color);
		if (jIsConfiguredCheckBox != null) jIsConfiguredCheckBox.setBackground(color);
	}
	
	public void setParentFrame(UpdateableGUI parentFrame) {
		this.parentFrame = parentFrame;
	}

	public UpdateableGUI getParentFrame() {
		return parentFrame;
	}

	public void setVisualizedObject(BufferedProjectionSink sink){
		citation = sink;
		config = sink;
		updateGUI();
	}
	
	public void updateGUI(){
		if (citation != null) {
		border.setTitle(citation.toString());
		} else {
			border.setTitle("Not configured");
		}
		if (config != null) {
		jIsConfiguredCheckBox.setSelected(config.isConfigured());
		if (Configuration.getGlobalConfiguration().getCitationFormat() == Configuration.MEDLINE_CITATION_FORMAT){
			jCitationTextField.setText(citation.getMedlineCitation());
			jCitationTextField.setFont(new Font(Font.SERIF, Font.PLAIN, 11));
		}
		if (Configuration.getGlobalConfiguration().getCitationFormat() == Configuration.BIBTEX_CITATION_FORMAT){
			jCitationTextField.setText(citation.getBibtexCitation());
			jCitationTextField.getCaret().setDot(0);
			jCitationTextField.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 11));
		}
		} else {
			jIsConfiguredCheckBox.setSelected(false);
		}
	}
	
	public void actionPerformed(ActionEvent e) {
		Object source = e.getSource();
		if (source != null){
			if (source.equals(jConfigureButton)){
				try {
					config.configure();
				} catch (Exception e1) {
					CONRAD.log(e1.getLocalizedMessage());
					System.out.println(e1.getLocalizedMessage());
				}
			}
		}
		updateGUI();
		if (parentFrame != null) parentFrame.updateGUI();
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
