package edu.stanford.rsl.apps.gui;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.Rectangle;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import javax.swing.JButton;

import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.pipeline.ParallelImageFilterPipeliner;
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
public class ConfigurePipelineFrame extends JFrame implements ActionListener, MouseListener, UpdateableGUI {
	/**
	 * 
	 */
	private static final long serialVersionUID = -6842659152489589850L;
	private JScrollPane jScrollPane1;
	private JButton jMoveDownButton;
	private JPanel jPipelinePanel;
	private JButton jSaveButton;
	private JButton jMoveUpButton;
	private JButton jRemoveButton;
	private JButton jAddButton;
	private ImageFilteringTool [] pipeline;
	private boolean exited = false;
	private UpdateableGUI parentFrame = null;
	private boolean saveToDisk = true;


	public ConfigurePipelineFrame(){
		pipeline = Configuration.getGlobalConfiguration().getFilterPipeline();
		initGUI();
		updateGUI();
		//pack();
	}

	public boolean isExited() {
		return exited;
	}

	public void exit(){
		exited = true;
		setVisible(false);
		if (parentFrame != null) parentFrame.updateGUI();
		System.out.println("exiting pipeline frame");
	}
	
	private GUICompatibleObjectVisualizationPanel [] panels = null;
	private int selectedIndex = 0;

	public void updateGUI(){
		Rectangle rect = jPipelinePanel.getVisibleRect();
		int visibleYCoord = 100*selectedIndex;
		if (!rect.contains(0, visibleYCoord)){
			rect = new Rectangle(0, visibleYCoord, jScrollPane1.getWidth(), visibleYCoord + jScrollPane1.getHeight());
		}
		visibleYCoord = 100*(selectedIndex+1);
		if (!rect.contains(0, visibleYCoord)){
			rect = new Rectangle(0, visibleYCoord - jScrollPane1.getHeight(), jScrollPane1.getWidth(), visibleYCoord);
		}
		jPipelinePanel = new JPanel();
		jPipelinePanel.setBackground(Color.WHITE);
		jPipelinePanel.setLayout(null);
		panels = new GUICompatibleObjectVisualizationPanel[pipeline.length];
		for (int i = 0; i < pipeline.length; i++){
			GUICompatibleObjectVisualizationPanel currentPanel = new GUICompatibleObjectVisualizationPanel(pipeline[i]);
			jPipelinePanel.add(currentPanel);		
			currentPanel.setLocation(0, 100 *i);
			currentPanel.addMouseListener(this);
			currentPanel.setParentFrame(this);
			panels[i] = currentPanel;
		}
		if((selectedIndex < panels.length)&&(selectedIndex >= 0)){
			panels[selectedIndex].setBackground(new Color(255,200,200));
		}
		jPipelinePanel.setPreferredSize(new Dimension(550, 100 * pipeline.length));
		jScrollPane1.setViewportView(jPipelinePanel);
		jScrollPane1.setPreferredSize(new Dimension(500,440));
		jPipelinePanel.scrollRectToVisible(rect);

	}

	private void initGUI() {
		try {
			{
				GridBagLayout thisLayout = new GridBagLayout();
				thisLayout.rowWeights = new double[] {0.0, 0.1, 0.1, 0.1};
				thisLayout.rowHeights = new int[] {30, 223, 223, 30};
				thisLayout.columnWeights = new double[] {0.1, 0.0, 0.0, 0.1};
				thisLayout.columnWidths = new int[] {0, 20, 570, 20};
				this.setBackground(Color.WHITE);
				getContentPane().setLayout(thisLayout);
				getContentPane().setBackground(Color.WHITE);
				setTitle("Configure Pipeline");
				{
					jScrollPane1 = new JScrollPane();
					getContentPane().add(jScrollPane1, new GridBagConstraints(2, 1, 1, 2, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.BOTH, new Insets(0, 0, 0, 0), 0, 0));
					{
						jPipelinePanel = new JPanel();
						jPipelinePanel.setBackground(Color.WHITE);
						jScrollPane1.setViewportView(jPipelinePanel);
					}
				}
				{
					jAddButton = new JButton();
					getContentPane().add(jAddButton, new GridBagConstraints(2, 0, 1, 1, 0.0, 0.0, GridBagConstraints.SOUTHWEST, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
					jAddButton.setText("add");
					jAddButton.addActionListener(this);
				}
				{
					jRemoveButton = new JButton();
					getContentPane().add(jRemoveButton, new GridBagConstraints(2, 0, 1, 1, 0.0, 0.0, GridBagConstraints.SOUTHWEST, GridBagConstraints.NONE, new Insets(0, 50, 0, 0), 0, 0));
					jRemoveButton.setText("remove");
					jRemoveButton.addActionListener(this);
				}
				{
					jMoveUpButton = new JButton();
					getContentPane().add(jMoveUpButton, new GridBagConstraints(2, 0, 1, 1, 0.0, 0.0, GridBagConstraints.SOUTHEAST, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
					jMoveUpButton.setText("up");
					jMoveUpButton.addActionListener(this);
				}
				{
					jMoveDownButton = new JButton();
					getContentPane().add(jMoveDownButton, new GridBagConstraints(2, 0, 1, 1, 0.0, 0.0, GridBagConstraints.SOUTHEAST, GridBagConstraints.NONE, new Insets(0, 0, 0, 50), 0, 0));
					jMoveDownButton.setText("down");
					jMoveDownButton.addActionListener(this);
				}
				{
					jSaveButton = new JButton();
					getContentPane().add(jSaveButton, new GridBagConstraints(2, 3, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
					jSaveButton.setText("save and exit");
					jSaveButton.addActionListener(this);
				}
			}
			pack();
		} catch(Exception e) {
			e.printStackTrace();
		}
	}

	public void actionPerformed(ActionEvent e) {
		Object source = e.getSource();
		if (source != null){
			if (source.equals(this.jSaveButton)){
				Configuration config = Configuration.getGlobalConfiguration();
				config.setFilterPipeline(pipeline);
				ImageFilteringTool [] backup = ParallelImageFilterPipeliner.getPipelineClone(pipeline);
				Configuration.setGlobalConfiguration(config);
				if (saveToDisk) Configuration.saveConfiguration();
				config.setFilterPipeline(backup);
				Configuration.setGlobalConfiguration(config);
				exited = true;
				if (parentFrame != null) parentFrame.updateGUI();
				this.setVisible(false);
			}
			if (source.equals(this.jAddButton)){
				ImageFilteringTool [] filters = ImageFilteringTool.getFilterTools();
				ImageFilteringTool newFilter = (ImageFilteringTool) JOptionPane.showInputDialog(null, "Please select the filter type to add:", "Filter Selection", JOptionPane.DEFAULT_OPTION, null, filters, filters[0]);
				if (newFilter != null) {
					ImageFilteringTool [] newPipeline = new ImageFilteringTool [pipeline.length + 1];
					for (int i = 0; i < newPipeline.length; i++){
						if (i < selectedIndex) newPipeline[i] = pipeline[i];
						if (i == selectedIndex) {
							newPipeline[i] = newFilter;
							i++;
						}
						if (i > selectedIndex){
							if ((i < newPipeline.length) && ((i -1 >=0))) {			
								newPipeline[i] = pipeline[i-1];
							}
						}
					}
					if (selectedIndex  == -1) {
						newPipeline[0] = newFilter;
					}
					pipeline = newPipeline;
				}
				this.updateGUI();
			}
			if (source.equals(this.jRemoveButton)){
				ImageFilteringTool [] newPipeline = new ImageFilteringTool [pipeline.length -1];
				for (int i = 0; i < newPipeline.length + 1; i++){
					if (i < selectedIndex) newPipeline[i] = pipeline[i];
					if (i == selectedIndex) {
						i++;
					}
					if ((i > selectedIndex)&&(i < newPipeline.length + 1)){
						newPipeline[i-1] = pipeline[i];
					}
				}
				if(selectedIndex >= newPipeline.length) selectedIndex--;
				pipeline = newPipeline;
				this.updateGUI();
			}
			if (source.equals(this.jMoveUpButton)){
				if (selectedIndex != 0) {
					ImageFilteringTool [] newPipeline = new ImageFilteringTool [pipeline.length];
					for (int i = 0; i < newPipeline.length; i++){
						if (i < selectedIndex - 1) newPipeline[i] = pipeline[i];
						if (i == selectedIndex) {
							newPipeline[i-1] = pipeline[i];
							newPipeline[i] = pipeline[i-1];
						}
						if (i > selectedIndex){
							newPipeline[i] = pipeline[i];
						}
					}
					selectedIndex--;
					pipeline = newPipeline;
					this.updateGUI();
				}
			}
			if (source.equals(this.jMoveDownButton)){
				if (selectedIndex != pipeline.length - 1) {
					ImageFilteringTool [] newPipeline = new ImageFilteringTool [pipeline.length];
					for (int i = 0; i < newPipeline.length; i++){
						if (i < selectedIndex) newPipeline[i] = pipeline[i];
						if (i == selectedIndex) {
							newPipeline[i+1] = pipeline[i];
							newPipeline[i] = pipeline[i+1];
						}
						if (i > selectedIndex + 1){
							newPipeline[i] = pipeline[i];
						}
					}
					selectedIndex++;
					pipeline = newPipeline;
					this.updateGUI();
				}
			}
		}
	}

	public void setParentFrame(UpdateableGUI parentFrame) {
		this.parentFrame = parentFrame;
	}

	public UpdateableGUI getParentFrame() {
		return parentFrame;
	}

	public void mouseClicked(MouseEvent e) {
		for (int i = 0; i < panels.length; i++){
			if(e.getSource().equals(panels[i])) selectedIndex = i;
		}
		this.updateGUI();
	}

	public void mouseEntered(MouseEvent e) {

	}

	public void mouseExited(MouseEvent e) {

	}

	public void mousePressed(MouseEvent e) {

	}

	public void mouseReleased(MouseEvent e) {

	}

	/**
	 * @param saveToDisk the saveToDisk to set
	 */
	public void setSaveToDisk(boolean saveToDisk) {
		this.saveToDisk = saveToDisk;
	}

	/**
	 * @return the saveToDisk
	 */
	public boolean isSaveToDisk() {
		return saveToDisk;
	}


}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
