

package edu.stanford.rsl.conrad.phantom.electrondensity;

import ij.gui.GenericDialog;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;

import javax.swing.AbstractButton;
import javax.swing.GroupLayout;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JTextField;
import javax.swing.LayoutStyle;
import javax.swing.text.JTextComponent;

import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;

/**
 * Temporary GUI
 * @author Rotimi X Ojo
 *
 */

public class CrisEDPhantomGUI extends JPanel implements ActionListener,ItemListener{

	private static final long serialVersionUID = 8752339017895320468L;
	
	
	private GenericDialog gd;
	//
	private JPanel innerRing;
	private JPanel innerRingHolder;
	private JPanel outerRing;
	private JRadioButton [] innerBut;
	private JRadioButton [] outterBut;
	private JTextComponent insertNameField;
	private JTextField materialField;
	private AbstractButton innerDiskActivator;
	private JLabel insertNameLabel;
	private JLabel materialLabel;
	private JLabel instructionLabel;
	private JCheckBox bufferedMaterial;
	private JLabel bufferedMaterialLabel;
	private JButton okButton;
	private JRadioButton currButton;

	private CrisEDPhantomM062 data;

	public CrisEDPhantomGUI (CrisEDPhantomM062 data) {
		this.data = data;
		gd = new GenericDialog("Configure Phantom");
		initComponents();
		createLayout();
		gd.add(this);
		gd.showDialog();			
	}

	
	private void initComponents() {
		innerBut = new JRadioButton[9];
		outterBut = new JRadioButton[8];
		initButtons(innerBut, "I",0);
		initButtons(outterBut, "O",0);
		//
		outerRing = new JPanel();
	    outerRing.setBackground(Color.BLACK); 
        outerRing.setName("OutterRing"); 
        //
        innerRing = new JPanel();
        innerRingHolder = new JPanel();
        innerRing.setBackground(Color.BLACK); 
        innerRing.setName("Inner Ring"); 
        innerRingHolder.setBackground(Color.BLACK); 
        
        //
        instructionLabel = new JLabel("Click on slot to add or remove insert.");
        instructionLabel.setName("instructionLabel"); 
        //
        insertNameLabel = new JLabel("Insert: ");
        insertNameLabel.setName("insertNameLabel");        
        insertNameField = new JTextField("NULL");
        insertNameField.setEditable(false);
        insertNameField.setName("insertNameField"); 
        //
        materialLabel = new JLabel("Material: ");
        materialLabel.setName("materialLabel");        
        materialField = new JTextField("air");
        materialField.setName("materialField"); 
        materialField.setMinimumSize(new Dimension(100,10));
        //
        innerDiskActivator = new JRadioButton("Add Inner Disk");
        innerDiskActivator.setSelected(true);
        innerDiskActivator.setToolTipText("Adds or Remove Inner Disk"); 
        innerDiskActivator.setName("innerDiskActivator"); 
        innerDiskActivator.addItemListener(this);
        //
        bufferedMaterialLabel = new JLabel(" Water Buffered:");
        bufferedMaterialLabel.setToolTipText("Check if material is buffered with water");
        bufferedMaterial = new JCheckBox();
        bufferedMaterial.setToolTipText("Check if material is buffered with water");
        //
        okButton = new JButton("OK");
        okButton.setName("okButton");
        okButton.addActionListener(this);

	}
	
	private void initButtons(JRadioButton[] buts, String prefix, int offset) {
		for(int i = 0; i < buts.length; i++){
			buts[i] = getNewRadioButton(prefix + "-" + (i+offset));
		}		
	}


	private void createLayout() {
		  GroupLayout jPanel7Layout = new GroupLayout(innerRing);
	        innerRing.setLayout(jPanel7Layout);
	        jPanel7Layout.setHorizontalGroup(
	            jPanel7Layout.createParallelGroup(GroupLayout.Alignment.LEADING)
	            .addGroup(jPanel7Layout.createSequentialGroup()
	                .addGap(15, 15, 15)
	                .addComponent(innerBut[3])
	                .addComponent(innerBut[2])
	                .addComponent(innerBut[1]))
	            .addGroup(jPanel7Layout.createSequentialGroup()
	                .addGap(6, 6, 6)
	                .addComponent(innerBut[4])
	                .addGap(10, 10, 10)
	                .addComponent(innerBut[8])
	                .addGap(8, 8, 8)
	                .addComponent(innerBut[0]))
	            .addGroup(jPanel7Layout.createSequentialGroup()
	                .addGap(15, 15, 15)
	                .addComponent(innerBut[5])
	                .addComponent(innerBut[6])
	                .addComponent(innerBut[7]))
	        );
	        jPanel7Layout.setVerticalGroup(
	            jPanel7Layout.createParallelGroup(GroupLayout.Alignment.LEADING)
	            .addGroup(jPanel7Layout.createSequentialGroup()
	                .addGroup(jPanel7Layout.createParallelGroup(GroupLayout.Alignment.LEADING)
	                    .addGroup(jPanel7Layout.createSequentialGroup()
	                        .addGap(7, 7, 7)
	                        .addComponent(innerBut[3]))
	                    .addComponent(innerBut[2])
	                    .addGroup(jPanel7Layout.createSequentialGroup()
	                        .addGap(7, 7, 7)
	                        .addComponent(innerBut[1])))
	                .addGroup(jPanel7Layout.createParallelGroup(GroupLayout.Alignment.LEADING)
	                    .addComponent(innerBut[4])
	                    .addComponent(innerBut[8])
	                    .addComponent(innerBut[0]))
	                .addGroup(jPanel7Layout.createParallelGroup(GroupLayout.Alignment.LEADING)
	                    .addComponent(innerBut[5])
	                    .addGroup(jPanel7Layout.createSequentialGroup()
	                        .addGap(7, 7, 7)
	                        .addComponent(innerBut[6]))
	                    .addComponent(innerBut[7])))
	        );
	        
	        innerRingHolder.setSize(innerRing.getPreferredSize());
	        innerRingHolder.add(innerRing);
	        
	        GroupLayout OutterRingLayout = new GroupLayout(outerRing);
	        outerRing.setLayout(OutterRingLayout);
	        OutterRingLayout.setHorizontalGroup(
	            OutterRingLayout.createParallelGroup(GroupLayout.Alignment.LEADING)
	            .addGroup(OutterRingLayout.createSequentialGroup()
	                .addGap(31, 31, 31)
	                .addComponent(outterBut[4])
	                .addGap(2, 2, 2)
	                .addGroup(OutterRingLayout.createParallelGroup(GroupLayout.Alignment.LEADING)
	                    .addComponent(outterBut[3])
	                    .addGroup(OutterRingLayout.createSequentialGroup()
	                        .addGap(3, 3, 3)
	                        .addComponent(outterBut[5])))
	                .addGroup(OutterRingLayout.createParallelGroup(GroupLayout.Alignment.LEADING)
	                    .addGroup(OutterRingLayout.createSequentialGroup()
	                        .addGap(36, 36, 36)
	                        .addComponent(outterBut[2]))
	                    .addComponent(innerRingHolder, GroupLayout.PREFERRED_SIZE, 85, GroupLayout.PREFERRED_SIZE)
	                    .addGroup(OutterRingLayout.createSequentialGroup()
	                        .addGap(35, 35, 35)
	                        .addComponent(outterBut[6])))
	                .addGroup(OutterRingLayout.createParallelGroup(GroupLayout.Alignment.LEADING)
	                    .addGroup(OutterRingLayout.createSequentialGroup()
	                        .addGap(2, 2, 2)
	                        .addComponent(outterBut[1]))
	                    .addComponent(outterBut[7]))
	                .addGap(3, 3, 3)
	                .addComponent(outterBut[0]))
	        );
	        OutterRingLayout.setVerticalGroup(
	            OutterRingLayout.createParallelGroup(GroupLayout.Alignment.LEADING)
	            .addGroup(OutterRingLayout.createSequentialGroup()
	                .addGap(87, 87, 87)
	                .addComponent(outterBut[4]))
	            .addGroup(OutterRingLayout.createSequentialGroup()
	                .addGap(39, 39, 39)
	                .addComponent(outterBut[3])
	                .addGap(76, 76, 76)
	                .addComponent(outterBut[5]))
	            .addGroup(OutterRingLayout.createSequentialGroup()
	                .addGap(21, 21, 21)
	                .addComponent(outterBut[2])
	                .addGap(16, 16, 16)
	                .addComponent(innerRingHolder, GroupLayout.PREFERRED_SIZE, 80, GroupLayout.PREFERRED_SIZE)
	                .addGap(17, 17, 17)
	                .addComponent(outterBut[6]))
	            .addGroup(OutterRingLayout.createSequentialGroup()
	                .addGap(39, 39, 39)
	                .addComponent(outterBut[1])
	                .addGap(76, 76, 76)
	                .addComponent(outterBut[7]))
	            .addGroup(OutterRingLayout.createSequentialGroup()
	                .addGap(89, 89, 89)
	                .addComponent(outterBut[0]))
	        );


	        GroupLayout mainPanelLayout = new GroupLayout(this);
	        this.setLayout(mainPanelLayout);
	        mainPanelLayout.setHorizontalGroup(
	            mainPanelLayout.createParallelGroup(GroupLayout.Alignment.LEADING)
	            .addGroup(mainPanelLayout.createSequentialGroup()
	                .addGap(172, 172, 172)
	                .addComponent(okButton, GroupLayout.PREFERRED_SIZE, 71, GroupLayout.PREFERRED_SIZE)
	                .addContainerGap(206, Short.MAX_VALUE))
	            .addGroup(mainPanelLayout.createSequentialGroup()
	                .addGroup(mainPanelLayout.createParallelGroup(GroupLayout.Alignment.TRAILING)
	                    .addGroup(mainPanelLayout.createSequentialGroup()
	                        .addGap(26, 26, 26)
	                        .addComponent(outerRing, GroupLayout.DEFAULT_SIZE, 257, Short.MAX_VALUE))
	                    .addGroup(GroupLayout.Alignment.LEADING, mainPanelLayout.createSequentialGroup()
	                        .addGap(107, 107, 107)
	                        .addComponent(instructionLabel)))
	                .addGap(39, 39, 39)
	                .addGroup(mainPanelLayout.createParallelGroup(GroupLayout.Alignment.LEADING)
	                    .addGroup(mainPanelLayout.createSequentialGroup()
	                        .addGroup(mainPanelLayout.createParallelGroup(GroupLayout.Alignment.LEADING)
	                            .addComponent(insertNameLabel)
	                            .addComponent(materialLabel)
	                            .addComponent(bufferedMaterialLabel))	                            
	                        .addGap(18, 18, 18)
	                        .addGroup(mainPanelLayout.createParallelGroup(GroupLayout.Alignment.LEADING, false)
	                            .addComponent(materialField)
	                            .addComponent(insertNameField) 
	                            .addComponent(bufferedMaterial)))	                           
	                    .addComponent(innerDiskActivator))
	                .addGap(22, 22, 22))
	        );
	        mainPanelLayout.setVerticalGroup(
	            mainPanelLayout.createParallelGroup(GroupLayout.Alignment.LEADING)
	            .addGroup(GroupLayout.Alignment.TRAILING, mainPanelLayout.createSequentialGroup()
	                .addContainerGap()
	                .addComponent(instructionLabel)
	                .addGap(18, 18, 18)
	                .addComponent(outerRing, GroupLayout.PREFERRED_SIZE, 199, GroupLayout.PREFERRED_SIZE)
	                .addPreferredGap(LayoutStyle.ComponentPlacement.RELATED)
	                .addComponent(okButton)
	                .addContainerGap())
	            .addGroup(mainPanelLayout.createSequentialGroup()
	                .addGap(60, 60, 60)
	                .addGroup(mainPanelLayout.createParallelGroup(GroupLayout.Alignment.BASELINE)
	                    .addComponent(insertNameLabel)
	                    .addComponent(insertNameField))
	                .addPreferredGap(LayoutStyle.ComponentPlacement.UNRELATED)
	                .addGroup(mainPanelLayout.createParallelGroup(GroupLayout.Alignment.BASELINE)
	                    .addComponent(materialLabel)
	                    .addComponent(materialField)).addPreferredGap(LayoutStyle.ComponentPlacement.UNRELATED)
		                .addGroup(mainPanelLayout.createParallelGroup(GroupLayout.Alignment.BASELINE)                
	                    .addComponent(bufferedMaterialLabel)
	                    .addComponent(bufferedMaterial))
	                .addGap(67, 67, 67)
	                .addComponent(innerDiskActivator)
	                .addContainerGap(81, Short.MAX_VALUE))
	        );
		
	}




	private JRadioButton getNewRadioButton(String name){
		JRadioButton  button = new JRadioButton();
		button.setBackground(Color.BLACK);
		button.setForeground(Color.GREEN);
		button.setToolTipText("0.0");
	    button.setName(name); 
	    button.addItemListener(this);
		return button;
	}

	@Override
	public void actionPerformed(ActionEvent e) {

		if(e.getSource().equals(okButton)){
			setInsertMaterial(currButton,materialField.getText(),bufferedMaterial.isSelected()? Insert.BUFFERED_INSERT:Insert.UNBUFFERED_INSERT);
			gd.setVisible(false);
		}

	}


	private String getInsertMaterialName(JRadioButton but) {
		if(but==null){
			return "air";
		}
		String name = but.getName().trim();
		String ring = name.substring(0, name.indexOf("-"));
		int index = new Integer(name.substring(name.indexOf("-")+1));
		if(ring.equals("I")){
			return data.getInsertValue(CrisEDPhantomM062.INNER_RING, index);
		}else{
			return data.getInsertValue(CrisEDPhantomM062.OUTER_RING, index);
		}	
	}
	
	private int getInsertMaterialBufferState(JRadioButton but) {
		if(but==null){
			return Insert.UNBUFFERED_INSERT;
		}
		String name = but.getName().trim();
		String ring = name.substring(0, name.indexOf("-"));
		int index = new Integer(name.substring(name.indexOf("-")+1));
		if(ring.equals("I")){
			return data.getInsertBufferState(CrisEDPhantomM062.INNER_RING, index);
		}else{
			return data.getInsertBufferState(CrisEDPhantomM062.OUTER_RING, index);
		}	
	}


	private void setInsertMaterial(JRadioButton but, String material, int buffered) {
		if(but == null){
			return;
		}
		String name = but.getName().trim();
		String ring = name.substring(0, name.indexOf("-"));
		
		int index = new Integer(name.substring(name.indexOf("-")+1));
		Insert ins = new Insert(MaterialsDB.getMaterial(material), buffered);
		if(ring.equals("I")){
			data.setInsert(CrisEDPhantomM062.INNER_RING, index, ins);
		}else{
			data.setInsert(CrisEDPhantomM062.OUTER_RING, index, ins);
		}		
		but.setToolTipText(material+ "");
	}


	@Override
	public void itemStateChanged(ItemEvent e) {
		JRadioButton but = (JRadioButton) e.getSource();
		String name = but.getName().trim();
		if (name.equals("innerDiskActivator")) {
			if (but.isSelected()) {
				innerRing.setVisible(true);
				data.setRingState(CrisEDPhantomM062.INNER_RING, true);
			} else {
				innerRing.setVisible(false);
				data.setRingState(CrisEDPhantomM062.INNER_RING, false);
			}
		}else if(name.equals("outerDiskActivator")){
			if (but.isSelected()) {
				outerRing.setVisible(true);
				data.setRingState(CrisEDPhantomM062.OUTER_RING, true);
			} else {
				outerRing.setVisible(false);
				data.setRingState(CrisEDPhantomM062.OUTER_RING, false);
			}
		}else {			
			insertNameField.setText(but.getName());
			
			if (but.isSelected()) {					
				setInsertMaterial(currButton,materialField.getText(),bufferedMaterial.isSelected()? Insert.BUFFERED_INSERT:Insert.UNBUFFERED_INSERT);
				materialField.setText(getInsertMaterialName(but));
				bufferedMaterial.setSelected(getInsertMaterialBufferState(but)== Insert.BUFFERED_INSERT);
				currButton = but;
			} else {
				setInsertMaterial(currButton,materialField.getText(),bufferedMaterial.isSelected()? Insert.BUFFERED_INSERT:Insert.UNBUFFERED_INSERT);
				materialField.setText(getInsertMaterialName(currButton));
				bufferedMaterial.setSelected(getInsertMaterialBufferState(currButton)== Insert.BUFFERED_INSERT);
				setInsertMaterial(but, "air",Insert.UNBUFFERED_INSERT);
			}
	}
	
}


	





}

/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/