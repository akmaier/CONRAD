package edu.stanford.rsl.apps.gui.pointselector;

import ij.ImageJ;

import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.beans.XMLDecoder;
import java.beans.XMLEncoder;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import javax.swing.DefaultComboBoxModel;

import javax.swing.JButton;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.TableModel;

import javax.swing.JFrame;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.ListModel;

import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.FileUtil;
import edu.stanford.rsl.conrad.utils.XmlUtils;

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
public class PointSelector extends JFrame implements ActionListener, ListSelectionListener {
	/**
	 * 
	 */
	private static final long serialVersionUID = -5274899502815414214L;
	private JButton jButtonAdd;
	private JButton jButtonDelete;
	private JList<String> jList;
	private JTable jInfoTable;
	private JButton jButtonImport;
	private JButton jButtonExport;
	private JScrollPane jScrollPane;
	private JButton jButtonOpenInROImanager;

	private PointSelectorWorker psWorker;

	public PointSelector () {
		initGUI();
		psWorker = new PointSelectorWorker();
	}


	private void initGUI(){
		{
			GridBagLayout thisLayout = new GridBagLayout();
			thisLayout.rowWeights = new double[] {0.1, 0.1, 0.1, 0.1};
			thisLayout.rowHeights = new int[] {7, 7, 7, 7};
			thisLayout.columnWeights = new double[] {0.1, 0.1, 0.0, 0.1, 0.1};
			thisLayout.columnWidths = new int[] {20, 7, 69, 7, 7};
			getContentPane().setLayout(thisLayout);
			{
				jButtonAdd = new JButton();
				getContentPane().add(jButtonAdd, new GridBagConstraints(0, 3, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
				jButtonAdd.setText("Add");
				jButtonAdd.addActionListener(this);
			}
			{
				jButtonExport = new JButton();
				getContentPane().add(jButtonExport, new GridBagConstraints(4, 3, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
				jButtonExport.setText("ExportToXML");
				jButtonExport.addActionListener(this);
			}
			{
				jButtonOpenInROImanager = new JButton();
				getContentPane().add(jButtonOpenInROImanager, new GridBagConstraints(4, 2, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
				jButtonOpenInROImanager.setText("OpenInROImanager");
				jButtonOpenInROImanager.addActionListener(this);
				jButtonOpenInROImanager.setVisible(false);
			}
			{
				jButtonImport = new JButton();
				getContentPane().add(jButtonImport, new GridBagConstraints(3, 3, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
				jButtonImport.setText("ImportFromXML");
				jButtonImport.addActionListener(this);
			}
			{
				jButtonDelete = new JButton();
				getContentPane().add(jButtonDelete, new GridBagConstraints(1, 3, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
				jButtonDelete.setText("Delete");
				jButtonDelete.addActionListener(this);
			}
			{
				TableModel jInfoTableModel = 
						new DefaultTableModel();
				jInfoTable = new JTable();
				jScrollPane = new JScrollPane(jInfoTable);
				getContentPane().add(jScrollPane, new GridBagConstraints(3, 0, 2, 3, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
				jInfoTable.setModel(jInfoTableModel);
			}
			{
				ListModel<String> jListModel = 
						new DefaultComboBoxModel<String>();
				jList = new JList<String>();
				getContentPane().add(jList, new GridBagConstraints(0, 0, 2, 3, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
				jList.setModel(jListModel);
				jList.addListSelectionListener(this);
			}
		}
		{
			this.setSize(681, 320);
		}
		// TODO Auto-generated method stub


	}


	@Override
	public void actionPerformed(ActionEvent arg0) {
		if (arg0.getSource().equals(jButtonExport)){
			
			Thread test = new Thread(){
				@Override
				public void run() {
					try {
						if (psWorker.getNumberOfPointSets() > 0){
							XmlUtils.exportToXML(psWorker.getAllPointSets());
						}
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
			};
			test.start();
		}

		if (arg0.getSource().equals(jButtonImport)){

			Thread test = new Thread(){
				@Override
				public void run() {
					try {
						psWorker.setAllPointSets(XmlUtils.importFromXML());
						updateListStrings();
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
			};
			test.start();
		}

		if (arg0.getSource().equals(jButtonAdd)){

			Thread thrd = new Thread(){
				@Override
				public void run() {
					if (psWorker.evaluate()) {
						int nrSets = psWorker.getNumberOfPointSets();
						((DefaultComboBoxModel<String>)jList.getModel()).addElement("PointSet_" + nrSets + "_" + 
								psWorker.getPointSet(nrSets-1).size() + "_Points");
					}
				}
			};
			thrd.start();
		}

		
		if (arg0.getSource().equals(jButtonOpenInROImanager)){

			Thread thrd = new Thread(){
				@Override
				public void run() {
					if (!jList.isSelectionEmpty()){
						psWorker.setRoiManagerPointSet(jList.getSelectedIndex());
					}
				}
			};
			thrd.start();
		}
		
		
		if (arg0.getSource().equals(jButtonDelete)){
			int res;
			if (jList.isSelectionEmpty()){
				res = JOptionPane.showConfirmDialog(this, "Clear ALL datasets?", "Delete datasets", JOptionPane.OK_CANCEL_OPTION, JOptionPane.QUESTION_MESSAGE);
				if (res == 0){
					psWorker.removeAllPointSets();
				}
			}
			else
			{
				res = JOptionPane.showConfirmDialog(this, "Clear selected datasets?", "Delete datasets", JOptionPane.OK_CANCEL_OPTION, JOptionPane.QUESTION_MESSAGE);
				if (res == 0){
					for (int i = 0; i < jList.getSelectedIndices().length; ++i)
						psWorker.removePointSet(jList.getSelectedIndices()[jList.getSelectedIndices().length-i-1]);

				}
			}
			if (res==0) {
				jInfoTable.setModel(new DefaultTableModel());
				jButtonOpenInROImanager.setVisible(false);
			}
			updateListStrings();
		}

	}


	@Override
	public void valueChanged(ListSelectionEvent arg0) {
		if (arg0.getSource().equals(jList)){

			Thread test = new Thread(){
				@Override
				public void run() {
					if(!jList.isSelectionEmpty()){
						int idx = jList.getSelectedIndex();
						Set<Integer> sortSet = new HashSet<Integer>();
						Iterator<double[]> it = psWorker.getPointSet(idx).iterator();
						while (it.hasNext()){
							double[] p = it.next();
							sortSet.add((int)p[p.length-1]);
						}

						jInfoTable.setModel(new DefaultTableModel(
								new String[][] { { Integer.toString(idx), 
									Integer.toString(psWorker.getPointSet(idx).size()),
									Integer.toString(sortSet.size()) } },
									new String[] { "Nr.", "#Points", "#Slices" }));
						jInfoTable.setPreferredScrollableViewportSize(jInfoTable.getPreferredSize());
						//jScrollPane.setSize(jInfoTable.getPreferredSize());
						jButtonOpenInROImanager.setVisible(true);
					}	
				}
			};
			test.start();
		}

	}




	private void updateListStrings(){
		((DefaultComboBoxModel<String>)jList.getModel()).removeAllElements();
		int nrSets = psWorker.getNumberOfPointSets();
		if (nrSets > 0){
			for (int i=0; i < nrSets; ++i)
				((DefaultComboBoxModel<String>)jList.getModel()).addElement("PointSet_" + i + "_" + 
						psWorker.getPointSet(i).size() + "_Points");
		}
	}


	public void startMarkerDetectionTool() {
		
	}
	
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		CONRAD.setup();
		PointSelector ps = new PointSelector();
		ps.setVisible(true);
	}

}

/*
 * Copyright (C) 2010-2014 - Martin Berger 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/