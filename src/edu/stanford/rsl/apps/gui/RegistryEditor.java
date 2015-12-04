/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.apps.gui;

import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridBagConstraints;
import java.awt.Rectangle;
import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.UnsupportedFlavorException;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.font.FontRenderContext;
import java.awt.font.LineBreakMeasurer;
import java.awt.font.TextLayout;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Field;
import java.text.AttributedCharacterIterator;
import java.text.AttributedString;
import java.text.BreakIterator;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import javax.swing.DropMode;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.ListSelectionModel;
import javax.swing.TransferHandler;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.TableCellRenderer;

import com.thoughtworks.qdox.JavaDocBuilder;
import com.thoughtworks.qdox.model.JavaClass;
import com.thoughtworks.qdox.model.JavaField;

import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.RegKeys;
import edu.stanford.rsl.conrad.utils.UserUtil;


public class RegistryEditor extends JPanel implements ActionListener {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8853718009928894577L;
	private Configuration config;
	private JButton add;
	private JButton delete;
	private JButton showKeys;
	private JTable table;

	public RegistryEditor(Configuration config){
		super();
		this.config = config;
	}

	public void initGUI(){
		this.setLayout(ConfigurationFrame.createSubPaneLayout());
		this.setBackground(Color.WHITE);
		Set<String> keys = config.getRegistryKeys();

		String [][] tableData = new String [keys.size()][2];
		String [] label = {"Key", "Value"};

		Iterator<String> iter = keys.iterator();
		int i = 0;
		while (iter.hasNext()){
			tableData[i][0] = iter.next();
			tableData[i][1] = config.getRegistryEntry(tableData[i][0]);
			i++;
		}

		table = new JTable();
		DefaultTableModel model = new DefaultTableModel(tableData, label);
		table.setModel(model);
		table.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
		table.setFillsViewportHeight(true);
		table.setDropMode(DropMode.ON);
		table.setPreferredScrollableViewportSize(new Dimension(495, 270));


		table.setTransferHandler(new TransferHandler() {

			/**
			 * 
			 */
			private static final long serialVersionUID = 6224911732660417665L;

			public boolean canImport(TransferSupport support) {
				// for the demo, we'll only support drops (not clipboard paste)
				if (!support.isDrop()) {
					return false;
				}

				// we only import Strings and files
				if (!(support.isDataFlavorSupported(DataFlavor.stringFlavor)||support.isDataFlavorSupported(DataFlavor.javaFileListFlavor))) {
					return false;
				}

				return true;
			}

			@SuppressWarnings("unchecked")
			public boolean importData(TransferSupport support) {
				// if we can't handle the import, say so
				if (!canImport(support)) {
					return false;
				}

				// fetch the drop location
				JTable.DropLocation dl = (JTable.DropLocation) support
				.getDropLocation();

				int row = dl.getRow();

				// fetch the data and bail if this fails
				String data;
				try {
					data = (String) support.getTransferable().getTransferData(
							DataFlavor.stringFlavor);
				} catch (UnsupportedFlavorException e) {
					try {
						java.util.List<File> l =
							(java.util.List<File>) support.getTransferable().getTransferData(DataFlavor.javaFileListFlavor);
						data = l.get(0).getAbsolutePath();

					} catch (UnsupportedFlavorException e1) {
						// TODO Auto-generated catch block
						e1.printStackTrace();
						return false;
					} catch (IOException e1) {
						// TODO Auto-generated catch block
						e1.printStackTrace();
						return false;
					}			
				} catch (IOException e) {
					return false;
				}
				DefaultTableModel model = (DefaultTableModel) table.getModel();
				model.setValueAt(data, row, 1);

				Rectangle rect = table.getCellRect(row, 0, false);
				if (rect != null) {
					table.scrollRectToVisible(rect);
				}

				return true;
			}
		});

		JScrollPane scroll = new JScrollPane(table);

		add(scroll, ConfigurationFrame.createConstraints(1, 1, 3, 2, GridBagConstraints.NORTH, GridBagConstraints.BOTH, 0, 0, 0, 0));

		add = new JButton("Add key");
		add.addActionListener(this);
		delete = new JButton("Remove key");
		delete.addActionListener(this);
		showKeys = new JButton("Show available keys");
		showKeys.addActionListener(this);
		add(add, ConfigurationFrame.createConstraints(1, 0, 1, 1, GridBagConstraints.WEST, GridBagConstraints.NONE, 5, 0, 5, 0));
		add(showKeys, ConfigurationFrame.createConstraints(2, 0, 1, 1, GridBagConstraints.CENTER, GridBagConstraints.NONE, 5, 0, 5, 0));
		add(delete, ConfigurationFrame.createConstraints(3, 0, 1, 1, GridBagConstraints.EAST, GridBagConstraints.NONE, 5, 0, 5, 0));
		setOpaque(true);
	}


	public static void main(String [] args){
		Configuration.loadConfiguration();
		JFrame frame =  new JFrame();
		frame.add(new RegistryEditor(Configuration.getGlobalConfiguration()));
		frame.setVisible(true);
		frame.setSize(new Dimension(495, 270));
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

	JFrame keysDescription;
	JTable descTable;

	public class MultilineTableCell 
	implements TableCellRenderer {
		class CellArea extends DefaultTableCellRenderer {
			/**
			 * 
			 */
			private static final long serialVersionUID = 1L;
			private String text;
			protected int rowIndex;
			protected int columnIndex;
			protected JTable table;
			protected Font font;
			private int paragraphStart,paragraphEnd;
			private LineBreakMeasurer lineMeasurer;

			public CellArea(String s, JTable tab, int row, int column,boolean isSelected) {
				text = s;
				rowIndex = row;
				columnIndex = column;
				table = tab;
				font = table.getFont();
				if (isSelected) {
					setForeground(table.getSelectionForeground());
					setBackground(table.getSelectionBackground());
				}
			}
			
			public void paintComponent(Graphics gr) {
				super.paintComponent(gr);
				if ( text != null && !text.isEmpty() ) {
					Graphics2D g = (Graphics2D) gr;
					if (lineMeasurer == null) {
						AttributedCharacterIterator paragraph = new AttributedString(text).getIterator();
						paragraphStart = paragraph.getBeginIndex();
						paragraphEnd = paragraph.getEndIndex();
						FontRenderContext frc = g.getFontRenderContext();
						lineMeasurer = new LineBreakMeasurer(paragraph,BreakIterator.getWordInstance(), frc);
					}
					float breakWidth = (float)table.getColumnModel().getColumn(columnIndex).getWidth();
					float drawPosY = 0;
					// Set position to the index of the first character in the paragraph.
					lineMeasurer.setPosition(paragraphStart);
					// Get lines until the entire paragraph has been displayed.
					while (lineMeasurer.getPosition() < paragraphEnd) {
						// Retrieve next layout. A cleverer program would also cache
						// these layouts until the component is re-sized.
						TextLayout layout = lineMeasurer.nextLayout(breakWidth);
						// Compute pen x position. If the paragraph is right-to-left we
						// will align the TextLayouts to the right edge of the panel.
						// Note: this won't occur for the English text in this sample.
						// Note: drawPosX is always where the LEFT of the text is placed.
						float drawPosX = layout.isLeftToRight()
						? 0 : breakWidth - layout.getAdvance();
						// Move y-coordinate by the ascent of the layout.
						drawPosY += layout.getAscent();
						// Draw the TextLayout at (drawPosX, drawPosY).
						layout.draw(g, drawPosX, drawPosY);
						// Move y-coordinate in preparation for next layout.
						drawPosY += layout.getDescent() + layout.getLeading();
					}
					table.setRowHeight(rowIndex,(int) drawPosY);
				}
			}
		}
		public Component getTableCellRendererComponent(
				JTable table, Object value,boolean isSelected, boolean hasFocus, int row,int column
		)
		{
			CellArea area = new CellArea(value.toString(),table,row,column,isSelected);
			return area;
		}   
	}

	/**
	 * Example for getting comments from a java file using qdox.
	 * This requires access to the actual java file in the list of ressources.
	 * (Use source folder as output folder for class files in eclipse)
	 */
	private void generateKeyTableFromSources(){
		// read java sources to parser from qdox
		InputStream stream = RegKeys.class.getResourceAsStream("RegKeys.java");
		InputStreamReader reader = new InputStreamReader(stream);
		JavaDocBuilder builder = new JavaDocBuilder();
		builder.addSource(reader);
		
		// generate table with line wrapping cells in colum 2
		final int wordWrapColumnIndex = 1;
		descTable = new JTable() {    
			/**
			 * 
			 */
			private static final long serialVersionUID = 7516482246121817003L;

			public TableCellRenderer getCellRenderer(int row, int column) {
				if (column == wordWrapColumnIndex ) {
					return new MultilineTableCell();
				}
				else {
					return super.getCellRenderer(row, column);
				}
			}
		};
		
		// Use qdox parser to parse RegKeys sources
		JavaClass cls = builder.getClassByName("edu.stanford.rsl.conrad.utils.RegKeys");
		// get list of fields
		JavaField[] fields = cls.getFields();

		// build table model
		String [][] tableData = new String [fields.length][2];
		String [] label = {"Key", "Description"};
		int i = 0;
		for (JavaField field : fields){
			tableData[i][0] = field.getName();
			tableData[i][1] = field.getComment();
			if (tableData[i][1] != null){
				tableData[i][1] = field.getComment().replace("<BR>", "\n")	
				.replace("<br>", "\n")
				.replace("<b>", "\n")
				.replace("</b>", "\n");
			} else {
				tableData[i][1] = "";
			}
			i++;
		}

		DefaultTableModel model = new DefaultTableModel(tableData, label);
		descTable.setModel(model);
	}
	
	private void generateKeyTableFromClassFiles(){
		// get fields from class file
		Field[] declaredFields = RegKeys.class.getDeclaredFields();
		List<Field> staticFields = new ArrayList<Field>();
		// add only static fields
		for (Field field : declaredFields) {
			if (java.lang.reflect.Modifier.isStatic(field.getModifiers())) {
				staticFields.add(field);
			}
		}
		descTable = new JTable();
		// build table model
		String [][] tableData = new String [staticFields.size()][2];
		String [] label = {"Key", "Description"};
		int i = 0;
		for (Field field : staticFields){
			tableData[i][0] = field.getName();
			tableData[i][1] = "Comment unavailable. (Use source folder as output folder for class files in eclipse or include sources in jar file.)";
			i++;
		}
		DefaultTableModel model = new DefaultTableModel(tableData, label);
		descTable.setModel(model);
	}
	
	public void actionPerformed(ActionEvent e) {
		if (e.getSource().equals(add)){
			try {
				String key = UserUtil.queryString("Enter key name:", "");
				DefaultTableModel model = (DefaultTableModel) table.getModel();
				boolean found = false;
				for (int i = 0; i < model.getRowCount(); i++){
					if (model.getValueAt(i, 0).equals(key)) found = true;
				}
				if (! found) {
					model.addRow(new String [] {key, ""});
					model.fireTableDataChanged();
				}
			} catch (Exception e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}	
		}
		if (e.getSource().equals(showKeys)){
			try {
				if (keysDescription ==null){
					keysDescription = new JFrame("Available Keys");
					keysDescription.setSize(640,480);
					keysDescription.setBackground(Color.WHITE);
					
					try{
						// Read key descriptions from source file
						// getting comments requires access to the actual java files in the 
						// resource path.
						generateKeyTableFromSources();
					} catch (Exception e2){
						// Generate key list using introspection only
						// will be executed if source is not found.
						generateKeyTableFromClassFiles();
					}
					descTable.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
					descTable.setFillsViewportHeight(true);
					descTable.setPreferredScrollableViewportSize(new Dimension(640, 480));
					keysDescription.add(new JScrollPane(descTable));
					keysDescription.setVisible(true);
					keysDescription.setLocation(CONRAD.getWindowTopCorner());
					System.out.println("created");
				} else {
					if( keysDescription.isVisible()){
						keysDescription.setVisible(false);
						keysDescription = null;
					} else {
						keysDescription.setVisible(true);
					}
				}
			} catch (Exception e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}	
		}
		if (e.getSource().equals(delete)){
			if (table.getSelectedRowCount() == 1){
				DefaultTableModel model = (DefaultTableModel) table.getModel();
				int selectedRow = table.getSelectedRow();
				model.removeRow(selectedRow);
			} else {
				// won't delete multiple keys at once.
				table.getSelectionModel().clearSelection();
			}
		}

	}

	public void updateToConfiguration(){
		config.resetRegistry();
		DefaultTableModel model = (DefaultTableModel) table.getModel();
		int lenght = model.getRowCount();
		for (int i = 0; i < lenght; i++){
			config.setRegistryEntry((String) model.getValueAt(i, 0), (String) model.getValueAt(i, 1));
		}
	}

}


