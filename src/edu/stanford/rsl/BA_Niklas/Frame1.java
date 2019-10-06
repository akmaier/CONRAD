package edu.stanford.rsl.BA_Niklas;

import java.awt.EventQueue;

import javax.swing.JFrame;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JTextField;
import javax.swing.JLabel;
import javax.swing.JOptionPane;

import java.awt.event.ActionListener;
import java.io.IOException;
import java.awt.event.ActionEvent;

import edu.stanford.rsl.BA_Niklas.Bubeck_Niklas_BA;
import java.awt.Checkbox;



public class Frame1 {

	private JFrame frame;
	private JCheckBox simCheckbox;
	private JCheckBox trcCheckbox;
	private Checkbox checkbox;
	
	private JTextField xend;
	private JTextField ystart;
	private JTextField yend;
	private JTextField nr_ellipses;
	private JLabel lblNrellipses;
	private JLabel lblStart;
	private JLabel lblEnd;
	private JTextField xstart;
	private JLabel lblValue;
	private JTextField value;
	private JTextField iter_num;
	private JLabel lblIternum;
	private JTextField error_val;
	private JLabel lblErrorval;
	private JCheckBox visCheckbox;
	private JTextField xstart2;
	private JTextField ystart2;
	private JTextField xend2;
	private JTextField yend2;
	private JTextField noisetype;
	private JTextField path;

	/**
	 * Launch the application.
	 */
	public static void main(String[] args) {
		EventQueue.invokeLater(new Runnable() {
			public void run() {
				try {
					Frame1 window = new Frame1();
					window.frame.setVisible(true);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
	}

	/**
	 * Create the application.
	 */
	public Frame1() {
		initialize();
	}

	/**
	 * Initialize the contents of the frame.
	 */
	private void initialize() {
		frame = new JFrame();
		frame.setBounds(100, 100, 451, 644);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.getContentPane().setLayout(null);
		
		nr_ellipses = new JTextField();
		nr_ellipses.setText("2");
		nr_ellipses.setBounds(288, 21, 36, 20);
		frame.getContentPane().add(nr_ellipses);
		nr_ellipses.setColumns(10);
		
		lblNrellipses = new JLabel("nr_ellipses");
		lblNrellipses.setBounds(232, 24, 46, 14);
		frame.getContentPane().add(lblNrellipses);
		
		lblStart = new JLabel("start");
		lblStart.setBounds(242, 130, 23, 14);
		frame.getContentPane().add(lblStart);
		
		lblEnd = new JLabel("end");
		lblEnd.setBounds(288, 130, 23, 14);
		frame.getContentPane().add(lblEnd);
		
		xstart = new JTextField();
		xstart.setText("0");
		xstart.setBounds(242, 155, 36, 20);
		frame.getContentPane().add(xstart);
		xstart.setColumns(10);
		
		JCheckBox simCheckbox = new JCheckBox("simulate data");
		simCheckbox.setBounds(47, 20, 97, 23);
		frame.getContentPane().add(simCheckbox);
		simCheckbox.setSelected(true);
		
		JCheckBox trcCheckbox = new JCheckBox("truncate data");
		trcCheckbox.setSelected(true);
		trcCheckbox.setBounds(47, 154, 97, 23);
		frame.getContentPane().add(trcCheckbox);
		
		xend = new JTextField();
		xend.setText("75");
		xend.setBounds(288, 155, 36, 20);
		frame.getContentPane().add(xend);
		xend.setColumns(10);
		
		JLabel lblXaxes = new JLabel("x-axes");
		lblXaxes.setBounds(189, 158, 46, 14);
		frame.getContentPane().add(lblXaxes);
		
		JLabel lblYaxes = new JLabel("y-axes");
		lblYaxes.setBounds(189, 184, 46, 14);
		frame.getContentPane().add(lblYaxes);
		
		ystart = new JTextField();
		ystart.setText("0");
		ystart.setBounds(242, 181, 36, 20);
		frame.getContentPane().add(ystart);
		ystart.setColumns(10);
		
		yend = new JTextField();
		yend.setText("90");
		yend.setBounds(288, 181, 36, 20);
		frame.getContentPane().add(yend);
		yend.setColumns(10);
		
		xstart2 = new JTextField();
		xstart2.setText("125");
		xstart2.setColumns(10);
		xstart2.setBounds(344, 155, 36, 20);
		frame.getContentPane().add(xstart2);
		
		ystart2 = new JTextField();
		ystart2.setText("270");
		ystart2.setColumns(10);
		ystart2.setBounds(344, 181, 36, 20);
		frame.getContentPane().add(ystart2);
		
		xend2 = new JTextField();
		xend2.setText("200");
		xend2.setColumns(10);
		xend2.setBounds(390, 155, 36, 20);
		frame.getContentPane().add(xend2);
		
		yend2 = new JTextField();
		yend2.setText("360");
		yend2.setColumns(10);
		yend2.setBounds(390, 181, 36, 20);
		frame.getContentPane().add(yend2);
		
		JLabel label = new JLabel("start");
		label.setBounds(343, 130, 23, 14);
		frame.getContentPane().add(label);
		
		JLabel label_1 = new JLabel("end");
		label_1.setBounds(390, 130, 23, 14);
		frame.getContentPane().add(label_1);
		
		JCheckBox darkCheckbox = new JCheckBox("Only Dark");
		darkCheckbox.setSelected(true);
		darkCheckbox.setBounds(70, 180, 97, 23);
		frame.getContentPane().add(darkCheckbox);
		
		lblValue = new JLabel("value");
		lblValue.setBounds(232, 230, 46, 14);
		frame.getContentPane().add(lblValue);
		
		value = new JTextField();
		value.setText("0");
		value.setBounds(288, 227, 23, 20);
		frame.getContentPane().add(value);
		value.setColumns(10);
		
		iter_num = new JTextField();
		iter_num.setText("5");
		iter_num.setBounds(245, 343, 23, 20);
		frame.getContentPane().add(iter_num);
		iter_num.setColumns(10);
		
		lblIternum = new JLabel("iter_num");
		lblIternum.setBounds(189, 346, 46, 14);
		frame.getContentPane().add(lblIternum);
		
		error_val = new JTextField();
		error_val.setText("0");
		error_val.setBounds(245, 368, 23, 20);
		frame.getContentPane().add(error_val);
		error_val.setColumns(10);
		
		lblErrorval = new JLabel("error_val");
		lblErrorval.setBounds(189, 371, 46, 14);
		frame.getContentPane().add(lblErrorval);
		
		visCheckbox = new JCheckBox("show visualizations");
		visCheckbox.setBounds(47, 464, 97, 23);
		frame.getContentPane().add(visCheckbox);

		
		JCheckBox noisechecked = new JCheckBox("add noise");
		noisechecked.setBounds(70, 46, 97, 23);
		frame.getContentPane().add(noisechecked);
		
		noisetype = new JTextField();
		noisetype.setText("gaussian");
		noisetype.setBounds(173, 47, 86, 20);
		frame.getContentPane().add(noisetype);
		noisetype.setColumns(10);
		
		JCheckBox singleMaterialCheck = new JCheckBox("Single material");
		singleMaterialCheck.setBounds(93, 316, 97, 23);
		frame.getContentPane().add(singleMaterialCheck);
		
		JCheckBox multiMaterialCheck = new JCheckBox("Multiple material");
		multiMaterialCheck.setBounds(268, 316, 112, 23);
		frame.getContentPane().add(multiMaterialCheck);
		
		path = new JTextField();
		path.setText("C:\\Users\\Niklas\\Documents\\Uni\\Bachelorarbeit\\Bilder\\BilderTestFilled");
		path.setBounds(219, 491, 181, 20);
		frame.getContentPane().add(path);
		path.setColumns(10);
		
		JCheckBox chckbxSaveImages = new JCheckBox("save images ");
		chckbxSaveImages.setBounds(47, 490, 97, 23);
		frame.getContentPane().add(chckbxSaveImages);
		
		JLabel lblTo = new JLabel("to");
		lblTo.setBounds(173, 494, 16, 14);
		frame.getContentPane().add(lblTo);
		
		JButton btnCompute = new JButton("Compute");
		btnCompute.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
				// perfom computing action

				String[] args = {Boolean.toString(simCheckbox.isSelected()),	// 0
						nr_ellipses.getText(),									// 1
						Boolean.toString(trcCheckbox.isSelected()),				// 2
						xstart.getText(),										// 3
						xend.getText(),											// 4
						xstart2.getText(),										// 5
						xend2.getText(),										// 6
						ystart.getText(),										// 7
						yend.getText(),											// 8
						ystart2.getText(),										// 9
						yend2.getText(),										// 10
						Boolean.toString(darkCheckbox.isSelected()),			// 11
						value.getText(),										// 12
//						Boolean.toString(iterCheckbox.isSelected()),			// 13
						iter_num.getText(),										// 14
						error_val.getText(),									// 15
						Boolean.toString(visCheckbox.isSelected()),				// 16
						Boolean.toString(noisechecked.isSelected()),			// 17
						noisetype.getText(),									//18
						Boolean.toString(singleMaterialCheck.isSelected()),
						Boolean.toString(multiMaterialCheck.isSelected()),
						path.getText(),
						Boolean.toString(chckbxSaveImages.isSelected())
				}; 
				try {
					Bubeck_Niklas_BA.all(args);
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		});
	
				
		btnCompute.setBounds(340, 571, 89, 23);
		frame.getContentPane().add(btnCompute);
		
		
		

		
		
		
		
		
	}
}
