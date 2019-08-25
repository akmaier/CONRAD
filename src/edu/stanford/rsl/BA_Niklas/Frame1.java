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
	private JCheckBox iterCheckbox;
	private JTextField iter_num;
	private JLabel lblIternum;
	private JTextField error_val;
	private JLabel lblErrorval;
	private JCheckBox visCheckbox;
	private JTextField xstart2;
	private JTextField ystart2;
	private JTextField xend2;
	private JTextField yend2;

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
		nr_ellipses.setText("5");
		nr_ellipses.setBounds(288, 21, 36, 20);
		frame.getContentPane().add(nr_ellipses);
		nr_ellipses.setColumns(10);
		
		lblNrellipses = new JLabel("nr_ellipses");
		lblNrellipses.setBounds(232, 24, 46, 14);
		frame.getContentPane().add(lblNrellipses);
		
		lblStart = new JLabel("start");
		lblStart.setBounds(245, 83, 23, 14);
		frame.getContentPane().add(lblStart);
		
		lblEnd = new JLabel("end");
		lblEnd.setBounds(288, 83, 23, 14);
		frame.getContentPane().add(lblEnd);
		
		xstart = new JTextField();
		xstart.setText("0");
		xstart.setBounds(245, 108, 36, 20);
		frame.getContentPane().add(xstart);
		xstart.setColumns(10);
		
		JCheckBox simCheckbox = new JCheckBox("simulate data");
		simCheckbox.setBounds(47, 20, 97, 23);
		frame.getContentPane().add(simCheckbox);
		simCheckbox.setSelected(true);
		
		JCheckBox trcCheckbox = new JCheckBox("truncate data");
		trcCheckbox.setBounds(47, 107, 97, 23);
		frame.getContentPane().add(trcCheckbox);
		
		xend = new JTextField();
		xend.setText("50");
		xend.setBounds(288, 108, 36, 20);
		frame.getContentPane().add(xend);
		xend.setColumns(10);
		
		JLabel lblXaxes = new JLabel("x-axes");
		lblXaxes.setBounds(189, 111, 46, 14);
		frame.getContentPane().add(lblXaxes);
		
		JLabel lblYaxes = new JLabel("y-axes");
		lblYaxes.setBounds(189, 136, 46, 14);
		frame.getContentPane().add(lblYaxes);
		
		ystart = new JTextField();
		ystart.setText("0");
		ystart.setBounds(245, 133, 36, 20);
		frame.getContentPane().add(ystart);
		ystart.setColumns(10);
		
		yend = new JTextField();
		yend.setText("90");
		yend.setBounds(288, 133, 36, 20);
		frame.getContentPane().add(yend);
		yend.setColumns(10);
		
		xstart2 = new JTextField();
		xstart2.setText("150");
		xstart2.setColumns(10);
		xstart2.setBounds(344, 108, 36, 20);
		frame.getContentPane().add(xstart2);
		
		ystart2 = new JTextField();
		ystart2.setText("270");
		ystart2.setColumns(10);
		ystart2.setBounds(344, 133, 36, 20);
		frame.getContentPane().add(ystart2);
		
		xend2 = new JTextField();
		xend2.setText("200");
		xend2.setColumns(10);
		xend2.setBounds(390, 108, 36, 20);
		frame.getContentPane().add(xend2);
		
		yend2 = new JTextField();
		yend2.setText("360");
		yend2.setColumns(10);
		yend2.setBounds(390, 133, 36, 20);
		frame.getContentPane().add(yend2);
		
		JLabel label = new JLabel("start");
		label.setBounds(340, 83, 23, 14);
		frame.getContentPane().add(label);
		
		JLabel label_1 = new JLabel("end");
		label_1.setBounds(387, 83, 23, 14);
		frame.getContentPane().add(label_1);
		
		JCheckBox darkCheckbox = new JCheckBox("Only Dark");
		darkCheckbox.setBounds(70, 132, 97, 23);
		frame.getContentPane().add(darkCheckbox);
		
		lblValue = new JLabel("value");
		lblValue.setBounds(242, 164, 46, 14);
		frame.getContentPane().add(lblValue);
		
		value = new JTextField();
		value.setText("0");
		value.setBounds(288, 164, 23, 20);
		frame.getContentPane().add(value);
		value.setColumns(10);
		
		iterCheckbox = new JCheckBox("iter_reconstruction");
		iterCheckbox.setBounds(47, 196, 97, 23);
		frame.getContentPane().add(iterCheckbox);
		
		iter_num = new JTextField();
		iter_num.setText("50");
		iter_num.setBounds(288, 197, 23, 20);
		frame.getContentPane().add(iter_num);
		iter_num.setColumns(10);
		
		lblIternum = new JLabel("iter_num");
		lblIternum.setBounds(232, 200, 46, 14);
		frame.getContentPane().add(lblIternum);
		
		error_val = new JTextField();
		error_val.setText("0.5");
		error_val.setBounds(288, 234, 23, 20);
		frame.getContentPane().add(error_val);
		error_val.setColumns(10);
		
		lblErrorval = new JLabel("error_val");
		lblErrorval.setBounds(232, 237, 46, 14);
		frame.getContentPane().add(lblErrorval);
		
		visCheckbox = new JCheckBox("show visualizations");
		visCheckbox.setBounds(47, 285, 97, 23);
		frame.getContentPane().add(visCheckbox);
		
		
		JButton btnCompute = new JButton("Compute");
		btnCompute.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
				// perfom computing action

				String[] args = {Boolean.toString(simCheckbox.isSelected()), nr_ellipses.getText(), Boolean.toString(trcCheckbox.isSelected()), xstart.getText(), xend.getText(), xstart2.getText(), xend2.getText(), ystart.getText(), yend.getText(), ystart2.getText(), yend2.getText(),Boolean.toString(darkCheckbox.isSelected()), value.getText(), Boolean.toString(iterCheckbox.isSelected()), iter_num.getText(), error_val.getText(), Boolean.toString(visCheckbox.isSelected())}; 
				try {
					Bubeck_Niklas_BA.main(args);
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
