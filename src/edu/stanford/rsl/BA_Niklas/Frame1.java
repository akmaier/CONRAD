package edu.stanford.rsl.BA_Niklas;

import java.awt.EventQueue;

import javax.swing.JFrame;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JTextField;
import javax.swing.JLabel;
import javax.swing.JOptionPane;

import java.awt.event.ActionListener;
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
		lblStart.setBounds(288, 83, 23, 14);
		frame.getContentPane().add(lblStart);
		
		lblEnd = new JLabel("end");
		lblEnd.setBounds(340, 83, 23, 14);
		frame.getContentPane().add(lblEnd);
		
		xstart = new JTextField();
		xstart.setText("25");
		xstart.setBounds(288, 108, 36, 20);
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
		xend.setBounds(340, 108, 36, 20);
		frame.getContentPane().add(xend);
		xend.setColumns(10);
		
		JLabel lblXaxes = new JLabel("x-axes");
		lblXaxes.setBounds(242, 111, 46, 14);
		frame.getContentPane().add(lblXaxes);
		
		JLabel lblYaxes = new JLabel("y-axes");
		lblYaxes.setBounds(242, 139, 46, 14);
		frame.getContentPane().add(lblYaxes);
		
		ystart = new JTextField();
		ystart.setText("125");
		ystart.setBounds(288, 136, 36, 20);
		frame.getContentPane().add(ystart);
		ystart.setColumns(10);
		
		yend = new JTextField();
		yend.setText("150");
		yend.setBounds(340, 136, 36, 20);
		frame.getContentPane().add(yend);
		yend.setColumns(10);
		
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
				
				String[] args = {Boolean.toString(simCheckbox.isSelected()), nr_ellipses.getText(), Boolean.toString(trcCheckbox.isSelected()), xstart.getText(), xend.getText(), ystart.getText(), yend.getText(), value.getText(), Boolean.toString(iterCheckbox.isSelected()), iter_num.getText(), error_val.getText(), Boolean.toString(visCheckbox.isSelected())}; 
				Bubeck_Niklas_BA.main(args);
			}
		});
		btnCompute.setBounds(340, 571, 89, 23);
		frame.getContentPane().add(btnCompute);
		

		
		
		
		
		
	}
}
