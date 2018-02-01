package edu.stanford.rsl.tutorial.mipda;

import java.awt.EventQueue;

import javax.swing.DefaultComboBoxModel;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import java.lang.InterruptedException;
import java.lang.reflect.InvocationTargetException;

import org.junit.internal.TextListener;
import org.junit.runner.JUnitCore;

import ModuleCourseIntroduction.IntroTestClass;
import ModuleMathematicalTools.SVDTestClass;
import ModulePreprocessing.GeoUTestClass;
import ModulePreprocessing.UMTestClass;
import ModuleReconstruction.PBTestClass;
import ModuleReconstruction.FBTestClass;
import ModuleRegistration.REGTestClass;
import ModuleRegistration.MITestClass;

/**
 * Testing tool for programming exercises
 * of the course "Medical Image Processing for Diagnostic Applications (MIPDA)"
 * @author Frank Schebesch
 *
 */

public class RunExerciseEvaluation {

	private static final String[] exerciseNames = {"Intro","SVD","GeoU","UM","PB","FB","REG","MI"}; 
	
	public static void main(String[] args) {
		
		JUnitCore junitCore = new JUnitCore();
		junitCore.addListener(new TextListener(System.out));
		
        String choice = ask(exerciseNames);
        System.out.println("Running test " + choice);
        
		if (choice.equals("Intro")) {
			junitCore.run(IntroTestClass.class);
			//org.junit.runner.JUnitCore.main("ModuleCourseIntroduction.IntroTestClass");
		}
		else if (choice.equals("SVD")) {
			junitCore.run(SVDTestClass.class);
		}
		else if (choice.equals("GeoU")) {
			junitCore.run(GeoUTestClass.class);
		}
		else if (choice.equals("UM")) {
			junitCore.run(UMTestClass.class);
		}
		else if (choice.equals("PB")) {
			junitCore.run(PBTestClass.class);
		}
		else if (choice.equals("FB")) {
			junitCore.run(FBTestClass.class);
		}
		else if (choice.equals("REG")) {
			junitCore.run(REGTestClass.class);
		}
		else if (choice.equals("MI")) {
			junitCore.run(MITestClass.class);
		}
		else {
			System.out.println("Aborted, no test selected.");
//			System.out.println("No such test available.");
		}
	}

	// thanks to https://stackoverflow.com/questions/13408238/simple-java-gui-as-a-popup-window-and-drop-down-menu
	// (adapted)
    public static String ask(final String... values) {

        String result = null;

        if (EventQueue.isDispatchThread()) {

            JPanel panel = new JPanel();
            panel.add(new JLabel("Select the exercise test that you wish to run:"));
            
            DefaultComboBoxModel<String> model = new DefaultComboBoxModel<>();
            for (String value : values) {
                model.addElement(value);
            }
            JComboBox<String> comboBox = new JComboBox<>(model);
            panel.add(comboBox);

            int iResult = JOptionPane.showConfirmDialog(null, panel, "jUnit Exercise Test Selection", JOptionPane.OK_CANCEL_OPTION, JOptionPane.QUESTION_MESSAGE);
            switch (iResult) {
                case JOptionPane.OK_OPTION:
                    result = (String) comboBox.getSelectedItem();
                    break;
                case JOptionPane.OK_CANCEL_OPTION:
                	return "";
            }

        } else {

            Response response = new Response(values);
            try {
                SwingUtilities.invokeAndWait(response);
                result = response.getResponse();
            } catch (InterruptedException | InvocationTargetException ex) {
                ex.printStackTrace();
            }

        }

        return result;

    }
    // thanks to https://stackoverflow.com/questions/13408238/simple-java-gui-as-a-popup-window-and-drop-down-menu
    public static class Response implements Runnable {

        private String[] values;
        private String response;

        public Response(String... values) {
            this.values = values;
        }

        @Override
        public void run() {
            response = ask(values);
        }

        public String getResponse() {
            return response;
        }
    }
}
