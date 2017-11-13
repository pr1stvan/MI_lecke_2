package mi_lecke_2;

import java.util.Scanner;

public class NNSolutionThree {
	public static void main(String[] args){
		Scanner reader = new Scanner(System.in);
		nnSolutionThree(reader);
		reader.close();
	}
	
	public static void nnSolutionThree(Scanner reader){
		int[] inputArchitecture=MI2_IO.readArchitecture(reader);
		
		NeuralNetwork myNetwork=new NeuralNetwork(inputArchitecture);
		myNetwork.initializeWeightsFromConsole(reader);
		double[][] inputs=MI2_IO.readInputs(reader, inputArchitecture[0]);
		myNetwork.setInput(inputs[0]);
		MI2_IO.writeArchitecture(myNetwork.getArchitecture());
		
		myNetwork.calculateDeltaYperDeltaWij();
		
		myNetwork.printDeltaYperDeltaWij();
	}
}
