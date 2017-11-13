package mi_lecke_2;

import java.util.Scanner;

public class NNSolutionTwo {
	public static void main(String[] args){
		Scanner reader = new Scanner(System.in);
		nnSolutionTwo(reader);
		reader.close();
	}
	
	public static void nnSolutionTwo(Scanner reader){
		int[] inputArchitecture=MI2_IO.readArchitecture(reader);

	
		NeuralNetwork myNetwork=new NeuralNetwork(inputArchitecture);	
		myNetwork.initializeWeightsFromConsole(reader);
		
		double[][] inputs=MI2_IO.readInputs(reader, inputArchitecture[0]);
		System.out.println(inputs.length);
		for(int i=0;i<inputs.length;i++){
			myNetwork.setInput(inputs[i]);
			MI2_IO.writeDoubles(myNetwork.getOutput());
		}
	}
}
