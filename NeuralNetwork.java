package mi_lecke_2;

import java.util.ArrayList;

import java.util.Random;
import java.util.Scanner;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class NeuralNetwork {
	
	public Layer firstLayer;
	public Layer lastLayer;
	
	
	public NeuralNetwork(int[] architecture){
		firstLayer = new Layer(architecture[0], architecture[1]);
		
		Layer actualLayer = firstLayer;
		for(int i=2; i < architecture.length; i++){
			actualLayer.nextLayer = new Layer(architecture[i-1],architecture[i]);
			actualLayer.nextLayer.prevLayer=actualLayer;
	
			
			actualLayer=actualLayer.nextLayer;			
		}
		lastLayer=actualLayer;
		
	}	
	
	public void initializeWeightsRandom(){
		firstLayer.initializeWeightsDeep(new Random());
	}
	
	public int[] getArchitecture(){
		ArrayList<Integer> architecture = new ArrayList<Integer>();
		
		Layer item = firstLayer;
		
		architecture.add(new Integer(item.getNumberOfInputs()));
		architecture.add(new Integer(item.getNumberOfNeurons()));
		while(item.nextLayer!=null){
			item=item.nextLayer;
			architecture.add(new Integer(item.getNumberOfNeurons()));
		}
		
		int[] architectureArray=new int[architecture.size()];
		
		int i=0;
		for(Integer Item : architecture){
			architectureArray[i]=Item.intValue();
			i++;
		}
		

		return architectureArray;
	}
	
	public void initializeWeightsFromConsole(Scanner reader){
		
		for(Layer layer= firstLayer; layer != null; layer=layer.nextLayer){
	
			int columns=layer.getNumberOfInputs();
			int rows=layer.getNumberOfNeurons();
			
			double[][] weightMatrix=new double[rows][columns];
			double[] biasVector=new double[rows];

			for(int i=0; i<rows; i++){
				double[] line=MI2_IO.readLineOfDoubles(reader);
				
				if(columns!=line.length-1){
					System.out.println("Nem megfelelo szamu bemenet");
				}

				for(int j=0;j<columns;j++){
					weightMatrix[i][j]=line[j];
					
				}
				biasVector[i]= line[line.length-1];
			}
			
			layer.setW(MatrixUtils.createRealMatrix(weightMatrix));
			layer.setB(new ArrayRealVector(biasVector));
		}
		
		
		
	}
	public void setInput(double[] input){
		firstLayer.setXrecursive(new ArrayRealVector(input));
		
	}
	public double[] getOutput(){
		return lastLayer.getS().toArray();
	}
	
	public ArrayList<double[]> getWeightAndBiasValues(){
		ArrayList<double[]> weights = new ArrayList<double[]>();
		
		
		for(Layer layer= firstLayer; layer != null; layer=layer.nextLayer){
			double[][] W = layer.getW().getData();
			double[] b=layer.getB().toArray();
			
			for(int i=0;i<W.length;i++){
				double[] weightBiasArray= new double[W[0].length+1];
				for(int j=0;j<W[0].length;j++){
					weightBiasArray[j]=W[i][j];
				}
				weightBiasArray[weightBiasArray.length-1]=b[i];
				weights.add(weightBiasArray);
			}
		}
		return weights;
	}
	
	//================================================================
	//NNSolutionThree
	//================================================================
	public void calculateDeltaYperDeltaWij(){
		RealMatrix I=MatrixUtils.createRealIdentityMatrix(lastLayer.getS().getDimension());
		lastLayer.setDeltaYRecursive(I);
	}
	
	public void printDeltaYperDeltaWij(){
		firstLayer.printdYPerdWijMatrixRecursive();
	}
	
	//================================================================
	//NNSolutionThree
	//================================================================
	public void calculateDeltas(double[] epsilon){
		lastLayer.setDeltasRecursive(new ArrayRealVector(epsilon));
	}
	public void modifyWeigths(double u){
		firstLayer.modifyWeightsRecursive(u);
	}
}
