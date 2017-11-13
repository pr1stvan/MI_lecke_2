package mi_lecke_2;


import java.util.Random;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class Layer {
	
	public RealVector x;
	public RealVector b;
	public RealVector s;
	public RealMatrix W;
	
	//NNSolutionThree
	public RealMatrix deltaY=null;
	
	//NNSolutionFour
	public RealVector delta=null;
	
	public Layer nextLayer=null;
	public Layer prevLayer=null;
	
	public interface ActivationFunctionInterface {
		double f(double input);
		double df(double input);
	}
	ActivationFunctionInterface activation;
	
	public Layer(int numberOfInputs,int numberOfNeurons){
		x=new ArrayRealVector(numberOfInputs);
		b=new ArrayRealVector(numberOfNeurons);
		
		W=MatrixUtils.createRealMatrix(numberOfNeurons, numberOfInputs);
		
		//Relu
		activation=new ActivationFunctionInterface(){
			@Override
			public double f(double input) {
				return input > 0 ? input :0;
			}

			@Override
			public double df(double input) {
				return input > 0 ? 1 :0;
			}
		};
	}
	
	public void setActivationFunctionInterfaceRecursive(ActivationFunctionInterface a){
		this.activation=a;
		if(nextLayer!=null){
			nextLayer.setActivationFunctionInterfaceRecursive(a);
		}
	}
	
	public void initializeWeightsDeep(Random rnd){
		double[][] data=W.getData();
		
		for (int i = 0; i < data.length; i++) {
			for (int j = 0; j < data[i].length; j++) {
				data[i][j]=rnd.nextGaussian()*0.1;
			}
		}
		W=MatrixUtils.createRealMatrix(data);
		
		if(nextLayer !=null)nextLayer.initializeWeightsDeep(rnd);
	}
	
	public int getNumberOfInputs(){
		return x.getDimension();
	}
	
	public int getNumberOfNeurons(){
		return W.getRowDimension();
	}
	
	public RealMatrix getW(){
		return W;
	}
	public RealVector getS(){
		return s;
	}
	
	public void setW(RealMatrix W){
		this.W=W;
	}
	public void setB(ArrayRealVector b){
		this.b=b;
	}

	public RealVector getB(){
		return b;
	}
	
	public RealVector f(RealVector v){
		double[] data=v.toArray();
		for(int i=0;i<data.length;i++){
			data[i]=activation.f(data[i]);
		}
		return new ArrayRealVector(data);
	}
	public RealMatrix f(RealMatrix m){
		double[][] data=m.getData();
		
		for(int i=0; i< data.length;i++){
			for(int j=0;j<data[0].length;j++){
				data[i][j]=activation.f(data[i][j]);
			}
		}
		
		return MatrixUtils.createRealMatrix(data);
		
	}
	public RealMatrix df(RealMatrix m){
		double[][] data=m.getData();
		
		for(int i=0; i< data.length;i++){
			for(int j=0;j<data[0].length;j++){
				data[i][j]=activation.df(data[i][j]);
			}
		}
		
		return MatrixUtils.createRealMatrix(data);
		
	}
	
	public void setXrecursive(RealVector x) {
		this.x=x;
		s=W.operate(x);
		s=b.add(s);
//		System.out.println("a reteg merete:" +s.toArray().length);
//		System.out.println("s tartalma:" + Arrays.toString(s.toArray()));
		if(nextLayer!=null){
			nextLayer.setXrecursive(f(s));
		}
	}

	//===================================================================
	//NNSolutionThree
	//===================================================================
	public void setDeltaYRecursive(RealMatrix prevMatrix){
		if(nextLayer==null){
			deltaY=prevMatrix;
		}
		else{
			RealMatrix S=MatrixUtils.createRealDiagonalMatrix(s.toArray());
			deltaY=prevMatrix.multiply(df(S));
		}
		if(prevLayer!=null)prevLayer.setDeltaYRecursive(deltaY.multiply(W));
	}
	
	public void printdYPerdWijMatrix(){
		double[][] weights=W.getData();
		
		double[][] out=new double[weights.length][weights[0].length +1];
		
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[0].length; j++) {
				double xj=x.getEntry(j);			
				RealVector v=new ArrayRealVector(s.getDimension());
				v.setEntry(i, xj);
				RealVector deltaYperDeltaWij=deltaY.operate(v);
				out[i][j]=deltaYperDeltaWij.getEntry(0);
			}
			RealVector v2=new ArrayRealVector(s.getDimension());
			v2.setEntry(i, 1);
			RealVector deltaYperDeltaBi=deltaY.operate(v2);
			out[i][weights[0].length]=deltaYperDeltaBi.getEntry(0);
		}
		
		MI2_IO.writeMatrix(out);
	}
	
	public void printdYPerdWijMatrixRecursive(){
		printdYPerdWijMatrix();
		if(nextLayer!=null)nextLayer.printdYPerdWijMatrixRecursive();
	}
	
	//=====================================================================
	//NNSolutionFour
	//=====================================================================
	
	public void setDeltasRecursive(RealVector prevDeltaMultipliedByW){
		if(nextLayer==null){
			delta=prevDeltaMultipliedByW;
		}
		else{
			RealMatrix S=MatrixUtils.createRealDiagonalMatrix(s.toArray());
			delta=df(S).preMultiply(prevDeltaMultipliedByW);
		}
		if(prevLayer!=null)prevLayer.setDeltasRecursive(W.preMultiply(delta));
	}
	
	public void modifyWeightsRecursive(double u){
		RealVector twoUdelta=delta.mapMultiply(2*u);
		W=W.add(twoUdelta.outerProduct(x));
		b=b.add(twoUdelta);
		
		if(nextLayer!=null)nextLayer.modifyWeightsRecursive(u);
	}
	
}
