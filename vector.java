import java.util.*;

public class vector{
	int dim;
	double[] v;
	public vector(int m){
		dim=m;
		v=new double[dim];
	}

// Build a vector from a string. The string is assumed to compose of things like ".1 .2 .5"	
	public vector(String vec){
		StringTokenizer tk=new StringTokenizer(vec);
		dim=tk.countTokens();
		v=new double[dim];
		for(int i=0;i<dim;i++){
			v[i]=Double.parseDouble(tk.nextToken());
		}
	}
	
	public vector(int m, boolean rand, double s){
		dim=m;
		v=new double[m];
		if(rand){
			for(int i=0;i<m;i++){
				v[i]=s+(double)(i+1)/(2*m);
			}
		}
	}
	
	public vector(double[] vec){	
		dim=vec.length;
		v=new double[dim];
		for(int i=0;i<vec.length;i++){
			v[i]=vec[i];
		}
	}
	
	public vector(vector[] vecs){
		dim=0;
		for(int i=0;i<vecs.length;i++){
			dim=dim+vecs[i].dim;
		}
		v=new double[dim];
		int ctr=0;
		for(int i=0;i<vecs.length;i++){
			for(int j=0;j<vecs[i].dim;j++){
				v[ctr]=vecs[i].get(j);
				ctr++;
			}
		}
	}
	
	public double get(int m){
		return v[m];
	}
	
	public void set(int m, double x){
		v[m]=x;
	}
	
	public void acc(int m, double x){
		v[m]=v[m]+x;
	}
	
	public double dot_product(vector m){
		if(m.dim!=dim){
			System.out.println("Dot product error: vector dimension not agree!");
			System.out.println(this);
			System.out.println("______");
			System.out.println(m);
			System.exit(0);
		}
		double temp=0.0;
		for(int i=0;i<dim;i++){
			temp=temp+v[i]*m.v[i];
		}
		return temp;
	}
	
	public void add(double[] a){
		for(int i=0;i<dim;i++){
			this.set(i, a[i]+this.get(i));
		}
	}
	
	public void add(vector a){
		if(a.dim!=dim){
			System.out.println("Adding vectors of different dimension!");
		}
		for(int i=0;i<dim;i++){
			this.set(i, a.get(i)+this.get(i));
		}
	}
	
	public void multiply(double mul){
		for(int i=0;i<dim;i++){
			this.set(i, mul*this.get(i));		
		}
	}
	
	public double[] arrayForm(){
		double[] temp=new double[dim];
		for(int i=0;i<dim;i++){
			temp[i]=v[i];
		}
		return temp;
	}
	
	//Decompose this vector into a set of vectors, each with dimension d
	public vector[] arrayForm(int d){
		int l=(int)(dim/d);
		vector[] temp=new vector[l];
		int ctr=0;
		for(int i=0;i<l;i++){
			temp[i]=new vector(d);
			for(int j=0;j<d;j++){
				temp[i].set(j, this.get(ctr));
				ctr++;
			}
		}
		return temp;
	}
	
	public String toString(){
		String temp="";
		for(int i=0;i<v.length;i++){
			temp=temp+v[i]+" ";
		}
		return temp;
	}
}