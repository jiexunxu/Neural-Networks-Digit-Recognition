public class Matrix{
	double[][] M;
	int row;
	int col;
	boolean sparse;
	T_Tuple_LinkedList list;
	
	public Matrix(int m, int n, boolean spr){
		row=m;
		col=n;
		sparse=spr;
		if(!sparse){
			M=new double[m][n];			
		}else{
			list=new T_Tuple_LinkedList();
		}		
	}
	
	public void set(int m, int n, double v){
		if((n>=col)||(m>=row)){
			System.out.println("Setting an element that's out of bounds of this matrix!");
			return;
		}
		if(!sparse){
			M[m][n]=v;
		}else{
			if(v==0.0){
				return;
			}			
			T_Tuple t=new T_Tuple(m, n, v);
			list.add(t);
		}		
	}
	
	public double get(int m, int n){
		return M[m][n];
	}
	
	public int getRow(){
		return row;
	}
	
	public int getColumn(){
		return col;
	}
	
	//Assume that Mv, not vM
	public vector dot_product(vector v){		
		if(v.dim!=col){
			System.out.println("Dot product dimension not agree!");
			return null;
		}
		vector temp=new vector(row);
		if(!sparse){
			for(int i=0;i<row;i++){
				double sum=0;
				for(int j=0;j<col;j++){
					sum=sum+M[i][j]*v.get(j);
				}
				temp.set(i, sum);
			}
			return temp;
		}	
		list.reset_cur();		
		while(list.isValid()){
			T_Tuple t=list.next();
			temp.acc(t.col, t.value*v.get(t.row));
		}		
		return temp;
	}
	
	//Assume that vM, not Mv
	public vector product(vector v){
		vector temp=new vector(col);
		if(v.dim!=row){
			System.out.println("Vector matrix product dimension not agree!");
		}
		if(!sparse){
			for(int i=0;i<col;i++){
				double sum=0.0;
				for(int j=0;j<row;j++){
					sum=sum+M[j][i]*v.get(j);
				}
				temp.set(i, sum);
			}
			return temp;
		}
		list.reset_cur();
		while(list.isValid()){
			T_Tuple t=list.next();
			temp.acc(t.col, t.value*v.get(t.row));
		}
		return temp;
	}
	
	//Assume that vM, not Mv
	public vector product(double[] v){
		return product(new vector(v));
	}
	
	public String toString(){
		if(sparse){
			double[][] tp=new double[row][col];
			list.reset_cur();
			while(list.isValid()){
				T_Tuple t=list.next();
				tp[t.row][t.col]=t.value;
			}
			String tps="";
			for(int i=0;i<row;i++){
				for(int j=0;j<col;j++){
					tps=tps+Math.round(tp[i][j]*10000)/10000.0+" ";
				}
				tps=tps+"\n";
			}
			return tps;
		}
	
		String temp="";
		for(int i=0;i<row;i++){
			for(int j=0;j<col;j++){
				temp=temp+M[i][j]+" ";
			}
			temp=temp+"\n";
		}
		return temp;
	}	
}

class T_Tuple_LinkedList{
	T_Tuple head;
	T_Tuple current;
	int count;
	
	public void add(T_Tuple t){
		t.next=head;
		head=t;		
		count++;
	}
	
	public boolean isValid(){
		if(current==null){
			return false;
		}
		return true;
	}
	
	public T_Tuple next(){
		T_Tuple temp=current;
		current=current.next;
		return temp;
	}
	
	public void reset_cur(){
		current=head;
	}
} 

class T_Tuple{
	public int row;
	public int col;
	public double value;
	public T_Tuple next;
	public T_Tuple(int m, int n, double v){
		row=m;col=n;value=v;
	}
	
	public String toString(){
		return "T_Tuple: ("+row+", "+col+", "+value+")";
	}
}