class NeuralNetwork{

  constructor(a,b,c){

    this.inputs = a;
    this.hidden = b;
    this.outputs = c;
    this.lr = 0.75;

    this.m1 = new Matrix(this.inputs,this.hidden);
    this.m2 = new Matrix(this.hidden,this.outputs);

    this.bh = new Matrix(1,this.hidden);
    this.bo = new Matrix(1,this.outputs);
  }

  estimate (input){

    var out = [];

    var in_m = new Matrix(1,this.inputs);
    var out_m = new Matrix(1,this.outputs);
    in_m.data[0] = input;

    var hidden = product(in_m,this.m1);
    hidden.sum(this.bh);
    for (var i=0;i<hidden.r;i++){
      for (var j=0;j<hidden.c;j++){
        hidden.data[i][j] = sigmoid(hidden.data[i][j]);
      }
    }

    out_m = product(hidden,this.m2);
    out_m.sum(this.bo);
    for (var i=0;i<out_m.r;i++){
      for (var j=0;j<out_m.c;j++){
        out_m.data[i][j] = sigmoid(out_m.data[i][j]);
      }
    }
    out = out_m.data[0];
    return out;
  }

  evolve(input, tar){

    var out = [this.outputs];

    var in_m = new Matrix(1,this.inputs);
    var out_m = new Matrix(1,this.outputs);

    var err_bh = new Matrix(1,this.hidden);
    var err_bo = new Matrix(1,this.outputs);
    in_m.data[0] = input;

    var hidden = product(in_m,this.m1);

    hidden.sum(this.bh);
    for (var i=0;i<hidden.r;i++){
      for (var j=0;j<hidden.c;j++){
        hidden.data[i][j] = sigmoid(hidden.data[i][j]);
      }
    }

    out_m = product(hidden,this.m2);
    out_m.sum(this.bo);
    for (var i=0;i<out_m.r;i++){
      for (var j=0;j<out_m.c;j++){
        out_m.data[i][j] = sigmoid(out_m.data[i][j]);
      }
    }
    out = out_m.data[0];

    var err = new Matrix(1,this.outputs);
    var errors = [];

    for (var i=0;i<this.outputs;i++){
      errors[i] = tar[i]-out[i];

      err.data[0][i] = errors[i];
    }


    var h_err = product(this.m2,err.transpose());
    var h_errors = [];
    for (var i=0;i<h_err.r;i++){
      h_errors[i] = h_err.data[i][0];
    }
    var error_ho = new Matrix(this.m2.r,this.m2.c);


    for (var i=0;i<error_ho.r;i++){
      for (var j=0;j<error_ho.c;j++){
        error_ho.data[i][j] = this.lr * errors[j] * dsigmoid(out[j]) * hidden.data[0][i];
      }
    }


    var error_ih = new Matrix(this.m1.r,this.m1.c);

    for (var i=0;i<error_ih.r;i++){
      for (var j=0;j<error_ih.c;j++){
        error_ih.data[i][j] *= this.lr;
        error_ih.data[i][j] *= h_errors[j];
        error_ih.data[i][j] *= dsigmoid(hidden.data[0][j]);
        error_ih.data[i][j] *= input[i];
        }
    }

    for (var i=0;i<err_bo.r;i++){
      for (var j=0;j<err_bo.c;j++){
        err_bo.data[i][j] = this.lr * errors[j] * dsigmoid(out[j]);
      }
    }

    for (var i=0;i<err_bh.r;i++){
      for (var j=0;j<err_bh.c;j++){
        err_bh.data[i][j] *= this.lr;
        err_bh.data[i][j] *= h_errors[j];
        err_bh.data[i][j] *= dsigmoid(hidden.data[0][j]);
      }
    }

    for (var i=0;i<err_bh.r;i++){
      for (var j=0;j<err_bh.c;j++){
        this.bh.data[i][j]+=err_bh.data[i][j];
      }
    }

    for (var i=0;i<err_bo.r;i++){
      for (var j=0;j<err_bo.c;j++){
        this.bo.data[i][j] +=err_bo.data[i][j];
      }
    }


    for (var i=0;i<this.m1.r;i++){
      for (var j=0;j<this.m1.c;j++){
        this.m1.data[i][j] += error_ih.data[i][j];

      }
    }

    for (var i=0;i<this.m2.r;i++){
      for (var j=0;j<this.m2.c;j++){
        this.m2.data[i][j] += error_ho.data[i][j];
      }
    }

  }

}

class Matrix{

  constructor(r, c){
    this.r = r;
    this.c = c;
    this.data = createArray(r,c);
  }

  transpose(){

    var out = new Matrix(this.c,this.r);
    for (var i=0;i<this.r;i++){
      for (var j=0;j<this.c;j++){
        out.data[j][i] = this.data[i][j];
      }
    }
    return out;
  }

  sum(m){

   for (var i=0;i<this.r;i++){
      for (var j=0;j<this.c;j++){
        this.data[i][j] += m.data[i][j];
      }
    }
  }

  printData(){

    console.log(this.data);
  }

}

function createArray(r,c){

  var data = [];
  var temp = [];
  for (var i=0;i<r;i++){
    for (var j=0;j<c;j++){
      temp.push(random(0,1));
    }
    data.push(temp);
    temp = [];
  }
  return data;
}

function product(m1, m2){

  var rezult = new Matrix(m1.r,m2.c);
  for (var i=0;i<rezult.r;i++){
    for (var k=0;k<rezult.c;k++){
      rezult.data[i][k] = 0;
    }
  }

  for (var i=0;i<m1.r;i++){
    for (var k=0;k<m2.c;k++){
      for (var j=0;j<m1.c;j++){
        rezult.data[i][k]+=m1.data[i][j]*m2.data[j][k];
      }
    }
  }
  return rezult;
}

function sigmoid(x){
  return 1/(1+exp(-x));
}
function dsigmoid(y){
  return y*(1-y);
}
