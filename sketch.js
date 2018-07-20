var nn;

var f1 = true,f2 = false;
var targets;
var inputs;
var aux;

function setup(){
  nn = new NeuralNetwork(2,2,1);
  targets = [];
  inputs = [];

  createCanvas(600,400);

  for (var i=0;i<10000;i++){
    var aux = random(10);
    if (aux < 2.5){
      inputs[0] = 1;
      inputs[1] = 1;
      targets[0] = 0;
    }
    else if (aux < 5){
      inputs[0] = 1;
      inputs[1] = 0;
      targets[0] = 1;
    }
    else if (aux < 7.5){
      inputs[0] = 0;
      inputs[1] = 1;
      targets[0] = 1;
    }
    else {
      inputs[0] = 0;
      inputs[1] = 0;
      targets[0] = 0;
    }
    //println(inputs);
    nn.evolve(inputs,targets);
  }


  noLoop();
}

function draw(){
      inputs[0] = 1;
      inputs[1] = 1;
      console.log(nn.estimate(inputs));
      inputs[0] = 1;
      inputs[1] = 0;
      console.log(nn.estimate(inputs));
      inputs[0] = 0;
      inputs[1] = 1;
      console.log(nn.estimate(inputs));
      inputs[0] = 0;
      inputs[1] = 0;
      console.log(nn.estimate(inputs));
}
