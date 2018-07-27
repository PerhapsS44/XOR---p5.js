var nn;
var inputs = [];
var targets = [];

function setup() {
  nn = new NeuralNetwork(2,2,1);
  TrainNN(10000);
}

function TrainNN(n){
  for (var i=0;i<n;i++){
    chooseRandomTrainingSet();
    nn.evolve(inputs,targets);
  }
}

function chooseRandomTrainingSet(){
  var r = random(0,10);
  if (r < 2.5){
    inputs = [1,1];
    targets = [0];
  }
  else if (r < 5){
    inputs = [0,1];
    targets = [1];
  }
  else if (r < 7.5){
    inputs = [1,0];
    targets = [1];
  }
else{
    inputs = [0,0];
    targets = [0];
  }
}

function consoleOutputs(){
  console.log("XOR(1,1) = " + nn.estimate([1,1]));
  console.log("XOR(0,1) = " + nn.estimate([0,1]));
  console.log("XOR(1,0) = " + nn.estimate([1,0]));
  console.log("XOR(0,0) = " + nn.estimate([0,0]));
}

function draw() {
  createP("The Neural Network sais that XOR(1,1) = " + nn.estimate([1,1]));
  createP("The Neural Network sais that XOR(0,1) = " + nn.estimate([0,1]));
  createP("The Neural Network sais that XOR(1,0) = " + nn.estimate([1,0]));
  createP("The Neural Network sais that XOR(0,0) = " + nn.estimate([0,0]));
  noLoop();
}
