var nn;
var inputs = [];
var targets = [];

function setup() {
  nn = new NeuralNetwork(2,2,1);
  for (var i=0;i<1000;i++){
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

function draw() {
  // put drawing code here
}
