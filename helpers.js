module.exports.sigmoid = function sigmoid(array) {
  return array.map(x => 1 / (1 + Math.exp(x)));
}

module.exports.tanh = function tanh(array) {
  return array.map(x => Math.tanh(x));
}

// Hadamard Product + bias
// https://en.wikipedia.org/wiki/Hadamard_product_(matrices)
// only Nx1 vectors for now, x, y, b are arrays
module.exports.vector_mul_plus_b = function vector_mul_plus_b(x, y, b) {
  let res = [];
  for (let i = 0; i < x.length; i++) {
    res.push(x[i] * y[i] + b[i]);
  }
  return res;
}

module.exports.vector_sum = function vector_sum(x, y) {
  let res = [];
  for (let i = 0; i < x.length; i++) {
    res.push(x[i] + y[i]);
  }
  return res;
}

