export function sigmoid(array: any[]) {
  return array.map(x => 1 / (1 + Math.exp(x)));
}

export function tanh(array: any[]) {
  return array.map(x => Math.tanh(x));
}

// Hadamard Product + bias
// https://en.wikipedia.org/wiki/Hadamard_product_(matrices)
// only Nx1 vectors for now, x, y, b are arrays
export function vector_mul_plus_b(x: any[], y: any[], b: any[]) {
  const res = [];
  for (let i = 0; i < x.length; i++) {
    res.push(x[i] * y[i] + b[i]);
  }
  return res;
}

export function vector_sum(x: any[], y: any[]) {
  const res = [];
  for (let i = 0; i < x.length; i++) {
    res.push(x[i] + y[i]);
  }
  return res;
}

