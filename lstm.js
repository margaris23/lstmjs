// LSTM cell with Forget Gate

// Gates are sigmoid neural network layers!!!

// tanh pushes values between -1 and 1

function sigmoig(x) {
  return 1 / (1 + Math.exp(x));
}

// Hadamard Product + bias
// https://en.wikipedia.org/wiki/Hadamard_product_(matrices)
// only Nx1 vectors for now
function vector_mul_plus_b(x, y, b) {
  let res = 0;
  for (let i = 0; i < x.length; i++) {
    res += x[i] * y[i];
  }
  return res + b;
}

// const inputGate = 1;
// const forgetGate = 2;
// const outputGate = 3;

// memory from previous block
const C_t_prev = [];

// ==== START OF FORGET CELL
const X_t_input_vector = []; // should reset to 0
const h_t_prev_output = []; // should reset to 0

// function f_t_forgetGate(x, h) {
//   // sigmoid(Wf * [h_t_prev, x_t] + bf)
//   // Wf === weight, randomized initialy
//   // bf === bias, randomized initialy
//   return [];
// }

// FORGET GATE
let w_forget = []; // W, U contains both weights for h and x
let b_forget = 0;
const f_t_forgetGate = sigmoig(
  vector_mul_plus_b(
    w_forget,
    h_t_prev_output.concat(X_t_input_vector),
    b_forget
  )
);

// elementwise multiplication
const mul_forget = vector_mul_plus_b(C_t_prev, f_t_forgetGate);
// ==== END OF FORGET CELL

// ==== START OF INPUT CELL
function i_t_inputGate(x, h) {
  // sigmoid(Wi * [h_t_prev, x_t] + bi)
  return [];
}

// cell input activation vector
function C_tilde(x, h) {
  // tanh(Wc*[h_t_prev, x_t] + bc)
  return [];
}

// elementwise multiplication
const mul_input =
  C_tilde(X_t_input_vector, h_t_prev_output) *
  i_t_inputGate(X_t_input_vector, h_t_prev_output);
// ==== END OF INPUT CELL

// elementwise summation
const sum_forget_input = mul_forget + mul_input;

// ==== START OF OUTPUT CELL
function o_t_outputGate(x, h) {
  // sigmoid(Wo * [h_t_prev, x_t] + bo)
  return [];
}

// Cell State == Memory from current block
// Ct = f_t * C_t_prev + i_t * C_tilde_t
const S_t_next = sum_forget_input;

// elementwise multiplication
const mul_state_output =
  Math.tanh(s_cell_state) * o_t_outputGate(X_t_input_vector, h_t_prev_output);
// ==== END OF INPUT CELL

// Output of current block
const h_t_next = mul_state_output;
