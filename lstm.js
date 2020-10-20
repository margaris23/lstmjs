// LSTM cell

// Gates are sigmoid neural network layers!!!

// tanh pushes values between -1 and 1
// sigmoid = 1 / ( 1 + e ^ x)
function sigmoig(x) {
  return 1 / (1 + Math.exp(x));
}

// const inputGate = 1;
// const forgetGate = 2;
// const outputGate = 3;

// memory from previous block
const S_t_prev = []; // vector

// ==== START OF FORGET CELL
const X_t_input_vector = [];
const h_t_prev_output = [];

function f_t_forgetGate(x, h) {
  // TODO: find implementation
  // probably a sigmoig with output 0 or 1

  // sigmoid(Wf * [h_t_prev, x_t] + bf)
  // Wf === weight, randomized initialy
  // bf === bias, randomized initialy
  return [];
}

// elementwise multiplication
const mul_forget = S_t_prev * f_t_forgetGate(X_t_input_vector, h_t_prev_output);
// ==== END OF FORGET CELL

// ==== START OF INPUT CELL
function i_t_inputGate(x, h) {
  // TODO: find implementation
  // sigmoid(Wi * [h_t_prev, x_t] + bi)
  return [];
}

// Or Else Candidate C_tilde_t
function s_tilde(x, h) {
  // TODO: find implementation
  // tanh(Wc*[h_t_prev, x_t] + bc)
  return [];
}

// elementwise multiplication
const mul_input =
  s_tilde(X_t_input_vector, h_t_prev_output) *
  i_t_inputGate(X_t_input_vector, h_t_prev_output);
// ==== END OF INPUT CELL

// elementwise summation
const sum_forget_input = mul_forget + mul_input;

// ==== START OF OUTPUT CELL
function o_t_outputGate(x, h) {
  // TODO: find implementation
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
