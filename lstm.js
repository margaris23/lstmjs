// LSTM cell with Forget Gate

// Gates are sigmoid neural network layers!!!
// tanh pushes values between -1 and 1
const { sigmoid, tanh, vector_mul_plus_b, vector_sum } = require('./helpers');

// memory from previous block
const C_t_prev = [];  // should reset to zero

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
const f_t_forgetGate = sigmoid(
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
let w_input = []; // W, U contains both weights for h and x
let b_input = 0;
const i_t_inputGate = sigmoid(
  vector_mul_plus_b(
    w_input,
    h_t_prev_output.concat(X_t_input_vector),
    b_input
  )
);

// cell input activation vector
let w_c = []; // W, U contains both weights for h and x
let b_c = 0;
const c_t_activator = tanh(
  vector_mul_plus_b(
    w_c,
    h_t_prev_output.concat(X_t_input_vector),
    b_c
  )
);

// elementwise multiplication
const mul_input = vector_mul_plus_b(c_t_activator, i_t_inputGate);
// ==== END OF INPUT CELL

// elementwise summation
const sum_forget_input = vector_sum(mul_forget, mul_input);

// ==== START OF OUTPUT CELL
let w_output = []; // W, U contains both weights for h and x
let b_output = 0;
const o_t_outputGate = sigmoid(
  vector_mul_plus_b(
    w_output,
    h_t_prev_output.concat(X_t_input_vector),
    b_output
  )
);

// Cell State == Memory from current block
// Ct = f_t * C_t_prev + i_t * C_tilde_t
const S_t_next = sum_forget_input;

// elementwise multiplication
const mul_state_output = vector_mul_plus_b(o_t_outputGate, tanh(S_t_next));
// ==== END OF INPUT CELL

// Output of current block
const h_t_next = mul_state_output;
