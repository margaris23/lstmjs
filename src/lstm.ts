// LSTM cell with Forget Gate

// Gates are sigmoid neural network layers!!!
// tanh pushes values between -1 and 1
import { sigmoid, tanh, vector_mul_plus_b, vector_sum } from './helpers';

// memory from previous block
const C_t_prev: any[] = [];  // should reset to zero

// ==== START OF FORGET CELL
const X_t_input_vector: any[] = []; // should reset to 0
const h_t_prev_output: any[] = []; // should reset to 0

// function f_t_forgetGate(x, h) {
//   // sigmoid(Wf * [h_t_prev, x_t] + bf)
//   // Wf === weight, randomized initialy
//   // bf === bias, randomized initialy
//   return [];
// }

// FORGET GATE
let w_forget: any[] = []; // W, U contains both weights for h and x
let b_forget: any[] = [];
const f_t_forgetGate = sigmoid(
  vector_mul_plus_b(
    w_forget,
    h_t_prev_output.concat(X_t_input_vector),
    b_forget
  )
);

// elementwise multiplication // TODO: fix
const mul_forget: any[] = vector_mul_plus_b(C_t_prev, f_t_forgetGate, []);
// ==== END OF FORGET CELL

// ==== START OF INPUT CELL
let w_input: any[] = []; // W, U contains both weights for h and x
let b_input: any[] = [];
const i_t_inputGate: any[] = sigmoid(
  vector_mul_plus_b(
    w_input,
    h_t_prev_output.concat(X_t_input_vector),
    b_input
  )
);

// cell input activation vector
let w_c: any[] = []; // W, U contains both weights for h and x
let b_c: any[] = [];
const c_t_activator: any[] = tanh(
  vector_mul_plus_b(
    w_c,
    h_t_prev_output.concat(X_t_input_vector),
    b_c
  )
);

// elementwise multiplication
const mul_input: any[] = vector_mul_plus_b(c_t_activator, i_t_inputGate, []);
// ==== END OF INPUT CELL

// elementwise summation
const sum_forget_input: any[] = vector_sum(mul_forget, mul_input);

// ==== START OF OUTPUT CELL
let w_output: any[] = []; // W, U contains both weights for h and x
let b_output: any[] = [];
const o_t_outputGate: any[] = sigmoid(
  vector_mul_plus_b(
    w_output,
    h_t_prev_output.concat(X_t_input_vector),
    b_output
  )
);

// Cell State == Memory from current block
// Ct = f_t * C_t_prev + i_t * C_tilde_t
const S_t_next: any[] = sum_forget_input;

// elementwise multiplication
const mul_state_output: any[] = vector_mul_plus_b(o_t_outputGate, tanh(S_t_next), []);
// ==== END OF INPUT CELL

// Output of current block
const h_t_next: any[] = mul_state_output;
