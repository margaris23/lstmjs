// LSTM cell with Forget Gate

// Gates are sigmoid neural network layers!!!
// tanh pushes values between -1 and 1
import { sigmoid, tanh, vector_mul_plus_b, vector_sum } from './helpers';

// memory from previous block
const cellStatePrev: any[] = [];  // should reset to zero

// ==== START OF FORGET CELL
const xTInputVector: any[] = []; // should reset to 0
const hTPrevOutput: any[] = []; // should reset to 0

// function f_t_forgetGate(x, h) {
//   // sigmoid(Wf * [h_t_prev, x_t] + bf)
//   // Wf === weight, randomized initialy
//   // bf === bias, randomized initialy
//   return [];
// }

// FORGET GATE
const wForget: any[] = []; // W, U contains both weights for h and x
const bForget: any[] = [];
const fTForgetGate = sigmoid(
  vector_mul_plus_b(
    wForget,
    hTPrevOutput.concat(xTInputVector),
    bForget
  )
);

// elementwise multiplication // TODO: fix
const mulForget: any[] = vector_mul_plus_b(cellStatePrev, fTForgetGate, []);
// ==== END OF FORGET CELL

// ==== START OF INPUT CELL
const wInput: any[] = []; // W, U contains both weights for h and x
const bInput: any[] = [];
const iTInputGate: any[] = sigmoid(
  vector_mul_plus_b(
    wInput,
    hTPrevOutput.concat(xTInputVector),
    bInput
  )
);

// cell input activation vector
const wC: any[] = []; // W, U contains both weights for h and x
const bC: any[] = [];
const cTActivator: any[] = tanh(
  vector_mul_plus_b(
    wC,
    hTPrevOutput.concat(xTInputVector),
    bC
  )
);

// elementwise multiplication
const mulInput: any[] = vector_mul_plus_b(cTActivator, iTInputGate, []);
// ==== END OF INPUT CELL

// elementwise summation
const sumForgetInput: any[] = vector_sum(mulForget, mulInput);

// ==== START OF OUTPUT CELL
const wOutput: any[] = []; // W, U contains both weights for h and x
const bOutput: any[] = [];
const oTOutputGate: any[] = sigmoid(
  vector_mul_plus_b(
    wOutput,
    hTPrevOutput.concat(xTInputVector),
    bOutput
  )
);

// Cell State == Memory from current block
// Ct = f_t * C_t_prev + i_t * C_tilde_t
const sTNext: any[] = sumForgetInput;

// elementwise multiplication
const mulStateOutput: any[] = vector_mul_plus_b(oTOutputGate, tanh(sTNext), []);
// ==== END OF INPUT CELL

// Output of current block
const hTNext: any[] = mulStateOutput;
