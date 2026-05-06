import Fsa.V5.Types
import Fsa.V5.Drift
import Fsa.V5.Cost
import Fsa.V5.Schedule
import Fsa.V5.Plant
import Fsa.V5.Obs
import Lean.Data.Json

/-!
# CLI bridge for differential testing.

Reads a JSON object on stdin describing a single drift call, writes a
JSON object on stdout with the computed derivative. Used by the Python
`tests/test_lean4_diff.py` harness (charter §5 step 6).

Input schema:
```
{
  "fn":     "drift",                            // dispatch tag
  "state":  [B, S, F, A, KFB, KFS],             // 6 floats
  "phi":    [Phi_B, Phi_S],                     // 2 floats
  "params": { "tau_B": ..., ..., "n_dec": ... } // 28 named-field floats
}
```

Output schema:
```
{ "deriv": [dB, dS, dF, dA, dKFB, dKFS] }       // 6 floats
```

The Python harness can also invoke the binary in batch mode by sending
one JSON object per line on stdin and reading one per line on stdout
(simple line-delimited JSON). See `python_bridge/drift.py`.
-/

open Lean (Json)
open Fsa.V5

/-- Strict numeric extraction. Float-coerces both Int and Float JSON
    nodes; errors on anything else. -/
private def jsonToFloat? (j : Json) : Except String Float :=
  match j with
  | .num n   => Except.ok n.toFloat
  | _        => Except.error s!"expected JSON number, got {j.compress}"

private def getFloat (obj : Json) (key : String) : Except String Float := do
  let v ← obj.getObjVal? key
  jsonToFloat? v

private def getFloatArr (j : Json) : Except String (Array Float) := do
  let arr ← j.getArr?
  arr.mapM jsonToFloat?

private def getMatrix (j : Json) : Except String (Array (Array Float)) := do
  let arr ← j.getArr?
  arr.mapM getFloatArr

private def getStateFromArr (j : Json) : Except String State6D := do
  let arr ← j.getArr?
  if arr.size != 6 then
    Except.error s!"state array must have 6 elements, got {arr.size}"
  else
    pure {
      B   := (← jsonToFloat? arr[0]!),
      S   := (← jsonToFloat? arr[1]!),
      F   := (← jsonToFloat? arr[2]!),
      A   := (← jsonToFloat? arr[3]!),
      KFB := (← jsonToFloat? arr[4]!),
      KFS := (← jsonToFloat? arr[5]!)
    }

private def getPhiFromArr (j : Json) : Except String BimodalPhi := do
  let arr ← j.getArr?
  if arr.size != 2 then
    Except.error s!"phi array must have 2 elements, got {arr.size}"
  else
    pure {
      Phi_B := (← jsonToFloat? arr[0]!),
      Phi_S := (← jsonToFloat? arr[1]!)
    }

private def getObsParams (j : Json) : Except String ObsParams := do
  pure {
    HR_base     := (← getFloat j "HR_base"),
    kappa_B_HR  := (← getFloat j "kappa_B_HR"),
    alpha_A_HR  := (← getFloat j "alpha_A_HR"),
    beta_C_HR   := (← getFloat j "beta_C_HR"),
    sigma_HR    := (← getFloat j "sigma_HR"),
    k_C         := (← getFloat j "k_C"),
    k_A         := (← getFloat j "k_A"),
    c_tilde     := (← getFloat j "c_tilde"),
    S_base      := (← getFloat j "S_base"),
    k_F         := (← getFloat j "k_F"),
    k_A_S       := (← getFloat j "k_A_S"),
    beta_C_S    := (← getFloat j "beta_C_S"),
    sigma_S_obs := (← getFloat j "sigma_S_obs"),
    mu_step0    := (← getFloat j "mu_step0"),
    beta_B_st   := (← getFloat j "beta_B_st"),
    beta_F_st   := (← getFloat j "beta_F_st"),
    beta_A_st   := (← getFloat j "beta_A_st"),
    beta_C_st   := (← getFloat j "beta_C_st"),
    sigma_st    := (← getFloat j "sigma_st"),
    beta_S_VL   := (← getFloat j "beta_S_VL"),
    beta_F_VL   := (← getFloat j "beta_F_VL"),
    sigma_VL    := (← getFloat j "sigma_VL")
  }

private def getParams (j : Json) : Except String Params := do
  pure {
    tau_B      := (← getFloat j "tau_B"),
    kappa_B    := (← getFloat j "kappa_B"),
    epsilon_AB := (← getFloat j "epsilon_AB"),
    tau_S      := (← getFloat j "tau_S"),
    kappa_S    := (← getFloat j "kappa_S"),
    epsilon_AS := (← getFloat j "epsilon_AS"),
    tau_F      := (← getFloat j "tau_F"),
    lambda_A   := (← getFloat j "lambda_A"),
    KFB_0      := (← getFloat j "KFB_0"),
    KFS_0      := (← getFloat j "KFS_0"),
    tau_K      := (← getFloat j "tau_K"),
    mu_K       := (← getFloat j "mu_K"),
    mu_0       := (← getFloat j "mu_0"),
    mu_B       := (← getFloat j "mu_B"),
    mu_S       := (← getFloat j "mu_S"),
    mu_F       := (← getFloat j "mu_F"),
    mu_FF      := (← getFloat j "mu_FF"),
    eta        := (← getFloat j "eta"),
    sigma_B    := (← getFloat j "sigma_B"),
    sigma_S    := (← getFloat j "sigma_S"),
    sigma_F    := (← getFloat j "sigma_F"),
    sigma_A    := (← getFloat j "sigma_A"),
    sigma_K    := (← getFloat j "sigma_K"),
    B_dec      := (← getFloat j "B_dec"),
    S_dec      := (← getFloat j "S_dec"),
    mu_dec_B   := (← getFloat j "mu_dec_B"),
    mu_dec_S   := (← getFloat j "mu_dec_S"),
    n_dec      := (← getFloat j "n_dec")
  }

/-- Float → JSON-number string. We avoid `Lean.JsonNumber` because it
    round-trips Float values lossily (mantissa+exponent). Special-case
    ±inf/NaN to Python-extended-JSON tokens (`Infinity`, `-Infinity`,
    `NaN`) — Python's stdlib `json` accepts these natively. -/
private def floatToJson (x : Float) : String :=
  if x.isNaN then "NaN"
  else if x.isInf then (if x < 0.0 then "-Infinity" else "Infinity")
  else toString x

/-- Format the 6D state derivative as a raw JSON string. -/
private def formatDeriv (y : State6D) : String :=
  "{\"deriv\":[" ++ floatToJson y.B ++ "," ++ floatToJson y.S ++ "," ++
    floatToJson y.F ++ "," ++ floatToJson y.A ++ "," ++ floatToJson y.KFB ++ "," ++
    floatToJson y.KFS ++ "]}"

private def formatScalar (label : String) (x : Float) : String :=
  "{\"" ++ label ++ "\":" ++ floatToJson x ++ "}"

/-- Format an Array Float as a JSON array. -/
private def formatFloatArr (xs : Array Float) : String :=
  let body := xs.foldl
    (fun (acc : String) (x : Float) =>
      if acc.isEmpty then floatToJson x else acc ++ "," ++ floatToJson x)
    ""
  "[" ++ body ++ "]"

/-- Format a Schedule (Array BimodalPhi) as a JSON array of [Phi_B,
    Phi_S] pairs. -/
private def formatSchedule (sched : Schedule) : String :=
  let body := sched.foldl
    (fun (acc : String) (b : BimodalPhi) =>
      let pair := "[" ++ floatToJson b.Phi_B ++ "," ++ floatToJson b.Phi_S ++ "]"
      if acc.isEmpty then pair else acc ++ "," ++ pair)
    ""
  "{\"schedule\":[" ++ body ++ "]}"

/-- Format the next-state output from a plant emStep call. -/
private def formatNextState (y : State6D) : String :=
  "{\"next_state\":[" ++ floatToJson y.B ++ "," ++ floatToJson y.S ++ "," ++
    floatToJson y.F ++ "," ++ floatToJson y.A ++ "," ++ floatToJson y.KFB ++ "," ++
    floatToJson y.KFS ++ "]}"

private def formatError (msg : String) : String :=
  "{\"error\":" ++ (Json.str msg).compress ++ "}"

/-- Process a single JSON request → output string (one line of JSON). -/
def handleRequest (input : Json) : Except String String := do
  let fn ← input.getObjValAs? String "fn"
  match fn with
  | "drift" =>
    let state  ← (← input.getObjVal? "state")  |> getStateFromArr
    let phi    ← (← input.getObjVal? "phi")    |> getPhiFromArr
    let params ← (← input.getObjVal? "params") |> getParams
    let dy := drift state params phi
    pure (formatDeriv dy)
  | "muBar" =>
    -- Inputs: scalar A, BimodalPhi, Params. Output: scalar mu_bar.
    let A      ← getFloat input "A"
    let phi    ← (← input.getObjVal? "phi")    |> getPhiFromArr
    let params ← (← input.getObjVal? "params") |> getParams
    pure (formatScalar "muBar" (muBar A phi params))
  | "findASep" =>
    -- Inputs: BimodalPhi, Params. Output: scalar A_sep (may be ±inf).
    let phi    ← (← input.getObjVal? "phi")    |> getPhiFromArr
    let params ← (← input.getObjVal? "params") |> getParams
    pure (formatScalar "A_sep" (findASep phi params))
  | "schedule" =>
    -- Inputs: theta (RBF coeffs), Phi_design matrix, c_phi scalar,
    -- phi_max scalar, n_anchors. Output: Schedule (Array BimodalPhi).
    let theta     ← (← input.getObjVal? "theta") |> getFloatArr
    let phiDesign ← (← input.getObjVal? "phi_design") |> getMatrix
    let cPhi      ← getFloat input "c_phi"
    let phiMax    ← getFloat input "phi_max"
    let nAnchors  ← input.getObjValAs? Nat "n_anchors"
    pure (formatSchedule
      (scheduleFromTheta theta phiDesign cPhi phiMax nAnchors))
  | "emStep" =>
    -- Inputs: state, phi, params, sigma_diag (6 floats), dt, noise (6 floats).
    -- Output: next_state (6 floats).
    let state     ← (← input.getObjVal? "state")  |> getStateFromArr
    let phi       ← (← input.getObjVal? "phi")    |> getPhiFromArr
    let params    ← (← input.getObjVal? "params") |> getParams
    let sigmaDiag ← (← input.getObjVal? "sigma_diag") |> getFloatArr
    let dt        ← getFloat input "dt"
    let noise     ← (← input.getObjVal? "noise") |> getFloatArr
    let yNext := emStep state phi params sigmaDiag dt noise
    pure (formatNextState yNext)
  | "hrMean" =>
    let state    ← (← input.getObjVal? "state")  |> getStateFromArr
    let C        ← getFloat input "C"
    let opParams ← (← input.getObjVal? "obs_params") |> getObsParams
    pure (formatScalar "hr_mean" (hrMean state C opParams))
  | "sleepProb" =>
    let state    ← (← input.getObjVal? "state")  |> getStateFromArr
    let C        ← getFloat input "C"
    let opParams ← (← input.getObjVal? "obs_params") |> getObsParams
    pure (formatScalar "sleep_prob" (sleepProb state C opParams))
  | "stressMean" =>
    let state    ← (← input.getObjVal? "state")  |> getStateFromArr
    let C        ← getFloat input "C"
    let opParams ← (← input.getObjVal? "obs_params") |> getObsParams
    pure (formatScalar "stress_mean" (stressMean state C opParams))
  | "stepsLogMean" =>
    let state    ← (← input.getObjVal? "state")  |> getStateFromArr
    let C        ← getFloat input "C"
    let opParams ← (← input.getObjVal? "obs_params") |> getObsParams
    pure (formatScalar "steps_log_mean" (stepsLogMean state C opParams))
  | "volumeLoadMean" =>
    let state    ← (← input.getObjVal? "state")  |> getStateFromArr
    let opParams ← (← input.getObjVal? "obs_params") |> getObsParams
    pure (formatScalar "vl_mean" (volumeLoadMean state opParams))
  | other => Except.error s!"unknown fn: {other}"

/-- Process a single line of input → write a single line of output. -/
def processLine (line : String) : IO Unit := do
  match Json.parse line with
  | Except.error e =>
    IO.println (formatError s!"parse: {e}")
  | Except.ok j =>
    match handleRequest j with
    | Except.error e =>
      IO.println (formatError e)
    | Except.ok out =>
      IO.println out

/-- Main loop: read stdin line-by-line, process each line, flush after each.
    Designed for the Python `subprocess.Popen` long-lived worker pattern. -/
def main : IO Unit := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  let mut keepGoing := true
  while keepGoing do
    let line ← stdin.getLine
    if line.isEmpty then
      keepGoing := false
    else
      processLine line
      stdout.flush
