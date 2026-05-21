import os
import subprocess
import random
import time

def random_params():
    return {
        "TUNER_QOL_TARGET": str(random.randint(10, 25)),
        "TUNER_PROD_TARGET": str(random.randint(10, 20)),
        "TUNER_PG_TARGET": str(random.randint(10, 20)),
        "TUNER_ENV_LOW": str(random.randint(8, 15)),
        "TUNER_ENV_HIGH": str(random.randint(16, 25)),
        "TUNER_SAN_TARGET": str(random.randint(20, 29)),
        "TUNER_MCTS_QOL_LIMIT": str(random.randint(10, 20)),
        "TUNER_MCTS_POL_LIMIT": str(random.randint(5, 15)),
        "TUNER_MCTS_ENV_LIMIT": str(random.randint(10, 20)),
        "TUNER_MCTS_EDU_LIMIT": str(random.randint(10, 20))
    }

def run_tuner():
    log_file = os.path.join(os.path.dirname(__file__), "best_hyperparameters.log")
    print(f"Starting Hyperparameter Tuner. Writing results > 20 to {log_file}")

    with open(log_file, "a") as f:
        f.write("=== Hyperparameter Tuning Session Started ===\n")

    while True:
        params = random_params()

        env = os.environ.copy()
        env.update(params)
        env["PYTHONUTF8"] = "1"

        try:
            tester_path = os.path.join(os.path.dirname(__file__), "SOVEREIGN_JULES_TESTER.py")
            result = subprocess.run(
                [sys.executable, tester_path],
                env=env,
                capture_output=True,
                text=True,
                timeout=30 # Prevent hangs
            )

            output = result.stdout

            score_line = next((line for line in output.split('\n') if '> Average Stability Score:' in line), None)

            if score_line:
                score = float(score_line.split(':')[1].strip())
                if score > 20:
                    log_entry = f"Score: {score:.2f} | Params: {params}\n"
                    print(f"FOUND MATCH! {log_entry.strip()}")
                    with open(log_file, "a") as f:
                        f.write(log_entry)
        except Exception as e:
            print(f"Error during run: {e}")
            time.sleep(1)

if __name__ == "__main__":
    import sys
    run_tuner()
