import subprocess
import concurrent.futures
MAIS = 'MAIS.py'
OTHGN = 'Run_exact.py'
OSIA = "Journal_Run_OSIA.py"
OVIA = 'Journal_Run_OVIA.py'
SSIA = 'Journal_Run_SSIA.py'
SVIA = 'Journal_Run_SVIA.py'

# num_nodes
args = ['../data/new_data_di_ER_6_0.4_1000_test_chromatic','6']


commands = [MAIS, OTHGN, OSIA, OVIA, SSIA, SVIA]
#commands = [OVIA]
# for command in commands:
#     print("Running:", command)
#     subprocess.run(["python", command] + args)

# Parallel running
def run_command(command):
    print("Running:", command)
    subprocess.run(["python", command] + args)

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(run_command, commands)


