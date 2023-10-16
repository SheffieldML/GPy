import os
import subprocess


python_files = [file for file in os.listdir() if file.endswith(".py")]

python_test_files = [file for file in python_files if "test" in file]
non_test_python_files = [file for file in python_files if "test" not in file]
print("Python Test Files: ", python_test_files)

print("Non-test Python Files:\n", non_test_python_files)

for file in python_test_files:
    if file.endswith("_tests.py"):
        test_name = file.split("_tests.py")[0]
    elif file.endswith("_test.py"):
        test_name = file.split("_test.py")[0]
    else:
        raise ValueError(f"File is not named as expected: {file}")

    to_file = "test_" + test_name + ".py"

    # print(" ".join(["git", "mv", "-f", file, to_file]))
    subprocess.run(["git", "mv", "-f", file, to_file])
