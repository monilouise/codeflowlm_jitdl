import subprocess

def execute_command(command):
    print(command)
    process = subprocess.Popen(command.strip(), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')

    for line in iter(process.stdout.readline, ''):
        print(line, end='', flush=True)

    process.stdout.close()
    process.wait()