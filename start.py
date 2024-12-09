from interface import iFace
import subprocess

def start_kg_program():
    
    try:
        print("Building Docker Image...")
        result=subprocess.run(["bash", "script/docker_run.sh"], capture_output=True, text=True, check=True)
        # print(result.stdout)
        print("Launching Interface...")
        iFace()
    except subprocess.CalledProcessError as e:
        print(f"Erreur : {e.stderr}")
