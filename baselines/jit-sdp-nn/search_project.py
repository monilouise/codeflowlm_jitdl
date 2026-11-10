import os
import shutil

def copiar_por_projeto(project_name):
    base_dir = "mlruns/0"
    #base_dir = "mlruns_all_jit_defects4j"
    destino_base = f"mlruns/por_projeto/{project_name}"

    for root, dirs, files in os.walk(base_dir):
        if "dataset" in files:
            dataset_path = os.path.join(root, "dataset")
            try:
                with open(dataset_path, "r") as f:
                    conteudo = f.read().strip()
                if conteudo == project_name:
                    root2 = root[:root.rfind(os.sep)]
                    rel_path = os.path.relpath(root2, base_dir)
                    destino_dir = os.path.join(destino_base, rel_path)
                    shutil.copytree(root2, destino_dir, dirs_exist_ok=True)
            except Exception as e:
                print(f"Erro ao processar {dataset_path}: {e}")

if __name__ == "__main__":
    nome_projeto = "commons-math"  # Substitua pelo nome do projeto desejado
    copiar_por_projeto(nome_projeto)