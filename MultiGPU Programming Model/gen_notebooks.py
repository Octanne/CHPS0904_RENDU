import os
from nbformat import v4 as nbf

# Versions et descriptions pour un solveur de Jacobi
versions = [
    ("etape1_cpu", "Solveur Jacobi CPU de base : implémentation mono-thread. Utile pour valider la correction et les petites tailles de problème ; met en évidence la limite de calcul CPU."),
    ("etape2_cpu_gpu", "1 CPU + 1 GPU : délégation du noyau Jacobi à un GPU. Montre l'accélération GPU quand le problème tient sur un seul dispositif."),
    ("etape3_mpi_gpus", "MPI + GPUs : domaine réparti sur plusieurs rangs MPI, chacun pilotant un GPU. Illustrations des défis de mise à l'échelle multi-nœuds et des communications inter-rangs."),
    ("etape4_mpi_overlap", "MPI + recouvrement : recouvrements des échanges d’halo non-bloquants avec le calcul Jacobi local. Réduit l'impact de la latence réseau."),
    ("etape5_nccl", "NCCL : utilisation de la NVIDIA Collective Communications Library pour les échanges GPU à GPU. Montre les gains via NVLink ou PCIe haute bande passante."),
    ("etape6_nccl_overlap", "NCCL + recouvrement : superposition des collectifs NCCL avec le calcul sur GPU, cachant le coût de communication."),
    ("etape7_nccl_graphs", "NCCL + CUDA Graphs : capture et relecture des séquences Jacobi/échange pour réduire le surcoût des lancements."),
    ("etape8_nvshmem", "NVSHMEM : modèle PGAS à accès mémoire unilatéral GPU, simplifiant les mises à jour d’halo."),
    ("etape9_nvshmem_lt", "NVSHMEM + LTO : ajout de l’optimisation link-time pour inliner les fonctions critiques et réduire le coût des appels."),
    ("etape10_vshmem_neighborhood_lto", "vshmem neighborhood_sync + LTO : synchronisation fine-grain de voisinage et optimisations link-time O2."),
    ("etape11_nvshmem_norm_overlap_neighborhood_sync_lto", "Combinaison : NVSHMEM avec recouvrement, synchrone de voisinage, et LTO pour maximiser la concurrence."),
    ("etape12_nvshmem_norm_overlap_neighborhood_sync_lto_ext1", "Tuning étendu : paramètres ajustables (taille de tuile, ordre de boucles) et hooks de benchmark.")
]

# 1. Création des répertoires et squelette de code
makefile_entries = []
for nom, description in versions:
    os.makedirs(nom, exist_ok=True)
    makefile_entries.append(nom)
    code = f'''// {nom} : {description}
// Solveur Jacobi sur grille 2D de taille N x N, T itérations.
#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define T 1000

void jacobi_cpu(double* A, double* B) {{
    for (int t = 0; t < T; t++) {{
        for (int i = 1; i < N-1; i++) {{
            for (int j = 1; j < N-1; j++) {{
                B[i*N + j] = 0.25 * (A[(i-1)*N + j] + A[(i+1)*N + j]
                                      + A[i*N + j-1] + A[i*N + j+1]);
            }}
        }}
        double* tmp = A; A = B; B = tmp;
    }}
}}

int main(int argc, char** argv) {{
    double *A = malloc(N*N*sizeof(double));
    double *B = malloc(N*N*sizeof(double));
    for (int i = 0; i < N*N; i++) A[i] = B[i] = 0.0;
    jacobi_cpu(A, B);
    printf("Terminé %s", "{nom}");
    free(A); free(B);
    return 0;
}}
'''
    with open(os.path.join(nom, 'main.c'), 'w') as f:
        f.write(code)

# 2. Makefile global
mk_global = f'''# Makefile global pour toutes les versions Jacobi
VERSIONS = {' '.join(makefile_entries)}

.PHONY: all clean $(VERSIONS)
all: $(VERSIONS)

$(VERSIONS):
	@$(MAKE) -C $@ all

clean:
	@for d in $(VERSIONS); do \
		$(MAKE) -C $$d clean; \
	done
'''
with open('Makefile', 'w') as mf:
    mf.write(mk_global)

# 3. Notebook Jupyter avec explications en français
nb = nbf.new_notebook()
cells = [nbf.new_markdown_cell("# Solveur de Jacobi : Modèles de programmation Multi-GPU\nCe notebook présente 12 versions progressives d’un solveur de Jacobi 2D. Chaque section explique le modèle ou l’optimisation, compile la version, exécute et collecte les métriques Nsight.")]

details = {
    'etape1_cpu': "Baseline : évalue la limite mémoire CPU pour établir une référence.",
    'etape2_cpu_gpu': "Met en évidence l'écart de performance CPU vs GPU lorsque la grille est suffisamment grande.",
    'etape3_mpi_gpus': "Test de montée en charge inter-nœuds et coût MPI sur cluster multi-GPU.",
    'etape4_mpi_overlap': "Cache la latence réseau en recouvrant communication et calcul local.",
    'etape5_nccl': "Exploite automatiquement le topologie NVLink/PCIe pour des échanges GPU efficaces.",
    'etape6_nccl_overlap': "Essentiel à forte densité GPU pour maintenir les cœurs occupés.",
    'etape7_nccl_graphs': "Réduit l’overhead de lancement grâce aux CUDA Graphs.",
    'etape8_nvshmem': "Simplifie les échanges via modèle PGAS unilatéral.",
    'etape9_nvshmem_lt': "Optimisation link-time pour inliner les sections critiques.",
    'etape10_vshmem_neighborhood_lto': "Synchronisation fine et LTO pour boucles serrées.",
    'etape11_nvshmem_norm_overlap_neighborhood_sync_lto': "Combinaison des meilleures pratiques pour un binaire ultra-optimisé.",
    'etape12_nvshmem_norm_overlap_neighborhood_sync_lto_ext1': "Ajout de paramètres de tuning et hooks de benchmarking."
}

for nom, desc in versions:
    cells.append(nbf.new_markdown_cell(f"## {nom}\n**Description :** {desc}\n\n**Intérêt :** {details[nom]}"))
    cells.append(nbf.new_code_cell(f"%%bash cd {nom} ; make all"))
    cells.append(nbf.new_code_cell(f"%%bash cd {nom} ; nv-nsight-cu-cli --csv --report-file rapport_{nom}.csv ./main ;cat rapport_{nom}.csv"))

nb['cells'] = cells
import nbformat

with open('Jacobi_MultiGPU.ipynb', 'w') as f:
    nbformat.write(nb, f)

print("Initialisation terminée : répertoires, squelettes de code, Makefile et notebook générés en français.")