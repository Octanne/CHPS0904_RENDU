# CHPS0904_RENDU

Ce rendu regroupe trois activités distinctes, chacune illustrant une approche différente de l'accélération sur GPU, adaptées à l'environnement ROMEO2025 :

## 1. GPU_Stream
Cette activité propose des exercices pour découvrir la programmation CUDA de base et la gestion des streams GPU. Elle permet de comprendre comment exploiter la parallélisation fine sur un seul GPU.

## 2. MultiGPU Programming Model
Cette partie explore les modèles de programmation multi-GPU à travers une progression de versions du solveur de Jacobi. On y découvre les techniques de distribution de calcul, de communication inter-GPU (MPI, NCCL, NVSHMEM), et d'optimisation de la concurrence sur plusieurs GPUs.

## 3. OpenACC
Cette activité initie à la programmation directive avec OpenACC, permettant d'accélérer des applications scientifiques sans réécriture lourde du code, en ciblant les GPU de ROMEO2025.

Chaque dossier est conçu pour être exécuté sur l'infrastructure ROMEO2025 et permet de se familiariser avec différentes méthodes d'accélération sur GPU, du plus bas niveau (CUDA) au plus haut niveau (OpenACC), en passant par la programmation multi-GPU.
