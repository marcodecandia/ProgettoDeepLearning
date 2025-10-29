#  Object-Level Visual Search and Scene Analysis with Foundational Models

##  Descrizione del progetto

Progetto realizzato per l'appello del 2 settembre 2025 del Corso di Deep Learning (LM Ingegneria Informatica).

Prof. Ing. Vito Walter Anelli      
Studenti: Marcantonio de Candia, Nicola Cipriani
---

## Struttura del progetto

```bash
ProgettoDeepLearning/
│
├── indexing/
│   ├── clip_embedding.py           # Gestione embedding CLIP (base e fine-tuned)
│   ├── dino_embedding.py           # Gestione embedding DINOv2
│   └── search_logic.py             # Logica di ricerca e confronto (FAISS / naive)
│   ├── image_loader.py             # Caricamento e preprocessing immagini
│   ├── indexing_pipeline_all.py    # Pipeline di indicizzazione completa             
│
├── scene_analysis/
│   ├── mtcnn_masks.py              # Rilevamento volti con MTCNN
│   ├── sam_masks.py                # Segmentazione automatica con SAM
│   ├── scene_analysis_pipeline_mtcnn.py  # Pipeline di analisi scena con MTCNN
│   └── scene_analysis_pipeline_sam.py    # Pipeline di analisi scena con SAM
│

├── embeddings/                     # Database di embedding precomputati
│   ├── clip_faiss.index
│   ├── clip_faiss_metadata.pkl
│   ├── clip_finetuned_faiss.index
│   ├── clip_finetuned_faiss_metadata.pkl
│   ├── clip_finetuned_python_list.pkl
│   ├── clip_python_list.pkl
│   ├── dino_faiss.index
│   ├── dino_faiss_metadata.pkl
│   └── dino_python_list.pkl
│
├── models/                         # Modelli addestrati
│   ├── sam_vit_b_01ec64.pth        # Modello SAM versione light pre-addestrato
│   └── trained_clip_model.pth      # Modello CLIP fine-tuned
│
├── data/                           # Dataset organizzato per classi
│   ├── train/
│   ├── test/
│   ├── valid/
│   └── complex_scenes/
│
├── training/                        # Sezione dedicata al fine-tuning del modello CLIP  
│   ├── data_augmentation.py         # Funzioni per la data augmentation
│   ├── training_loop.py             # Loop di addestramento del modello CLIP
│   └── valid_loop.py                # Loop di validazione del modello CLIP
│
├── testing/                        # Sezione dedicata al testing del modello  
│   ├── test_query.py              # Script di test per query di ricerca
│   ├── test_sceneanalysis.py      # Script di test per analisi di scena
│   └── test_utils.py              # Script contenente funzioni di utilità per i test
│
├── user_interface/                  
│   └── user_interface.py              # Interfaccia utente Gradio principale
│
├── requirements.txt                # Dipendenze Python
└── README.md                       # Questo file
```

##  Esecuzione del Progetto

Per avviare l’interfaccia Gradio principale:
```bash
python user_interface/user_interface.py
```