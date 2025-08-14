This github repository is supplementary to the final project done by Maya Rozenshtein and Sergiy Horef in the course - Language, Computation and Cognition (096222) taught at Technion, Israel in Spring 2025.

The files in the repository are organized as follows:
- [structured_part.ipynb](./structured_part.ipynb) is a notebook which contains the code (with cell outputs) for all relevant questions in the structured part of the project.
- [open_part.ipynb](./open_part.ipynb) is a notebook which contains the code (with cell outputs) for all the relevant results of the open part of the project.
- [learn_decoder.py](./learn_decoder.py) contains the functions used to learn the decoder for the structured part.
- [encoder.py](./encoder.py) contains different encoders we have tested and the auxillary functions to train and test them.
- [outputs](./outputs) contains trained models, saved figures and processed data as saved throughout different parts of the code. If needed, please consult the naming conventions used in the code to find the relevant file.
    - [Sentence Reconstruction Results using Glove](./outputs/sentence_reconstruction_results_glove.txt) is the txt file containing the results of the sentence reconstruction of the selected 50 sentences using GloVe embeddings.
    - [Sentence Reconstruction Results using LLM](./outputs/sentence_reconstruction_results_llm.txt) is the txt file containing the results of the sentence reconstruction of the selected 50 sentences using LLM embeddings.