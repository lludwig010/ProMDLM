first run prepare.py to generate the pkl array of sequences ready to go.
then train using train_custom.py

### Sequence Diversity Evaluation using Blastp

Blastp is a tool for aligning and comparing protein sequences. We use Blastp to evaluate the diversity of generated sequences by aligning them to the training set. We here provide a brief tutorial to reproduce that analysis. 

#### Crreating a dataset to align to based on the training set 

```bash
makeblastdb -in training_dataset.fasta -dbtype prot -out training_db