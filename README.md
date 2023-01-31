# AI-Surrogate-Modelling

AI-accelerated surrogate model for Aqueous Solubility prediction
Built in an hour or two to aid understanding for an interview - so do not judge the code so harshly!

There exist very clever networks in this industry implementing 3D molecule spacing / localisation etc**. Conversely, this is a very simple Multi-Layer feedforward network with no manual bias for inputs. More novel techniques can be added but initially just implemented a basic network in this field.

Data from:
https://www.kaggle.com/datasets/sorkun/aqsoldb-a-curated-aqueous-solubility-dataset

---

## To continue:
1. Graph test / train loss every epoch to determine optimum epoch.
1. Model scores well but test loss varies decently if retrained.
1. Change model architecture to something interesting - perhaps resolving #2. Repeat 1


** See [ANI-1](http://xlink.rsc.org/?DOI=c6sc05720a), which itself cites basic MLP models.
