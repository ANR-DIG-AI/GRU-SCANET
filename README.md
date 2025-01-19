# GRU-SCANET: Named Entity Recognition Architecture

GRU-SCANET is a state-of-the-art architecture for Named Entity Recognition (NER) that combines several advanced techniques to achieve high performance. Its design includes:
- **Positional Encoding**: To capture the position of tokens within a sequence.
- **Bidirectional GRU**: To model contextual dependencies in both forward and backward directions.
- **Multi-Head Attention**: To focus on relevant parts of the sequence.
- **CRF Decoder**: To optimize sequence labeling by considering global label dependencies.

## Features
- Evaluated on 8 datasets from **BioCreative**, targeting the detection of the following entity types:
  - **Gene/Protein**
  - **Disease**
  - **Drug/Chemical**
  - **Species**
  - **DNA**
- Robust performance across multiple datasets and domains in biomedical text.

## Installation
1. Prerequisites : 
    - Linux installed
    - Python (>=3.8) installed
    - Go to this page to install pytorch : https://pytorch.org/get-started/locally/
    - Clone the code
    - Download the data to replace the existing ones at [Data Link](https://drive.google.com/file/d/1vPMGqSp9sk8-eiVzjSUkfzCdfnjU8bx0/view?usp=sharing)

2. Clone the repository:
   ```bash
   git clone https://github.com/ANR-DIG-AI/GRU-SCANET
   cd GRU-SCANET
3. Install dependencies

    ``` bash 
    pip install -r requirements.txt
## Usage
### Training
To train the GRU-SCANET model on the provided datasets:

    sh ./job.sh

### Testing
To evaluate the trained model on a single text value:

    python gruscanet_test.py --text "The antipsychotic agent , remoxipride [ ( S ) - ( - ) - 3 - bromo - N - [ ( 1 - ethyl - 2 - pyrrolidinyl ) methyl ] - 2 , 6 - dimethoxybenz amide ] has been associated with acquired aplastic anemia ."
    
    Output: 

    O O O O B0 O B0 I0 I0 I0 I0 I0 I0 I0 I0 I0 I0 I0 I0 I0 I0 I0 I0 I0 I0 I0 I0 I0 I0 I0 I0 I0 I0 I0 I0 I0 I0 I0 I0 O O O O O O O O O

    Mapping: 

    B: Beginning, I: Inside, O: Outside .
    
    {
        "Drug/Chem": 0 or 6,
        "Species": 1 or 5,
        "Disease": 2 or 7,
        "Gene/Protein": 3,
        "DNA": 4
    } 
    
    NB: two values are used because each entity type were automated from two datasets associated to the same type.

    - B0 indicates that the associated word or token is the beginning of an entity of type Drug/Chem.

    - I0 indicates that the word or token is inside an entity of type Drug/Chem.

    - O indicates that the associated word or token is outside any typed entity.



## Contributions
Contributions are welcome! Feel free to fork the repository and submit a pull request for improvements or new features. Contact us at bill.happi@ird.fr .

## License
This project is licensed under the MIT License.

