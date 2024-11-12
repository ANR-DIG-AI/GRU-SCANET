Prerequisites : 
    - Linux installed
    - Python (>=3.8) installed
# Environment :

    - Go to this page to install pytorch : https://pytorch.org/get-started/locally/
    - Clone the code
# Download the data to replace the existing ones at [Data Link](https://drive.google.com/file/d/1vPMGqSp9sk8-eiVzjSUkfzCdfnjU8bx0/view?usp=sharing)
# GRU-SCANET :

   `$ sh ./job.sh `

# To evaluate GRU-SCANET without MHA : 

    apply the comment on line 138 of the file module/model.py and re-run the previous command.

# Show results : 

    Open /result/logs/logs.txt to get results of the evaluations.
