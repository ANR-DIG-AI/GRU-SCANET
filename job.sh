# pip install -r requirements.txt
# pip install -r _requirements.txt
cd ./ner/
python3.8 ./data_processing.py > ./result/event/log.txt
python3.8 ./biogru.py >> ./result/event/log.txt